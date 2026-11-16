# -*- coding: utf-8 -*-
"""知识库文档管理（阶段一）：文件名校验、文档枚举、内容去重、安全删除。

对应《my-ai 25 阶段一实施前设计审查 V1.6》第 2/3/4 节。设计边界：

- 只管理知识库目录**顶层**的白名单后缀（``.md`` / ``.txt`` / ``.pdf`` / ``.docx``）
  普通文件，不递归；
- ``document_id = sha256(filename)[:32]``，仅作**接口标识**，不是权限凭证：
  删除目标名只来自 ``os.scandir`` 的枚举结果，``document_id`` 永不参与路径拼接；
- 符号链接 / junction / 其它重解析点 / 目录 / 设备一律拒绝；
- 上传用 ``O_CREAT|O_EXCL`` 原子创建；写入失败回滚前用 ``(st_dev, st_ino)`` 复核身份，
  身份无法确认时**不删除**，如实报告残留；
- 不引入数据库、manifest、缓存系统或第三方依赖。

Windows 能力边界（已实测）：``os.O_NOFOLLOW`` 不存在，也没有「按目录句柄原子删除」的
标准库原语，因此检查与操作之间无法做到原子；本模块只保证「检测到不一致就拒绝」。
"""

import hashlib
import logging
import os
import re
import stat
import threading
from datetime import datetime
from pathlib import Path

import doc_formats


logger = logging.getLogger(__name__)

# 与 main.ALLOWED_KNOWLEDGE_SUFFIXES 保持同一口径：两者都来自 doc_formats。
# 保留原变量名，兼容既有调用与测试。
ALLOWED_SUFFIXES = doc_formats.SUPPORTED_SUFFIXES

# sha256 十六进制前 32 位。
DOCUMENT_ID_HEX_LENGTH = 32

_DOCUMENT_ID_RE = re.compile(r"^[0-9a-f]{%d}$" % DOCUMENT_ID_HEX_LENGTH)

# Windows 保留设备名（不含扩展名的基名比较）。
_RESERVED_DEVICE_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + ["COM%d" % index for index in range(1, 10)]
    + ["LPT%d" % index for index in range(1, 10)]
)

# 文件名中禁止出现的字符（Windows 非法字符）。
_FORBIDDEN_NAME_CHARS = frozenset('<>:"|?*')

# 仅保护上传/删除的文件系统变更临界区；不覆盖索引构建（索引侧由
# KnowledgeBase._init_lock 独立保护），两者互不替代。
_MUTATION_LOCK = threading.Lock()


# ============================================================
# 异常：统一携带 HTTP 语义，由 main 映射为 HTTPException
# ============================================================


class KnowledgeStoreError(Exception):
    """知识库管理操作的业务异常基类。"""

    status_code = 500
    error_type = "knowledge_store_error"

    def __init__(self, message, **extra):
        super().__init__(message)
        self.message = message
        self.extra = extra


class InvalidDocumentId(KnowledgeStoreError):
    status_code = 400
    error_type = "invalid_document_id"


class DocumentNotFound(KnowledgeStoreError):
    status_code = 404
    error_type = "document_not_found"


class ReparsePointRejected(KnowledgeStoreError):
    status_code = 403
    error_type = "reparse_point_rejected"


class NotRegularFile(KnowledgeStoreError):
    status_code = 403
    error_type = "not_regular_file"


class ConcurrentModification(KnowledgeStoreError):
    status_code = 409
    error_type = "concurrent_modification"


class FileBusy(KnowledgeStoreError):
    status_code = 409
    error_type = "file_busy"


class AlreadyExists(KnowledgeStoreError):
    status_code = 409
    error_type = "already_exists"


class DuplicateContent(KnowledgeStoreError):
    status_code = 409
    error_type = "duplicate_content"


class WriteFailed(KnowledgeStoreError):
    status_code = 500
    error_type = "write_failed"


class CreateRollbackFailed(KnowledgeStoreError):
    status_code = 500
    error_type = "rollback_failed"


class DeleteFailed(KnowledgeStoreError):
    status_code = 500
    error_type = "delete_failed"


class UnexpectedPathState(KnowledgeStoreError):
    status_code = 500
    error_type = "unexpected_path_state"


class KnowledgeDirUntrusted(KnowledgeStoreError):
    status_code = 503
    error_type = "knowledge_dir_untrusted"


class KnowledgeDirInvalid(KnowledgeStoreError):
    status_code = 500
    error_type = "knowledge_dir_invalid"


# ============================================================
# 基础判定
# ============================================================


def compute_document_id(filename):
    """由文件名派生文档标识：sha256(filename)[:32]。

    与文件系统大小写行为无关：id 只由 ``scandir`` 返回的**存储名**计算，
    因此同一物理文件只会得到一个 id；改名（含仅改大小写）即换 id。
    """
    if not isinstance(filename, str):
        raise TypeError("filename 必须是字符串")
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:DOCUMENT_ID_HEX_LENGTH]


def _is_reparse_point(st):
    """Windows 重解析点判定（覆盖 symlink / junction / 其它 tag）。"""
    attribute = getattr(st, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag) and bool(attribute & flag)


def _suffix_of(filename):
    return os.path.splitext(filename)[1].lower()


def _is_allowed_document_name(filename):
    return bool(filename) and _suffix_of(filename) in ALLOWED_SUFFIXES


def validate_knowledge_filename(filename):
    """校验规范化后的文件名；返回 None 表示通过，否则返回错误类型字符串。

    只拒绝「在 Windows 上会失败或危险」的名字，不做大小写规范化
    （大小写语义交给文件系统）。
    """
    if not isinstance(filename, str) or not filename:
        return "invalid_filename"
    if filename in (".", ".."):
        return "invalid_filename"
    if any(char in _FORBIDDEN_NAME_CHARS for char in filename):
        return "invalid_filename"
    if any(ord(char) < 32 for char in filename):
        return "invalid_filename"
    if filename != filename.rstrip(" ."):
        return "invalid_filename"
    if filename.split(".")[0].upper() in _RESERVED_DEVICE_NAMES:
        return "invalid_filename"
    return None


def inspect_knowledge_directory(directory):
    """检查知识库叶子目录，返回状态字符串。

    返回值：
    - ``"missing"``       路径不存在
    - ``"untrusted"``     叶子目录是重解析点（symlink / junction / 其它 tag）
    - ``"not_directory"`` 存在但不是目录（普通文件、设备等）
    - ``"inaccessible"``  存在但无法访问（权限等 I/O 异常）
    - ``"ok"``            真实目录

    必须用 ``lstat``：``exists()`` / ``is_dir()`` 会跟随重解析点，悬空链接还会
    被误判为不存在，从而绕过信任检查。只检查叶子目录，不检查祖先路径
    （``Desktop`` 等可能因 OneDrive 重定向正常地是重解析点）。
    """
    try:
        st = os.lstat(directory)
    except FileNotFoundError:
        return "missing"
    except NotADirectoryError:
        return "missing"
    except PermissionError:
        return "inaccessible"
    except OSError:
        return "inaccessible"

    if _is_reparse_point(st):
        return "untrusted"
    if not stat.S_ISDIR(st.st_mode):
        return "not_directory"
    return "ok"


def _timestamp(seconds):
    try:
        return datetime.fromtimestamp(seconds).astimezone().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _created_at(st):
    """返回 (created_at, created_at_source)。

    ``created_at`` 是 **best-effort** 创建时间：有 ``st_birthtime`` 时用它，
    否则退回 ``st_ctime`` —— 后者在 POSIX 上表示 inode 变更时间而非创建时间，
    因此必须由 ``created_at_source`` 如实标注来源。
    """
    birthtime = getattr(st, "st_birthtime", None)
    if birthtime:
        return _timestamp(birthtime), "birthtime"
    ctime = getattr(st, "st_ctime", None)
    if ctime:
        return _timestamp(ctime), "ctime_fallback"
    return None, "unavailable"


# ============================================================
# 枚举
# ============================================================


def _build_metadata(filename, suffix, st):
    created_at, created_at_source = _created_at(st)
    return {
        "document_id": compute_document_id(filename),
        "filename": filename,
        "file_type": suffix.lstrip("."),
        "file_size": st.st_size,
        "created_at": created_at,
        "created_at_source": created_at_source,
        "updated_at": _timestamp(st.st_mtime),
        "storage_type": "persistent",
        # 由接口层按 with_index 填充；默认不读取索引状态。
        "parse_status": "unknown",
        "chunk_count": None,
    }


def list_documents(directory):
    """列出知识库顶层可管理文档。

    目录不存在不是错误：返回 ``directory_exists=False`` 与空列表（已确认决策）。
    重解析点目录 / 非目录 / 不可访问 → 抛对应异常，由接口层映射 503 / 500。
    """
    state = inspect_knowledge_directory(directory)

    if state == "missing":
        return {"directory_exists": False, "documents": []}
    if state == "untrusted":
        raise KnowledgeDirUntrusted(
            "知识库目录是重解析点（junction/符号链接），拒绝以该目录为安全基准",
            path=str(directory),
        )
    if state in ("not_directory", "inaccessible"):
        raise KnowledgeDirInvalid(
            "知识库路径不可用（不是目录或无法访问）",
            path=str(directory),
        )

    documents = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if not _is_allowed_document_name(entry.name):
                    continue
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if _is_reparse_point(st):
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                documents.append(_build_metadata(entry.name, _suffix_of(entry.name), st))
    except OSError as exc:
        raise KnowledgeDirInvalid(
            "知识库目录无法读取", path=str(directory), reason=type(exc).__name__
        ) from exc

    documents.sort(key=lambda item: item["filename"])
    return {"directory_exists": True, "documents": documents}


def _scan_delete_index(directory):
    """建立 {document_id: (filename, kind)}；kind ∈ {regular, reparse, other}。

    只收白名单后缀条目：管理接口不应成为「任意文件删除器」。
    """
    index = {}
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if not _is_allowed_document_name(entry.name):
                    continue
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if _is_reparse_point(st):
                    kind = "reparse"
                elif stat.S_ISREG(st.st_mode):
                    kind = "regular"
                else:
                    kind = "other"
                index[compute_document_id(entry.name)] = (entry.name, kind)
    except OSError as exc:
        raise KnowledgeDirInvalid(
            "知识库目录无法读取", path=str(directory), reason=type(exc).__name__
        ) from exc
    return index


# ============================================================
# 内容去重
# ============================================================


def find_duplicate_content(directory, content, exclude_filename=None):
    """返回与 content **原始字节**相同的已有文件名；没有则返回 None。

    不做任何文本规范化：不转换换行符、不折叠空白、不剥离 Markdown 语法，
    也不以文件大小代替内容哈希。符号链接与非普通文件不参与比对。
    """
    target_digest = hashlib.sha256(content).hexdigest()

    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if not _is_allowed_document_name(entry.name):
                    continue
                if exclude_filename is not None and entry.name == exclude_filename:
                    continue
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if _is_reparse_point(st) or not stat.S_ISREG(st.st_mode):
                    continue
                try:
                    with open(entry.path, "rb") as handle:
                        existing = handle.read()
                except OSError:
                    continue
                if hashlib.sha256(existing).hexdigest() == target_digest:
                    return entry.name
    except OSError:
        return None
    return None


# ============================================================
# 创建（含失败回滚）
# ============================================================


def _write_all(fd, content):
    view = memoryview(content)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("写入未取得进展")
        view = view[written:]


def _rollback_created(path, st0):
    """尝试删除本函数创建的文件；返回 True 表示仍有残留。

    只有确认目标仍是本次创建的那个普通文件时才删除：身份不一致、类型变化、
    是重解析点、或无法确认（``st0`` 缺失 / ``lstat`` 失败）一律不删。
    """
    if st0 is None:
        return True
    try:
        current = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        return True

    if _is_reparse_point(current):
        return True
    if not stat.S_ISREG(current.st_mode):
        return True
    if (current.st_dev, current.st_ino) != (st0.st_dev, st0.st_ino):
        return True

    try:
        os.unlink(path)
    except OSError:
        return True
    return False


def create_document(directory, filename, content):
    """原子创建文档；失败时按身份复核回滚。

    调用方必须已持有 ``_MUTATION_LOCK``，并已完成同名判定与内容去重。
    """
    target = Path(directory) / filename
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)

    try:
        fd = os.open(target, flags)
    except FileExistsError as exc:
        raise AlreadyExists("同名文件已存在，未覆盖", filename=filename) from exc
    except OSError as exc:
        raise WriteFailed(
            "文件创建失败", filename=filename, reason=type(exc).__name__
        ) from exc

    st0 = None
    try:
        st0 = os.fstat(fd)
        _write_all(fd, content)
    except BaseException as exc:
        try:
            os.close(fd)
        except OSError:
            pass
        if _rollback_created(target, st0):
            logger.error(
                "上传写入失败且回滚未完成，可能留下残留文件：%s（%s: %s）",
                filename,
                type(exc).__name__,
                exc,
            )
            raise CreateRollbackFailed(
                "写入失败且回滚未完成，可能留下残留文件（内容完整性未经验证）",
                filename=filename,
                residual=True,
                reason=type(exc).__name__,
            ) from exc
        raise WriteFailed(
            "文件写入失败，已回滚", filename=filename, reason=type(exc).__name__
        ) from exc
    else:
        os.close(fd)

    return {
        "filename": filename,
        "bytes": len(content),
        "document_id": compute_document_id(filename),
    }


def _ensure_directory_ready(directory):
    """import 入口的目录准备：缺失则创建（保留既有行为），不可信则拒绝。"""
    state = inspect_knowledge_directory(directory)

    if state == "missing":
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise KnowledgeDirInvalid(
                "知识库目录无法创建", path=str(directory), reason=type(exc).__name__
            ) from exc
        state = inspect_knowledge_directory(directory)

    if state == "untrusted":
        raise KnowledgeDirUntrusted(
            "知识库目录是重解析点（junction/符号链接），拒绝写入",
            path=str(directory),
        )
    if state in ("not_directory", "inaccessible"):
        raise KnowledgeDirInvalid(
            "知识库路径不可用（不是目录或无法访问）",
            path=str(directory),
        )


def import_document(directory, filename, content):
    """导入一个文档：目录准备 → 同名判定 → 内容去重 → 原子创建。"""
    _ensure_directory_ready(directory)

    with _MUTATION_LOCK:
        target = Path(directory) / filename
        # lexists 而非 exists：悬空符号链接也会被视为「已存在」，
        # 避免沿链接写到知识库之外。
        if os.path.lexists(target):
            raise AlreadyExists("同名文件已存在，未覆盖", filename=filename)

        duplicate = find_duplicate_content(directory, content, exclude_filename=filename)
        if duplicate is not None:
            raise DuplicateContent(
                "内容相同的文件已存在，未覆盖",
                filename=filename,
                existing_filename=duplicate,
            )

        return create_document(directory, filename, content)


# ============================================================
# 删除
# ============================================================


def _assert_containment(directory, target):
    try:
        root = Path(directory).resolve()
        parent = target.resolve().parent
    except OSError as exc:
        raise UnexpectedPathState(
            "无法解析目标路径", filename=target.name
        ) from exc
    if parent != root:
        raise UnexpectedPathState("目标不在知识库目录顶层", filename=target.name)


def delete_document(directory, document_id):
    """按 document_id 删除一个文档。

    执行顺序（每个错误码都在对应路径上真实可达）：

    1. id 语法（不触盘）→ 400；
    2. 目录检查：缺失 → 404；重解析点 → 503；非目录/不可访问 → 500；
    3. 枚举建表：白名单后缀条目入表（含重解析点，仅用于识别）；
    4. 查表未命中 → 404；
    5. 重解析点 → 403（不删链接、不删目标）；非普通文件 → 403；
    6. lstat 复核类型 → 归属断言 → 身份二次比对 → 409；
    7. unlink：占用 → 409，其它 → 500。

    删除目标名**只**来自第 3 步的枚举结果，id 不参与路径拼接。
    """
    if not isinstance(document_id, str) or not _DOCUMENT_ID_RE.match(document_id):
        raise InvalidDocumentId(
            "document_id 必须是 32 位小写十六进制字符串", document_id=document_id
        )

    state = inspect_knowledge_directory(directory)
    if state == "missing":
        raise DocumentNotFound("文档不存在", document_id=document_id)
    if state == "untrusted":
        raise KnowledgeDirUntrusted(
            "知识库目录是重解析点（junction/符号链接），拒绝删除",
            path=str(directory),
        )
    if state in ("not_directory", "inaccessible"):
        raise KnowledgeDirInvalid(
            "知识库路径不可用（不是目录或无法访问）", path=str(directory)
        )

    index = _scan_delete_index(directory)
    hit = index.get(document_id)
    if hit is None:
        raise DocumentNotFound("文档不存在", document_id=document_id)

    filename, kind = hit
    if kind == "reparse":
        raise ReparsePointRejected(
            "目标是重解析点（符号链接/junction），拒绝删除", filename=filename
        )
    if kind != "regular":
        raise NotRegularFile("目标不是普通文件", filename=filename)

    target = Path(directory) / filename

    try:
        st0 = os.lstat(target)
    except OSError as exc:
        raise ConcurrentModification(
            "目标在枚举后已消失或被替换", filename=filename
        ) from exc
    if _is_reparse_point(st0):
        raise ReparsePointRejected(
            "目标是重解析点（符号链接/junction），拒绝删除", filename=filename
        )
    if not stat.S_ISREG(st0.st_mode):
        raise NotRegularFile("目标不是普通文件", filename=filename)

    _assert_containment(directory, target)

    try:
        st1 = os.lstat(target)
    except OSError as exc:
        raise ConcurrentModification(
            "目标在检查与删除之间被替换", filename=filename
        ) from exc
    if (st1.st_dev, st1.st_ino) != (st0.st_dev, st0.st_ino):
        raise ConcurrentModification(
            "目标在检查与删除之间被替换", filename=filename
        )

    freed_bytes = st0.st_size
    try:
        os.unlink(target)
    except PermissionError as exc:
        raise FileBusy("目标文件被占用，无法删除", filename=filename) from exc
    except OSError as exc:
        raise DeleteFailed(
            "删除失败", filename=filename, reason=type(exc).__name__
        ) from exc

    return {
        "document_id": document_id,
        "filename": filename,
        "freed_bytes": freed_bytes,
    }
