import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager

from image_client import DeepSeekVisionClient, VisionClient
from image_validator import MAX_IMAGE_SIZE_BYTES
from image_understanding import understand_image
from functools import lru_cache

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi import Depends, Request
from dotenv import load_dotenv

from agent import Agent
from memory import (
    ConversationAccessDenied,
    LEGACY_CONVERSATION_ID,
    Memory,
)
from pathlib import Path

from business.api import router as business_router
from business.db import init_database

import doc_formats
import knowledge_store

from auth import security as auth_security
from auth.deps import (
    SESSION_COOKIE,
    get_current_user,
    get_optional_user,
    get_page_user,
    get_page_user_strict,
    get_service,
    require_admin,
    require_password_changed,
    require_same_origin,
)
from auth.service import (
    ACTOR_TYPE_HTTP_ADMIN,
    AccountDisabled,
    AuthService,
    InvalidCredentials,
    RateLimited,
    UserNotFound,
    UsernameTaken,
)


logger = logging.getLogger(__name__)

# 必须在读取 PAI_ENV 之前加载 .env：
# PRODUCTION 判断依赖 os.getenv("PAI_ENV")，若 load_dotenv() 晚于该判断，
# 写在 .env 里的 PAI_ENV=production 不会生效，/docs 等路由仍会暴露。
load_dotenv()


def require_business_db(request: Request) -> None:
    """业务数据库不可用时，阻止六个业务接口执行。"""
    if not getattr(request.app.state, "business_db_available", False):
        raise HTTPException(
            status_code=503,
            detail={
                "type": "business_db_unavailable",
                "message": "业务数据库不可用，请检查启动日志",
            },
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化业务数据库与认证表结构；失败不阻止应用启动（D-01）。"""
    try:
        init_database()
        app.state.business_db_available = True
        logger.info("业务数据库初始化成功")
    except Exception:
        app.state.business_db_available = False
        logger.exception("业务数据库初始化失败，业务接口将返回 503")

    try:
        AuthService().ensure_schema()
        app.state.auth_schema_ready = True
        logger.info("认证表结构就绪")
    except Exception:
        app.state.auth_schema_ready = False
        logger.exception("认证表结构初始化失败，登录不可用")

    yield


#: 生产环境关闭交互式文档：PAI_ENV=production 时这三个路径一律 404。
PRODUCTION = os.getenv("PAI_ENV", "development").strip().lower() == "production"

app = FastAPI(
    lifespan=lifespan,
    docs_url=None if PRODUCTION else "/docs",
    redoc_url=None if PRODUCTION else "/redoc",
    openapi_url=None if PRODUCTION else "/openapi.json",
)

app.state.business_db_available = False
app.state.auth_schema_ready = False


def business_access_guard(
    request: Request,
    user: dict = Depends(get_current_user),
) -> dict:
    """业务接口鉴权：先登录；强制改密一律 403；GET 允许已登录用户，其余方法（6 个 POST）仅管理员。

    强制改密检查在角色检查之前：管理员被重置密码后（must_change_password=1）
    同样不能读写业务数据，必须先完成改密。检查发生在 router 依赖解析阶段，
    早于任何业务数据库访问。
    """
    if user.get("must_change_password"):
        raise HTTPException(
            status_code=403,
            detail={
                "type": "password_change_required",
                "message": "首次登录必须先修改密码，修改后才能继续使用",
            },
            headers={"Cache-Control": "no-store"},
        )
    if request.method.upper() != "GET" and user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail={"type": "forbidden", "message": "需要管理员权限"},
        )
    return user


app.include_router(
    business_router,
    dependencies=[Depends(require_business_db), Depends(business_access_guard)],
)

# load_dotenv() 已上移到模块顶部（须早于 PAI_ENV 判断），此处不再重复调用。

# 知识库目录与导入约束（供 /knowledge/import、/knowledge/search 使用）。
KNOWLEDGE_DIR = "rag_knowledge"
# 格式清单的唯一来源是 doc_formats.SUPPORTED_SUFFIXES（.md/.txt/.pdf/.docx 四种）：
# 新接口 /knowledge/documents、存储层与索引层都引用它，任何位置都不得再重复
# 维护一份格式清单。
# 保留原变量名，兼容既有调用与测试。
ALLOWED_KNOWLEDGE_SUFFIXES = doc_formats.SUPPORTED_SUFFIXES
MAX_KNOWLEDGE_FILE_BYTES = 10 * 1024 * 1024  # 单文件导入上限 10 MiB

# 旧接口 /knowledge/import 的**专用**后缀白名单：只服务 UTF-8 文本格式，
# 是上面统一清单的明确例外。它必须是 ALLOWED_KNOWLEDGE_SUFFIXES 的真子集
# （由测试锁定），不得在此扩充二进制格式——.pdf/.docx 统一走
# POST /knowledge/documents。
LEGACY_IMPORT_SUFFIXES = (".md", ".txt")


@lru_cache(maxsize=1)
def get_knowledge_base():
    """创建（惰性）项目知识库；目录缺失或不可用时返回 None。

    与 get_agent 分离：/knowledge/search 无需 LLM Key 即可检索；
    检索/问答共用同一个缓存的 KnowledgeBase 实例。

    目录信任检查用 lstat（不跟随重解析点）：叶子目录是 junction/符号链接时
    一律返回 None，使 /knowledge/search 沿用既有 503 契约、/chat 走既有降级
    分支。**不抛异常**——抛出会被 /chat 的 except 捕获成 502，改变对话契约。
    """
    from rag.knowledge_base import KnowledgeBase

    state = knowledge_store.inspect_knowledge_directory(KNOWLEDGE_DIR)

    if state != "ok":
        logger.warning(
            "知识目录不可用：state=%s path=%s", state, KNOWLEDGE_DIR
        )
        return None

    return KnowledgeBase(Path(KNOWLEDGE_DIR))


def _invalidate_knowledge_caches():
    """文件系统变更成功后清理两级缓存，顺序固定为 KnowledgeBase → Agent。

    测试可能用 lambda 或自定义替身替换这两个名字，因此按属性探测：
    没有 cache_clear 时跳过而不是抛 AttributeError（不改变既有测试契约）。
    """
    for factory in (get_knowledge_base, get_agent):
        clear = getattr(factory, "cache_clear", None)
        if callable(clear):
            clear()


_WITH_INDEX_VALUES = {"true": True, "1": True, "false": False, "0": False}


def _parse_with_index(values):
    """解析 with_index 查询参数。

    - 缺省：False；
    - 恰好一个且大小写不敏感地匹配 true/1/false/0（不做 strip，空串与带空白均拒绝）；
    - 重复出现或其它取值：400 invalid_parameter。
    """
    if not values:
        return False

    if len(values) > 1:
        raise HTTPException(
            status_code=400,
            detail={
                "type": "invalid_parameter",
                "message": "with_index 只能出现一次",
            },
        )

    raw = values[0]
    key = raw.lower() if isinstance(raw, str) else None
    if key not in _WITH_INDEX_VALUES:
        raise HTTPException(
            status_code=400,
            detail={
                "type": "invalid_parameter",
                "message": "with_index 只接受 true/false/1/0",
            },
        )

    return _WITH_INDEX_VALUES[key]


def _knowledge_http_error(exc):
    """把 knowledge_store 的业务异常映射为 HTTPException。"""
    detail = {"type": exc.error_type, "message": exc.message}
    detail.update(exc.extra)
    return HTTPException(status_code=exc.status_code, detail=detail)


@lru_cache(maxsize=1)
def get_memory() -> Memory:
    """创建（惰性）会话与消息存储。

    与 Agent 内部使用的 Memory 指向同一个 memory.db（相对路径），
    但独立暴露：缺少 DEEPSEEK_API_KEY 时 /chat 仍能完成会话记账，
    /conversations 也不依赖 Agent 构造。
    """
    return Memory()


CONVERSATION_ID_MAX_LENGTH = 64

MESSAGES_DEFAULT_LIMIT = 200
MESSAGES_MAX_LIMIT = 500


def normalize_conversation_id(raw: str | None) -> str:
    """归一化请求中的 conversation_id；缺失时生成新 UUID。

    只接受合法 UUID 或系统保留的 legacy 会话：客户端传来的值会先
    归一化为标准形式再进入数据库查询，任意用户字符串不会直接作为
    conversation_id 使用。
    """
    if raw is None:
        return str(uuid.uuid4())

    if not isinstance(raw, str):
        raise HTTPException(
            status_code=400,
            detail={
                "type": "invalid_conversation_id",
                "message": "conversation_id 必须是字符串",
            },
        )

    value = raw.strip()

    if not value:
        raise HTTPException(
            status_code=400,
            detail={
                "type": "invalid_conversation_id",
                "message": "conversation_id 不能为空",
            },
        )

    if len(value) > CONVERSATION_ID_MAX_LENGTH:
        raise HTTPException(
            status_code=400,
            detail={
                "type": "invalid_conversation_id",
                "message": "conversation_id 长度超过限制",
            },
        )

    if value == LEGACY_CONVERSATION_ID:
        return LEGACY_CONVERSATION_ID

    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(
            status_code=400,
            detail={
                "type": "invalid_conversation_id",
                "message": "conversation_id 必须是合法 UUID",
            },
        )


@lru_cache(maxsize=1)
def get_agent() -> Agent:
    """创建 DeepSeek Agent。"""

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY 环境变量")

    # 接入项目知识库（rag_knowledge 目录）。
    # KnowledgeBase 为惰性加载：此处不加载 Embedding 模型、不读文件。
    # "目录已接入"不等于"索引已就绪"：索引在首次检索时构建；
    # 目录为空或文件不可读会在检索时以明确诊断失败。
    knowledge_base = get_knowledge_base()

    if knowledge_base is None:
        # 目录缺失/不可用时不再退化为「不注册工具」：仍注入一个惰性
        # KnowledgeBase，让真正执行检索的请求在初始化阶段拿到明确诊断
        # （init_error_kind=dir_missing → degraded），而不是静默失去
        # 知识检索能力、把「没查」混同于「没查到」。
        # 该实例只在目录不可用时创建，且初始化即失败，不会加载 Embedding；
        # /knowledge/search 仍沿用 get_knowledge_base() 的 None → 503 分支，
        # 其响应结构不受影响。
        from rag.knowledge_base import KnowledgeBase

        knowledge_base = KnowledgeBase(Path(KNOWLEDGE_DIR))

        print(
            "[知识库] 警告：知识目录不存在或不可用"
            "（search_knowledge 仍注册，检索时返回 degraded 诊断）。"
        )
    else:
        print(
            "[知识库] 目录已接入"
            "（惰性加载：索引将在首次检索时构建）。"
        )

    return Agent(
        api_key,
        "https://api.deepseek.com",
        "deepseek-flash",
        knowledge_base=knowledge_base
    )


@lru_cache(maxsize=1)
def get_vision_client() -> VisionClient | None:
    """创建 DeepSeek 视觉客户端；缺少配置时返回 None。"""

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        # 与 get_agent() 的差异：缺配置不抛异常，
        # 交由 understand_image 的 model_not_configured 分支降级处理。
        # 注意：lru_cache 会缓存 None，运行期补配后需重启进程或 cache_clear()。
        return None

    # base_url / model / timeout 沿用 DeepSeekVisionClient 的默认值。
    # 此处只构造客户端，不发送请求；真实请求由 analyze() 触发。
    return DeepSeekVisionClient(api_key=api_key)


class ChatRequest(BaseModel):
    message: str
    # 可选：缺省时由服务端生成新 UUID，旧调用方因此保持兼容。
    conversation_id: str | None = None


@app.post("/chat")
def chat(request: ChatRequest, user: dict = Depends(require_password_changed)):
    """私人对话入口：必须登录，全部读写按当前用户归属隔离。"""

    conversation_id = None

    try:

        if not request.message.strip():
            raise HTTPException(
                status_code=422,
                detail={
                    "type": "invalid_request",
                    "message": "message 不能为空",
                },
            )

        # 1) 取得或生成会话标识（非法输入在此直接 400）
        conversation_id = normalize_conversation_id(
            request.conversation_id
        )

        # 2) 创建或确认会话记录，保证后续消息写入有一致的归属
        #    conversation_id 若已属于其他用户，ensure_conversation 抛
        #    ConversationAccessDenied（下面统一映射为 404，不泄露存在性）。
        get_memory().ensure_conversation(conversation_id, user_id=user["id"])

        # 3) 调用 Agent：本轮全部读写都绑定同一个 conversation_id 与 user_id
        result = get_agent().run(
            request.message, conversation_id, user_id=user["id"]
        )

        # ============================================================
        # 调试信息
        # ============================================================

        # 调试信息：不打印完整 result，也不打印 answer 原文。
        # 订单回答可能包含订单编号、客户编号、金额、付款时间等
        # 敏感数据，因此这里只输出安全元数据。
        print()
        print("========== Agent 返回结果 ==========")
        print("结果类型：", type(result))

        if isinstance(result, dict):

            answer = result.get("answer")
            answer_length = len(answer) if isinstance(answer, str) else 0

            print("success：", result.get("success"))
            print("answer 长度：", answer_length)

            if result.get("success") is False:
                print(
                    "错误类型：",
                    type(result.get("error")).__name__,
                )

        else:

            print("结果结构：非字典")

        print("====================================")
        print()

        # ============================================================
        # 统一 Agent 返回格式
        #
        # Agent 正常情况下：
        #
        # {
        #     "success": True,
        #     "answer": "100"
        # }
        #
        # ============================================================

        if isinstance(result, dict):

            success = result.get("success", False)
            sources = result.get("sources", [])
            if not isinstance(sources, list):
                sources = []

            # 检索状态（N1）：仅当 Agent 给出字典时才透传。
            # Agent 未执行检索时该字段整个省略，这里也不补默认值——
            # 否则会把「没有查」误报成「查了但没查到」。
            retrieval = result.get("retrieval")

            # Agent 成功
            if success:

                answer = result.get("answer")

                # 防止 answer 不存在
                if answer is None:

                    answer = "Agent 执行成功，但没有生成最终答案。"

                response = {
                    "success": True,
                    "answer": str(answer),
                    "sources": sources,
                    "conversation_id": conversation_id,
                }

                if isinstance(retrieval, dict):
                    response["retrieval"] = retrieval

                return response

            # Agent 失败
            else:

                error = result.get(
                    "error",
                    "Agent 执行失败。"
                )

                response = {
                    "success": False,
                    "answer": f"执行失败：{error}",
                    "error": str(error),
                    "sources": sources,
                    "conversation_id": conversation_id,
                }

                if isinstance(retrieval, dict):
                    response["retrieval"] = retrieval

                return response

        # ============================================================
        # 如果 Agent 直接返回字符串
        # ============================================================

        elif isinstance(result, str):

            return {
                "success": True,
                "answer": result,
                "sources": [],
                "conversation_id": conversation_id,
            }

        # ============================================================
        # 如果 Agent 返回其他类型
        # ============================================================

        else:

            return {
                "success": True,
                "answer": str(result),
                "sources": [],
                "conversation_id": conversation_id,
            }

    except HTTPException:
        raise

    except ConversationAccessDenied:
        # 会话属于其他用户：与「不存在」返回完全一致，避免探测他人会话 ID。
        raise HTTPException(
            status_code=404,
            detail={
                "type": "conversation_not_found",
                "message": "会话不存在",
            },
        )

    except Exception:
        # 不向前端泄露内部异常详情（可能含密钥、路径等敏感信息）；
        # 完整堆栈仅记录在服务端日志，响应体统一为脱敏信息。
        detail = {
            "type": "internal_error",
            "message": "服务内部错误，请稍后重试",
        }

        # 会话标识已确定时一并回传：请求失败后前端仍能留在原会话重试。
        if conversation_id is not None:
            detail["conversation_id"] = conversation_id

        logger.exception("/chat 处理失败")
        raise HTTPException(status_code=502, detail=detail)


@app.get("/conversations")
def list_conversations(
    limit: int = 50,
    user: dict = Depends(require_password_changed),
):
    """列出**当前用户**的会话元数据（不含消息正文），供前端切换历史对话。"""

    if isinstance(limit, bool) or not 1 <= limit <= 200:
        raise HTTPException(
            status_code=400,
            detail={
                "type": "invalid_limit",
                "message": "limit 必须在 1 到 200 之间",
            },
        )

    conversations = get_memory().list_conversations(
        limit=limit, user_id=user["id"]
    )

    return {
        "success": True,
        "conversations": conversations,
        "total": len(conversations),
    }


@app.get("/conversations/{conversation_id}/messages")
def conversation_messages(
    conversation_id: str,
    limit: int = MESSAGES_DEFAULT_LIMIT,
    user: dict = Depends(require_password_changed),
):
    """读取指定会话的消息，按时间正序返回。

    只返回该 conversation_id 下的消息；空会话返回空数组而不是 404，
    便于前端在本地已有 ID 但服务端尚无记录时正常渲染空会话。
    他人的会话同样返回空数组——两种情形必须不可区分，否则可以用
    状态码差异探测会话 ID 是否存在。
    """
    cid = normalize_conversation_id(conversation_id)

    if isinstance(limit, bool) or not 1 <= limit <= MESSAGES_MAX_LIMIT:
        raise HTTPException(
            status_code=400,
            detail={
                "type": "invalid_limit",
                "message": (
                    f"limit 必须在 1 到 {MESSAGES_MAX_LIMIT} 之间"
                ),
            },
        )

    try:
        messages, total = get_memory().get_messages(
            cid, limit=limit, user_id=user["id"]
        )
    except ConversationAccessDenied:
        # 他人的会话按「空会话」处理，与「会话不存在」返回完全一致。
        # 若这里返回 404，攻击者就能用 404/200 的差异探测某个会话 ID
        # 是否存在；读取本身不泄露内容，因此统一降级为空结果。
        messages, total = [], 0

    return {
        "success": True,
        "conversation_id": cid,
        "messages": messages,
        "total": total,
        # 明确告知是否还有更早的消息被 limit 截断，不静默丢弃历史。
        "has_more": total > len(messages),
    }


@app.get("/knowledge/search")
def knowledge_search(
    query: str,
    top_k: int = 3,
    threshold: float | None = None,
    user: dict = Depends(require_password_changed),
):
    """检索项目知识库（不经过 LLM），返回来源片段、分数与上下文。

    检索对**已登录用户**开放（HR 等普通账号可查），匿名一律 401。
    上传与删除是管理员专属，见下方 require_admin 依赖。

    业务失败（空白查询、空库、初始化失败等）沿用 KnowledgeBase 的
    {"ok": False, "stage", "error"} 结构，HTTP 200 返回，便于调用方区分。
    """
    kb = get_knowledge_base()

    if kb is None:
        raise HTTPException(
            status_code=503,
            detail={
                "type": "knowledge_unavailable",
                "message": "知识库目录不存在或不可用",
            },
        )

    return kb.search(query, top_k=top_k, threshold=threshold)


@app.post("/knowledge/import")
async def knowledge_import(
    file: UploadFile = File(...),
    _admin: dict = Depends(require_admin),
    _ready: dict = Depends(require_password_changed),
):
    """导入单个 .md / .txt 文本文件到知识库目录，并使索引缓存失效。

    与 ``POST /knowledge/documents`` 的分工：本接口只服务 **UTF-8 文本**，
    落盘前对内容执行 ``content.decode("utf-8")`` 校验；PDF / DOCX 请改用
    ``POST /knowledge/documents``（四格式统一入口，按原始字节落盘）。

    安全约束：仅接受 .md/.txt；限制大小 10 MiB；拒绝路径穿越、非法文件名、
    同名覆盖与内容重复；文件内容必须是有效 UTF-8 文本。
    落盘使用 O_CREAT|O_EXCL 原子创建，写入失败按身份复核回滚。
    """
    try:
        original = file.filename or ""
        safe_name = Path(original).name
        suffix = Path(safe_name).suffix.lower()

        # 文件名合法性（Windows 保留名 / 非法字符 / 结尾空格或点）：
        # 这类名字会让落盘直接失败并变成 500，故提前拒绝。
        filename_error = knowledge_store.validate_knowledge_filename(safe_name)
        if filename_error is not None:
            raise HTTPException(
                status_code=400,
                detail={
                    "type": filename_error,
                    "message": "文件名不合法",
                },
            )

        if suffix not in LEGACY_IMPORT_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail={
                    "type": "unsupported_format",
                    "message": (
                        "仅支持 "
                        + " / ".join(LEGACY_IMPORT_SUFFIXES)
                        + " 文件（UTF-8 文本）；PDF/DOCX 请使用 "
                        "POST /knowledge/documents"
                    ),
                },
            )

        content = await file.read(MAX_KNOWLEDGE_FILE_BYTES + 1)

        if len(content) > MAX_KNOWLEDGE_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail={
                    "type": "file_too_large",
                    "message": "文件超过大小限制",
                },
            )

        if not content.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "type": "empty_file",
                    "message": "文件内容为空",
                },
            )

        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail={
                    "type": "invalid_encoding",
                    "message": "文件不是有效的 UTF-8 文本",
                },
            )

        try:
            created = knowledge_store.import_document(
                KNOWLEDGE_DIR, safe_name, content
            )
        except knowledge_store.KnowledgeStoreError as exc:
            raise _knowledge_http_error(exc) from exc

        # 文件系统变更已完成，此时才清理缓存；顺序固定为知识库 → Agent。
        # 反序（先清后写）会让「清缓存 → 索引重建 → 写入完成」窗口内的
        # 新文件永久漏索引，直到下一次变更或重启。
        _invalidate_knowledge_caches()

        return {
            "ok": True,
            "filename": created["filename"],
            "bytes": created["bytes"],
            "document_id": created["document_id"],
            "storage_type": "persistent",
        }

    finally:
        # 与 /analyze-image 相同的关闭策略：有原始异常时只记录、不覆盖。
        has_active_exception = sys.exc_info()[0] is not None

        try:
            await file.close()
        except Exception:
            if has_active_exception:
                logger.exception("关闭上传文件失败；保留原始异常并继续传播")
            else:
                raise


@app.post("/knowledge/documents")
async def knowledge_upload_document(
    file: UploadFile = File(...),
    _admin: dict = Depends(require_admin),
    _ready: dict = Depends(require_password_changed),
):
    """上传一个知识库文档：TXT / MD / PDF / DOCX 的统一入口。

    与 ``/knowledge/import`` 的分工：本接口对四种格式**一视同仁**，只做
    「文件名 → 后缀 → 大小 → 非空」四道 HTTP 层闸门，随后把**原始 bytes**
    交给 ``knowledge_store.import_document``。任何格式的内容结构是否有效
    （PDF 能否解析、DOCX 是否损坏、TXT 是否合法 UTF-8）一律由索引期的
    Loader 负责判定，上传层不重复实现这套判定。

    为什么不做 ``content.decode("utf-8")``：PDF / DOCX 是二进制容器，强制
    解码必然抛 UnicodeDecodeError，会把完全合法的文件误判成「不是有效的
    UTF-8 文本」。同理也不做文件头嗅探——那属于 Loader 的职责，在此处再写
    一遍会形成第二套格式判定，二者迟早不一致。

    响应中的 ``ok=True`` **仅表示文件已落盘**：文件尚未被解析、未进入向量
    索引、也不保证可被检索。索引状态请用
    ``GET /knowledge/documents?with_index=true`` 查询，本接口不新增索引
    状态字段。
    """
    try:
        original = file.filename or ""
        # 只取 basename：目录部分（含 ../）在进入任何路径拼接之前就被丢弃。
        safe_name = Path(original).name

        filename_error = knowledge_store.validate_knowledge_filename(safe_name)
        if filename_error is not None:
            raise HTTPException(
                status_code=400,
                detail={
                    "type": filename_error,
                    "message": "文件名不合法",
                },
            )

        suffix = Path(safe_name).suffix.lower()
        if suffix not in ALLOWED_KNOWLEDGE_SUFFIXES:
            # 文案由唯一白名单派生，避免在此处再写一份格式字面量。
            raise HTTPException(
                status_code=400,
                detail={
                    "type": "unsupported_format",
                    "message": (
                        "仅支持 "
                        + " / ".join(ALLOWED_KNOWLEDGE_SUFFIXES)
                        + " 文件"
                    ),
                },
            )

        # 有界读取：最多拿到上限 + 1 字节即可判定超限，不会把超大文件整体读入。
        content = await file.read(MAX_KNOWLEDGE_FILE_BYTES + 1)

        if len(content) > MAX_KNOWLEDGE_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail={
                    "type": "file_too_large",
                    "message": "文件超过大小限制",
                },
            )

        # 内容校验只区分「空 / 非空」：非空文件一律按原始字节交给存储层，
        # 不做解码，也不做 PDF/DOCX 的文件头校验（见函数说明）。
        if not content.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "type": "empty_file",
                    "message": "文件内容为空",
                },
            )

        try:
            created = knowledge_store.import_document(
                KNOWLEDGE_DIR, safe_name, content
            )
        except knowledge_store.KnowledgeStoreError as exc:
            raise _knowledge_http_error(exc) from exc

        # 与 /knowledge/import 一致：文件系统变更成功后**才**清缓存，顺序固定为
        # 知识库 → Agent；任何失败路径都不清缓存。此处只调用 cache_clear()，
        # 不会初始化 KnowledgeBase / Embedding / Agent。
        _invalidate_knowledge_caches()

        return {
            "ok": True,
            "filename": created["filename"],
            "bytes": created["bytes"],
            "document_id": created["document_id"],
            "storage_type": "persistent",
        }

    finally:
        # 与 /knowledge/import、/analyze-image 相同的关闭策略：
        # 有原始异常时只记录、不覆盖。
        has_active_exception = sys.exc_info()[0] is not None

        try:
            await file.close()
        except Exception:
            if has_active_exception:
                logger.exception("关闭上传文件失败；保留原始异常并继续传播")
            else:
                raise


@app.get("/knowledge/documents")
def knowledge_documents(
    request: Request,
    user: dict = Depends(require_password_changed),
):
    """列出知识库文档（不构建索引、不加载 Embedding）。

    文档清单属于内部资产，仅对已登录用户开放；变更类操作见管理员依赖。

    ``with_index=true`` 时**只读取已有索引状态**：索引尚未构建时一律报
    ``unknown``，绝不为了统计 ``chunk_count`` 而触发构建或加载模型。
    """
    with_index = _parse_with_index(request.query_params.getlist("with_index"))

    try:
        listing = knowledge_store.list_documents(KNOWLEDGE_DIR)
    except knowledge_store.KnowledgeStoreError as exc:
        raise _knowledge_http_error(exc) from exc

    documents = listing["documents"]
    counts = None
    skipped_reparse = None

    if with_index and documents:
        knowledge_base = get_knowledge_base()
        if knowledge_base is not None:
            counts = knowledge_base.indexed_chunk_counts()
            if counts is not None:
                skipped_reparse = getattr(
                    knowledge_base, "skipped_reparse_count", None
                )

    if with_index:
        for document in documents:
            filename = document["filename"]
            if counts is None:
                # 索引未构建：既不假装已索引，也不假装未索引。
                document["parse_status"] = "unknown"
                document["chunk_count"] = None
            elif filename in counts:
                document["parse_status"] = "indexed"
                document["chunk_count"] = int(counts[filename])
            else:
                document["parse_status"] = "not_indexed"
                document["chunk_count"] = None

    return {
        "ok": True,
        "directory": KNOWLEDGE_DIR,
        "directory_exists": listing["directory_exists"],
        "count": len(documents),
        "with_index": with_index,
        "index_skipped_reparse": skipped_reparse,
        "documents": documents,
    }


@app.delete("/knowledge/documents/{document_id}")
def knowledge_delete_document(
    document_id: str,
    _admin: dict = Depends(require_admin),
    _ready: dict = Depends(require_password_changed),
):
    """删除一个知识库文档（只删顶层白名单扩展名的普通文件）。

    ``document_id`` 只用于查表：删除目标名来自受控目录枚举结果，
    用户输入永不参与路径拼接。
    """
    try:
        result = knowledge_store.delete_document(KNOWLEDGE_DIR, document_id)
    except knowledge_store.KnowledgeStoreError as exc:
        raise _knowledge_http_error(exc) from exc

    # 仅删除成功后清理缓存；4xx/409/500 失败路径一律不清。
    _invalidate_knowledge_caches()

    return {
        "ok": True,
        "document_id": result["document_id"],
        "filename": result["filename"],
        "freed_bytes": result["freed_bytes"],
    }


# AI 助手前端只暴露这三个固定文件。
# 路径在导入时一次性确定，任何请求参数都不参与路径拼接。
APP_UI_DIR = Path(__file__).resolve().parent / "app_ui"
APP_UI_INDEX = APP_UI_DIR / "index.html"
APP_UI_SCRIPT = APP_UI_DIR / "static" / "app.js"
APP_UI_STYLE = APP_UI_DIR / "static" / "style.css"


@app.get("/", response_class=HTMLResponse)
def index(_page_user: dict = Depends(get_page_user_strict)):
    """返回 AI 助手页面（未登录 302 到 /login）。"""
    return FileResponse(
        APP_UI_INDEX, media_type="text/html; charset=utf-8"
    )


@app.get("/app-ui/static/style.css")
def app_ui_style():
    """返回 AI 助手页面样式。"""
    return FileResponse(
        APP_UI_STYLE, media_type="text/css; charset=utf-8"
    )


@app.get("/app-ui/static/app.js")
def app_ui_script():
    """返回 AI 助手页面脚本。"""
    return FileResponse(
        APP_UI_SCRIPT, media_type="application/javascript; charset=utf-8"
    )


# 统一工作台外壳：只暴露这三个固定文件。
# 路径在导入时一次性确定，任何请求参数都不参与路径拼接。
WORKBENCH_INDEX = APP_UI_DIR / "workbench.html"
WORKBENCH_CSS = APP_UI_DIR / "static" / "workbench.css"
WORKBENCH_JS = APP_UI_DIR / "static" / "workbench.js"


@app.get("/workbench")
def workbench_index(_page_user: dict = Depends(get_page_user_strict)):
    """返回统一工作台页面（未登录 302 到 /login）。"""
    return FileResponse(
        WORKBENCH_INDEX, media_type="text/html; charset=utf-8"
    )


@app.get("/workbench/static/style.css")
def workbench_style():
    """返回统一工作台样式。"""
    return FileResponse(
        WORKBENCH_CSS, media_type="text/css; charset=utf-8"
    )


@app.get("/workbench/static/app.js")
def workbench_script():
    """返回统一工作台脚本。"""
    return FileResponse(
        WORKBENCH_JS, media_type="application/javascript; charset=utf-8"
    )


# 业务管理前端只暴露这三个固定文件。
# 路径在导入时一次性确定，任何请求参数都不参与路径拼接。
BUSINESS_UI_DIR = Path(__file__).resolve().parent / "business_ui"
BUSINESS_UI_INDEX = BUSINESS_UI_DIR / "index.html"
BUSINESS_UI_SCRIPT = BUSINESS_UI_DIR / "static" / "app.js"
BUSINESS_UI_STYLE = BUSINESS_UI_DIR / "static" / "style.css"


def _business_ui_response(file_path: Path, media_type: str) -> FileResponse:
    """返回固定的业务管理前端文件；文件缺失时返回 404。"""
    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "type": "business_ui_not_found",
                "message": "业务管理前端文件不存在",
            },
        )
    return FileResponse(file_path, media_type=media_type)


@app.get("/business-ui")
def business_ui_index(_page_user: dict = Depends(get_page_user_strict)):
    """返回业务管理前端页面（未登录 302 到 /login）。

    页面本身只对已登录用户可见；页面内的 6 个 POST 另由
    business_access_guard 要求管理员，因此普通用户打开后只能浏览。
    """
    return _business_ui_response(
        BUSINESS_UI_INDEX, "text/html; charset=utf-8"
    )


@app.get("/business-ui/app.js")
def business_ui_script():
    """返回业务管理前端脚本。"""
    return _business_ui_response(
        BUSINESS_UI_SCRIPT, "application/javascript; charset=utf-8"
    )


@app.get("/business-ui/style.css")
def business_ui_style():
    """返回业务管理前端样式。"""
    return _business_ui_response(
        BUSINESS_UI_STYLE, "text/css; charset=utf-8"
    )


# 知识库工作台前端同样只暴露这三个固定文件。
# 路径在导入时一次性确定，任何请求参数都不参与路径拼接。
# R1-2-S2-K1：仅接入「入口 + 访问守卫 + 页面骨架」；
# 页面不调用知识库业务接口，业务功能（列表/搜索/上传/删除）在 K2 接入。
KNOWLEDGE_UI_DIR = Path(__file__).resolve().parent / "knowledge_ui"
KNOWLEDGE_UI_INDEX = KNOWLEDGE_UI_DIR / "index.html"
KNOWLEDGE_UI_SCRIPT = KNOWLEDGE_UI_DIR / "static" / "app.js"
KNOWLEDGE_UI_STYLE = KNOWLEDGE_UI_DIR / "static" / "style.css"


def _knowledge_ui_response(file_path: Path, media_type: str) -> FileResponse:
    """返回固定的知识库前端文件；文件缺失时返回 404。"""
    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "type": "knowledge_ui_not_found",
                "message": "知识库前端文件不存在",
            },
        )
    return FileResponse(file_path, media_type=media_type)


@app.get("/knowledge-ui")
def knowledge_ui_index(_page_user: dict = Depends(get_page_user_strict)):
    """返回知识库前端页面（未登录 302 到 /login）。

    页面由 get_page_user_strict 守卫：未登录 302 /login；
    must_change_password=true 的用户 302 /change-password（K1 沿用
    FIX-01 的强制改密访问控制，不提供绕过入口）。
    所有已登录角色（普通用户/管理员）均可打开本页。
    """
    return _knowledge_ui_response(
        KNOWLEDGE_UI_INDEX, "text/html; charset=utf-8"
    )


@app.get("/knowledge-ui/static/app.js")
def knowledge_ui_script():
    """返回知识库前端脚本（匿名静态资源，固定文件）。"""
    return _knowledge_ui_response(
        KNOWLEDGE_UI_SCRIPT, "application/javascript; charset=utf-8"
    )


@app.get("/knowledge-ui/static/style.css")
def knowledge_ui_style():
    """返回知识库前端样式（匿名静态资源，固定文件）。"""
    return _knowledge_ui_response(
        KNOWLEDGE_UI_STYLE, "text/css; charset=utf-8"
    )


# ---------------------------------------------------------------------------
# 登录页面 + 真实认证后端
#
# 认证前端 + 真实后端（MVP）：
#   1. POST /auth/login 真实校验密码（Argon2id）并签发 HttpOnly 会话 Cookie；
#   2. 公开注册已彻底关闭：不存在任何无需管理员即可建号的 HTTP 路由，
#      登录页也没有注册入口；
#   3. 账号只能由管理员通过 CLI 创建（python -m auth.cli create-user）。
# 页面路由沿用其他前端块的做法：文件路径在导入期一次性确定。
# ---------------------------------------------------------------------------
AUTH_UI_DIR = APP_UI_DIR
AUTH_LOGIN_PAGE = AUTH_UI_DIR / "login.html"
AUTH_CSS = AUTH_UI_DIR / "static" / "auth.css"
AUTH_JS = AUTH_UI_DIR / "static" / "auth.js"

# 改密页面（R1-1-FIX-01）：强制改密用户被业务守卫 302 到这里，
# 普通用户也可自愿打开。样式复用登录页 auth.css，脚本独立成文件。
CHANGE_PASSWORD_PAGE = AUTH_UI_DIR / "change_password.html"
CHANGE_PASSWORD_JS = AUTH_UI_DIR / "static" / "change_password.js"

AUTH_STAGE = "session_cookie"


def _auth_ui_response(file_path: Path, media_type: str) -> FileResponse:
    """返回固定的登录/注册前端文件；文件缺失时返回 404。"""
    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "type": "auth_ui_not_found",
                "message": "登录页面文件不存在",
            },
        )
    return FileResponse(file_path, media_type=media_type)


def _auth_status_payload(user: dict | None) -> dict:
    """认证能力与当前登录态（不包含任何敏感字段）。"""
    return {
        "available": True,
        "stage": AUTH_STAGE,
        "authenticated": user is not None,
        "user": (
            {"username": user["username"], "role": user["role"]}
            if user is not None
            else None
        ),
        "ui": {
            "login_page": "/login",
            "stylesheet": "/auth/static/style.css",
            "script": "/auth/static/app.js",
        },
        "endpoints": {
            "login": {"method": "POST", "path": "/auth/login", "implemented": True},
            "logout": {"method": "POST", "path": "/auth/logout", "implemented": True},
            "session": {"method": "GET", "path": "/auth/me", "implemented": True},
        },
        "capabilities": {
            "password_login": True,
            "public_register": False,
            "invite_code_register": False,
            "session_cookie": True,
            "user_isolation": True,
        },
    }


class LoginRequest(BaseModel):
    """登录请求体：只接受用户名与密码。"""

    username: str
    password: str


class AdminCreateUserRequest(BaseModel):
    """管理员创建用户请求：只接受用户名，角色由服务端固定为 user。"""

    username: str


class AdminPatchUserRequest(BaseModel):
    """管理员启用/禁用用户请求。"""

    is_active: bool


class ChangePasswordRequest(BaseModel):
    """自助改密请求。"""

    current_password: str
    new_password: str


def _cookie_secure() -> bool:
    """生产（HTTPS）返回 True；本地 http 调试由 PAI_COOKIE_SECURE=0 显式关闭。"""
    explicit = os.getenv("PAI_COOKIE_SECURE")
    if explicit is not None:
        return explicit.strip() in ("1", "true", "True", "yes")
    origin = os.getenv("PAI_PUBLIC_ORIGIN", "")
    return origin.startswith("https://")


def _set_session_cookie(response: JSONResponse, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=int(auth_security.ABSOLUTE_TIMEOUT.total_seconds()),
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax",
        path="/",
    )


def _clear_session_cookie(response: JSONResponse) -> None:
    response.delete_cookie(key=SESSION_COOKIE, path="/")


def _auth_error(
    status_code: int,
    code: str,
    message: str,
    headers: dict | None = None,
    **extra,
) -> JSONResponse:
    """认证错误响应。

    同时给出扁平字段（``ok``/``error``/``message``，供登录页直接读取）
    与 ``detail`` 镜像（与全站其它接口的 ``{"detail": {"type", ...}}``
    风格一致），两处表达同一事实，不存在互相矛盾的可能。
    """
    payload = {
        "ok": False,
        "error": code,
        "message": message,
        "detail": {"type": code, "message": message},
    }
    payload.update(extra)
    response_headers = {"Cache-Control": "no-store"}
    if headers:
        response_headers.update(headers)
    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers=response_headers,
    )


@app.get("/login", response_class=HTMLResponse)
def login_page():
    """登录页面（允许匿名访问）。"""
    return _auth_ui_response(AUTH_LOGIN_PAGE, "text/html; charset=utf-8")


@app.get("/auth/static/style.css")
def auth_style():
    """登录页面样式。"""
    return _auth_ui_response(AUTH_CSS, "text/css; charset=utf-8")


@app.get("/auth/static/app.js")
def auth_script():
    """登录页面脚本。"""
    return _auth_ui_response(AUTH_JS, "application/javascript; charset=utf-8")


@app.get("/auth/status")
def auth_status(user: dict | None = Depends(get_optional_user)):
    """认证能力与当前登录态；允许匿名访问。"""
    return JSONResponse(
        content=_auth_status_payload(user),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/auth/me")
def auth_me(user: dict = Depends(get_current_user)):
    """返回当前登录用户（需登录），含前端所需的最小状态。"""
    return JSONResponse(
        content={
            "ok": True,
            "user": {
                "username": user["username"],
                "role": user["role"],
                "must_change_password": user.get("must_change_password", False),
            },
        },
        headers={"Cache-Control": "no-store"},
    )


@app.post("/auth/login")
async def auth_login(
    payload: LoginRequest,
    _origin: None = Depends(require_same_origin),
    service: AuthService = Depends(get_service),
):
    """用户名 + 密码登录；成功签发新会话 Cookie。

    失败语义：401 凭据错误 / 403 账号被禁用 / 429 失败次数过多。
    任何分支都不回显密码，也不区分"用户不存在"与"密码错误"。
    """
    try:
        result = service.login(payload.username, payload.password)
    except RateLimited:
        return _auth_error(
            429,
            "rate_limited",
            "登录失败次数过多，请稍后再试",
            headers={"Retry-After": "900"},
        )
    except AccountDisabled:
        return _auth_error(403, "account_disabled", "账号已被禁用")
    except InvalidCredentials:
        return _auth_error(401, "invalid_credentials", "用户名或密码不正确")

    response = JSONResponse(
        content={
            "ok": True,
            "user": {
                "username": result["user"]["username"],
                "role": result["user"]["role"],
            },
        },
        headers={"Cache-Control": "no-store"},
    )
    _set_session_cookie(response, result["token"])
    return response


@app.post("/auth/logout")
async def auth_logout(
    request: Request,
    _origin: None = Depends(require_same_origin),
    service: AuthService = Depends(get_service),
):
    """登出：撤销服务端会话并清除 Cookie（幂等，不要求登录态）。"""
    service.logout(request.cookies.get(SESSION_COOKIE))
    response = JSONResponse(
        content={"ok": True},
        headers={"Cache-Control": "no-store"},
    )
    _clear_session_cookie(response)
    return response


@app.get("/change-password", response_class=HTMLResponse)
def change_password_page(_page_user: dict = Depends(get_page_user)):
    """改密页面：登录即可访问。

    强制改密用户（must_change_password=1）被业务守卫重定向到本页；
    普通用户也可自愿打开修改密码。注意本页**不使用** get_page_user_strict，
    否则强制改密用户会被重定向到本页自身，形成循环。
    """
    return _auth_ui_response(CHANGE_PASSWORD_PAGE, "text/html; charset=utf-8")


@app.get("/auth/static/change_password.js")
def change_password_script():
    """改密页面脚本（匿名静态资源，与登录页脚本同级）。"""
    return _auth_ui_response(CHANGE_PASSWORD_JS, "application/javascript; charset=utf-8")


# ---------------------------------------------------------------------------
# 管理员账号管理 API（require_admin 前置 + 写操作 require_same_origin）
#
# 注意：require_admin 必须声明在 require_same_origin **之前**，否则匿名/普通
# 用户访问写接口会先命中同源 403，而不是 401/403 的角色语义（影响路由矩阵）。
# ---------------------------------------------------------------------------


def _admin_user_dto(user: dict) -> dict:
    """管理员视角的用户最小 DTO（永不含 password_hash / token / session）。"""
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "is_active": user["is_active"],
        "must_change_password": user["must_change_password"],
        "created_at": user["created_at"],
    }


@app.get("/admin/users")
def admin_list_users(
    admin: dict = Depends(require_admin),
    _ready: dict = Depends(require_password_changed),
    service: AuthService = Depends(get_service),
):
    """列出全部用户（仅管理员）。"""
    return JSONResponse(
        content={
            "ok": True,
            "users": [_admin_user_dto(user) for user in service.list_users()],
        },
        headers={"Cache-Control": "no-store"},
    )


@app.post("/admin/users")
async def admin_create_user(
    payload: AdminCreateUserRequest,
    admin: dict = Depends(require_admin),
    _ready: dict = Depends(require_password_changed),
    _origin: None = Depends(require_same_origin),
    service: AuthService = Depends(get_service),
):
    """管理员创建普通用户：role 固定 user，强制首次改密，临时密码仅回显一次。"""
    temporary_password = auth_security.new_temporary_password()
    try:
        user = service.create_user(
            payload.username,
            temporary_password,
            role="user",
            actor_user_id=admin["id"],
            actor_type=ACTOR_TYPE_HTTP_ADMIN,
            must_change_password=True,
        )
    except UsernameTaken:
        return _auth_error(409, "username_taken", "用户名已存在")
    except ValueError as exc:
        return _auth_error(400, "invalid_username", str(exc))
    return JSONResponse(
        status_code=201,
        content={
            "ok": True,
            "user": {
                "id": user["id"],
                "username": user["username"],
                "role": user["role"],
                "is_active": user["is_active"],
                "must_change_password": user["must_change_password"],
            },
            "temporary_password": temporary_password,
        },
        headers={"Cache-Control": "no-store"},
    )


@app.patch("/admin/users/{username}")
async def admin_patch_user(
    username: str,
    payload: AdminPatchUserRequest,
    admin: dict = Depends(require_admin),
    _ready: dict = Depends(require_password_changed),
    _origin: None = Depends(require_same_origin),
    service: AuthService = Depends(get_service),
):
    """管理员启用/禁用用户；禁用时撤销其全部会话。"""
    try:
        result = service.set_active(
            username,
            payload.is_active,
            actor_user_id=admin["id"],
            actor_type=ACTOR_TYPE_HTTP_ADMIN,
        )
    except UserNotFound:
        return _auth_error(404, "user_not_found", "用户不存在")
    except ValueError as exc:
        return _auth_error(400, "invalid_operation", str(exc))
    return JSONResponse(
        content={
            "ok": True,
            "username": auth_security.normalize_username(username),
            "is_active": result["is_active"],
            "revoked_sessions": result["revoked_sessions"],
        },
        headers={"Cache-Control": "no-store"},
    )


@app.post("/admin/users/{username}/reset-password")
async def admin_reset_password(
    username: str,
    admin: dict = Depends(require_admin),
    _ready: dict = Depends(require_password_changed),
    _origin: None = Depends(require_same_origin),
    service: AuthService = Depends(get_service),
):
    """管理员重置用户密码：强制首次改密 + 撤销全部会话，临时密码仅回显一次。"""
    temporary_password = auth_security.new_temporary_password()
    try:
        result = service.reset_password(
            username,
            temporary_password,
            actor_user_id=admin["id"],
            actor_type=ACTOR_TYPE_HTTP_ADMIN,
        )
    except UserNotFound:
        return _auth_error(404, "user_not_found", "用户不存在")
    except ValueError as exc:
        return _auth_error(400, "invalid_username", str(exc))
    return JSONResponse(
        content={
            "ok": True,
            "username": auth_security.normalize_username(username),
            "temporary_password": temporary_password,
            "revoked_sessions": result["revoked_sessions"],
            "must_change_password": True,
        },
        headers={"Cache-Control": "no-store"},
    )


@app.post("/admin/users/{username}/revoke-sessions")
async def admin_revoke_sessions(
    username: str,
    admin: dict = Depends(require_admin),
    _ready: dict = Depends(require_password_changed),
    _origin: None = Depends(require_same_origin),
    service: AuthService = Depends(get_service),
):
    """管理员撤销目标用户全部会话。"""
    target = service.get_user(username)
    if target is None:
        return _auth_error(404, "user_not_found", "用户不存在")
    revoked = service.revoke_all_sessions(
        target["id"],
        actor_user_id=admin["id"],
        actor_type=ACTOR_TYPE_HTTP_ADMIN,
    )
    return JSONResponse(
        content={
            "ok": True,
            "username": target["username"],
            "revoked_sessions": revoked,
        },
        headers={"Cache-Control": "no-store"},
    )


@app.post("/auth/change-password")
async def auth_change_password(
    request: Request,
    payload: ChangePasswordRequest,
    user: dict = Depends(get_current_user),
    _origin: None = Depends(require_same_origin),
    service: AuthService = Depends(get_service),
):
    """当前用户自助改密：验证旧密码 → 撤销其它会话、保留当前。"""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return _auth_error(401, "unauthenticated", "未登录或登录已失效")
    try:
        result = service.change_password(
            user["id"], payload.current_password, payload.new_password, token
        )
    except InvalidCredentials:
        return _auth_error(401, "invalid_credentials", "当前密码不正确")
    except ValueError as exc:
        return _auth_error(400, "invalid_password", str(exc))
    return JSONResponse(
        content={"ok": True, "must_change_password": result["must_change_password"]},
        headers={"Cache-Control": "no-store"},
    )


DEFAULT_IMAGE_QUESTION = "请描述这张图片的内容"

@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    question: str = Form(""),
    user: dict = Depends(require_password_changed),
):
    """接收图片上传，并限制读取大小（必须登录，避免匿名消耗视觉额度）。"""

    try:
        # 最多读取限制值 + 1 字节，以便识别超限文件。
        image_data = await file.read(MAX_IMAGE_SIZE_BYTES + 1)

        if len(image_data) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail={
                    "type": "image_too_large",
                    "message": "图片超过允许的大小限制",
                },
            )

        try:
            vision_client = get_vision_client()
        except Exception:
            logger.exception("视觉客户端初始化失败")
            raise

        # 问题为空或缺失时，使用默认描述提示词，保证“传图即出结果”。
        if not question.strip():
            question = DEFAULT_IMAGE_QUESTION

        result = understand_image(
            image_data=image_data,
            question=question,
            client=vision_client,
        )

        return {
            "success": result.success,
            "answer": result.answer,
            "error": (
                {
                    "type": result.error.type,
                    "message": result.error.message,
                }
                if result.error is not None
                else None
            ),
        }

    finally:
        # 进入清理前先判断是否已有异常正在传播：
        # - 有原始异常：关闭失败只能记录，不能覆盖原始异常；
        # - 无原始异常：关闭失败必须正常传播，不得吞掉。
        # 关闭操作只尝试一次，不在异常分支中重试。
        has_active_exception = sys.exc_info()[0] is not None

        try:
            await file.close()
        except Exception:
            if has_active_exception:
                logger.exception(
                    "关闭上传文件失败；保留原始异常并继续传播"
                )
            else:
                raise

