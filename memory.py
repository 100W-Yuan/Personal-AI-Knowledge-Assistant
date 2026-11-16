import re
import sqlite3
import uuid
from datetime import datetime
from typing import Optional


# ========================================================
# 会话标识（阶段 1）
#
# conversation_id 是消息的归属键：
# - 升级前的历史消息统一归入 legacy，保证一条都不丢；
# - 新会话一律使用 UUID（前端 crypto.randomUUID 或服务端生成）。
# 长期记忆 memories 刻意不参与会话隔离，仍保持全局共享。
# ========================================================

LEGACY_CONVERSATION_ID = "legacy"
LEGACY_CONVERSATION_TITLE = "历史对话"
DEFAULT_CONVERSATION_TITLE = "新对话"

CONVERSATION_ID_MAX_LENGTH = 64
_CONVERSATION_ID_PATTERN = re.compile(r"^[0-9a-zA-Z_-]{1,64}$")


class ConversationAccessDenied(PermissionError):
    """会话不属于当前用户。

    对外统一表现为 404（不泄露"该会话存在但无权访问"这一信息）；
    在库内部抛出可区分的异常，便于测试断言"确实拦住了"。
    """


class ConversationNotFound(LookupError):
    """会话不存在（或已被删除）。"""


def require_user_id(user_id: str) -> str:
    """校验用户归属参数。

    PAI-AUTH-MVP 起，所有用户私有数据的读写都必须显式带 ``user_id``；
    这里拒绝空值，避免"忘传就变成查全库"。
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id 必须是非空字符串（用户私有数据必须显式归属）")
    return user_id.strip()


def validate_conversation_id(conversation_id: str) -> str:
    """校验并归一化会话标识（只做格式约束，不承担身份认证）。"""

    if not isinstance(conversation_id, str):
        raise ValueError("conversation_id 必须是字符串")

    value = conversation_id.strip()

    if not value:
        raise ValueError("conversation_id 不能为空")

    if len(value) > CONVERSATION_ID_MAX_LENGTH:
        raise ValueError("conversation_id 长度超过限制")

    if not _CONVERSATION_ID_PATTERN.match(value):
        raise ValueError("conversation_id 含非法字符")

    return value


class Memory:
    """负责管理聊天记录和长期记忆。"""

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        # 登记本实例创建过的连接，供 close() 统一释放句柄（Windows 下
        # 未释放的句柄会导致临时数据库文件无法删除，即 WinError 32）。
        self._connections = set()
        self.init_database()

    def get_connection(self):
        connection = sqlite3.connect(self.db_path)
        self._connections.add(connection)
        return connection

    def close(self):
        """显式关闭所有由本实例创建、尚未关闭的 SQLite 连接。

        - 幂等：重复调用不报错；已关闭的连接再次 close() 是安全的。
        - 关闭后 Memory 仍可正常使用：各方法会按需新建连接。
        - 各方法自带的 finally-close 不需要改动，close() 只是兜底。
        """
        while self._connections:
            connection = self._connections.pop()
            try:
                connection.close()
            except sqlite3.Error:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    @staticmethod
    def infer_category(key: str) -> str:
        """根据记忆的 key 自动判断分类。"""
        category_rules = {
            "name": "identity",
            "identity": "identity",
            "major": "profile",
            "age": "profile",
            "location": "profile",
            "preference": "preference",
            "language_preference": "preference",
            "career_goal": "goal",
            "learning_goal": "goal",
            "project": "project",
            "project_name": "project",
            "skill": "skill",
            "skills": "skill",
        }

        return category_rules.get(key, "profile")

    def init_database(self):
        """创建必要的数据表，并兼容缺少新字段的旧数据库。

        迁移全程幂等：
        - 建表用 IF NOT EXISTS，加列前先 PRAGMA 探测，索引用 IF NOT EXISTS；
        - 回填只处理 conversation_id 为空的行；
        - 整个迁移在同一事务内完成，任一步失败整体回滚。
        因此重复执行既不会重复建对象，也不会丢消息。
        """
        connection = self.get_connection()

        try:
            cursor = connection.cursor()

            cursor.execute("BEGIN IMMEDIATE")

            try:
                self._create_tables(cursor)
                self._migrate_memories_columns(cursor)
                self._migrate_messages_conversation(cursor)
                self._migrate_user_columns(cursor)
                self._ensure_legacy_conversation(cursor)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

        finally:
            connection.close()

    @staticmethod
    def _create_tables(cursor):
        """建表：新库直接带 conversation_id，旧库由此保持原样。"""

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                user_id TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL DEFAULT 'legacy',
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'profile',
                reason TEXT NOT NULL
                    DEFAULT '未提供具体保存原因',
                user_id TEXT
            )
        """)

    @staticmethod
    def _migrate_user_columns(cursor):
        """为用户私有数据补 user_id 列（只加列，不改既有数据）。

        PAI-AUTH-MVP：conversations / memories 需要记录归属。
        存量数据保持 user_id = NULL（无归属），由 CLI ``create-admin``
        一次性归属给管理员，避免在无备份的情况下改写历史数据。
        """

        for table in ("conversations", "memories"):
            cursor.execute("PRAGMA table_info(%s)" % table)
            columns = {row[1] for row in cursor.fetchall()}
            if not columns:
                continue
            if "user_id" not in columns:
                cursor.execute("ALTER TABLE %s ADD COLUMN user_id TEXT" % table)

    @staticmethod
    def _migrate_memories_columns(cursor):
        """长期记忆表只补齐历史版本缺失的列，不改结构与内容。"""

        # 检查现有表结构，只添加真正缺失的字段。
        # 不删除旧字段，也不重建现有数据表。
        cursor.execute("PRAGMA table_info(memories)")
        columns = {
            row[1] for row in cursor.fetchall()
        }

        if "category" not in columns:
            cursor.execute("""
                ALTER TABLE memories
                ADD COLUMN category TEXT NOT NULL
                DEFAULT 'profile'
            """)

        if "reason" not in columns:
            cursor.execute("""
                ALTER TABLE memories
                ADD COLUMN reason TEXT NOT NULL
                DEFAULT '未提供具体保存原因'
            """)

    @staticmethod
    def _migrate_messages_conversation(cursor):
        """为 messages 补 conversation_id 列、回填历史行并建查询索引。"""

        cursor.execute("PRAGMA table_info(messages)")
        columns = {
            row[1] for row in cursor.fetchall()
        }

        if "conversation_id" not in columns:
            # ALTER 时 SQLite 会用默认值填充既有行，历史消息因此整体
            # 归入 legacy，既不删除也不改写正文。
            cursor.execute("""
                ALTER TABLE messages
                ADD COLUMN conversation_id TEXT NOT NULL
                DEFAULT 'legacy'
            """)

        # 兜底：任何情况下都不留下无归属消息。
        cursor.execute(
            """
            UPDATE messages
            SET conversation_id = ?
            WHERE conversation_id IS NULL OR conversation_id = ''
            """,
            (LEGACY_CONVERSATION_ID,),
        )

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id_id
            ON messages (conversation_id, id)
        """)

    @staticmethod
    def _ensure_legacy_conversation(cursor):
        """存在历史消息时才登记 legacy 会话，空库不产生占位会话。"""

        cursor.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (LEGACY_CONVERSATION_ID,),
        )

        if cursor.fetchone()[0] <= 0:
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """
            INSERT OR IGNORE INTO conversations (
                conversation_id, title, created_at, updated_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                LEGACY_CONVERSATION_ID,
                LEGACY_CONVERSATION_TITLE,
                now,
                now,
            ),
        )

    # ====================================================
    # 会话管理（阶段 1）
    # ====================================================

    def create_conversation(self, conversation_id=None, title=None, *, user_id: str):
        """创建会话并绑定所有者；已存在时校验归属后原样返回。"""

        owner = require_user_id(user_id)

        if conversation_id is None:
            cid = str(uuid.uuid4())
        else:
            cid = validate_conversation_id(conversation_id)

        normalized_title = (title or "").strip() or DEFAULT_CONVERSATION_TITLE
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        connection = self.get_connection()

        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO conversations (
                    conversation_id, title, created_at, updated_at, user_id
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (cid, normalized_title, now, now, owner),
            )
            connection.commit()

            cursor.execute(
                "SELECT user_id FROM conversations WHERE conversation_id = ?",
                (cid,),
            )
            row = cursor.fetchone()

        finally:
            connection.close()

        # INSERT OR IGNORE 不会覆盖已有行；归属不符说明该 id 已被他人占用。
        if row is not None and row[0] != owner:
            raise ConversationAccessDenied(cid)

        return self.get_conversation(cid, user_id=owner)

    def get_conversation(self, conversation_id, *, user_id: str):
        """读取会话元数据；不存在或不属于当前用户时返回 None（不泄露存在性）。"""

        cid = validate_conversation_id(conversation_id)
        owner = require_user_id(user_id)

        connection = self.get_connection()

        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT conversation_id, title, created_at, updated_at
                FROM conversations
                WHERE conversation_id = ? AND user_id = ?
                """,
                (cid, owner),
            )

            row = cursor.fetchone()

            if row is None:
                return None

            return {
                "conversation_id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
            }

        finally:
            connection.close()

    def ensure_conversation(self, conversation_id, *, user_id: str):
        """明确的 get-or-create 路径：会话不存在时才创建。

        保存消息前必须经过这里，避免出现「消息指向一个不存在的会话」。
        若该 conversation_id 已属于其他用户，抛 ConversationAccessDenied。
        """

        cid = validate_conversation_id(conversation_id)
        owner = require_user_id(user_id)

        existing = self.get_conversation(cid, user_id=owner)

        if existing is not None:
            return existing

        return self.create_conversation(cid, user_id=owner)

    def list_conversations(self, limit: int = 50, *, user_id: str):
        """列出**当前用户**的会话元数据（含消息条数），不含任何消息正文。"""

        owner = require_user_id(user_id)

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit 必须是整数")

        if not 1 <= limit <= 200:
            raise ValueError("limit 必须在 1 到 200 之间")

        connection = self.get_connection()

        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT
                    c.conversation_id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    COUNT(m.id)
                FROM conversations AS c
                LEFT JOIN messages AS m
                    ON m.conversation_id = c.conversation_id
                WHERE c.user_id = ?
                GROUP BY
                    c.conversation_id, c.title, c.created_at, c.updated_at
                ORDER BY c.updated_at DESC, c.conversation_id ASC
                LIMIT ?
                """,
                (owner, limit),
            )

            return [
                {
                    "conversation_id": row[0],
                    "title": row[1],
                    "created_at": row[2],
                    "updated_at": row[3],
                    "message_count": row[4],
                }
                for row in cursor.fetchall()
            ]

        finally:
            connection.close()

    def delete_conversation(self, conversation_id, *, user_id: str):
        """删除**自己的**一个会话及其全部消息；legacy 会话禁止删除。"""

        cid = validate_conversation_id(conversation_id)
        owner = require_user_id(user_id)

        if cid == LEGACY_CONVERSATION_ID:
            raise ValueError("历史会话 legacy 不允许删除")

        connection = self.get_connection()

        try:
            cursor = connection.cursor()

            cursor.execute("BEGIN IMMEDIATE")

            try:
                cursor.execute(
                    "SELECT user_id FROM conversations WHERE conversation_id = ?",
                    (cid,),
                )
                owner_row = cursor.fetchone()
                if owner_row is not None and owner_row[0] != owner:
                    connection.rollback()
                    raise ConversationAccessDenied(cid)

                cursor.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                    (cid,),
                )
                message_count = cursor.fetchone()[0]

                cursor.execute(
                    "DELETE FROM messages WHERE conversation_id = ?",
                    (cid,),
                )
                cursor.execute(
                    "DELETE FROM conversations "
                    "WHERE conversation_id = ? AND user_id = ?",
                    (cid, owner),
                )

                connection.commit()

            except ConversationAccessDenied:
                raise
            except Exception:
                connection.rollback()
                raise

        finally:
            connection.close()

        return {
            "conversation_id": cid,
            "deleted_messages": message_count,
        }

    def get_messages(self, conversation_id, limit: int = 200, *, user_id: str):
        """按会话读取消息，按时间正序返回（含 message_id）。

        返回 (messages, total)：total 为该会话消息总数，
        messages 为本次返回的切片，便于调用方如实报告是否还有更多。

        归属校验：会话不存在 → 返回空；会话属于他人 → ConversationAccessDenied。
        """

        cid = validate_conversation_id(conversation_id)
        owner = require_user_id(user_id)

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit 必须是整数")

        if not 1 <= limit <= 500:
            raise ValueError("limit 必须在 1 到 500 之间")

        connection = self.get_connection()

        try:
            cursor = connection.cursor()

            cursor.execute(
                "SELECT user_id FROM conversations WHERE conversation_id = ?",
                (cid,),
            )
            owner_row = cursor.fetchone()

            if owner_row is None:
                return ([], 0)
            if owner_row[0] is None:
                # 迁移前的历史数据尚未归属：对任何具体用户都不可见，
                # 既不算「不存在」也不算「属于他人」，直接返回空。
                return ([], 0)
            if owner_row[0] != owner:
                raise ConversationAccessDenied(cid)

            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                (cid,),
            )
            total = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT id, role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (cid, limit),
            )

            rows = cursor.fetchall()
            rows.reverse()

            return (
                [
                    {
                        "id": row[0],
                        "role": row[1],
                        "content": row[2],
                        "created_at": row[3],
                    }
                    for row in rows
                ],
                total,
            )

        finally:
            connection.close()

    def save_message(
        self,
        role: str,
        content: str,
        conversation_id: str = LEGACY_CONVERSATION_ID,
        *,
        user_id: str,
    ):
        """保存一条聊天消息，并绑定到指定会话。

        写入前先走 ensure_conversation（明确的 get-or-create），
        保证不会出现「消息指向不存在的会话」；会话属于他人时拒绝写入。
        同时刷新 conversations.updated_at，供会话列表排序。
        """
        cid = validate_conversation_id(conversation_id)
        owner = require_user_id(user_id)

        self.ensure_conversation(cid, user_id=owner)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        connection = self.get_connection()

        try:
            cursor = connection.cursor()

            cursor.execute(
                """
                INSERT INTO messages (
                    conversation_id, role, content, created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    cid,
                    role,
                    content,
                    now,
                ),
            )

            cursor.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE conversation_id = ? AND user_id = ?
                """,
                (now, cid, owner),
            )

            connection.commit()

        finally:
            connection.close()

    def get_recent_messages(
        self,
        limit: int = 10,
        conversation_id: str = LEGACY_CONVERSATION_ID,
        *,
        user_id: str,
    ):
        """获取指定会话最近的聊天消息，并按时间正序返回。

        必须按 conversation_id 过滤：不再读取全局最近消息，
        否则不同对话的历史会互相污染。会话属于他人时抛 ConversationAccessDenied。
        """
        cid = validate_conversation_id(conversation_id)
        owner = require_user_id(user_id)

        connection = self.get_connection()

        try:
            cursor = connection.cursor()

            cursor.execute(
                "SELECT user_id FROM conversations WHERE conversation_id = ?",
                (cid,),
            )
            owner_row = cursor.fetchone()
            if owner_row is None:
                return []
            if owner_row[0] is None:
                # 迁移前的历史数据尚未归属，对任何具体用户都不可见。
                return []
            if owner_row[0] != owner:
                raise ConversationAccessDenied(cid)

            cursor.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (cid, limit),
            )

            rows = cursor.fetchall()
            rows.reverse()

            return [
                {
                    "role": role,
                    "content": content,
                    "created_at": created_at,
                }
                for role, content, created_at in rows
            ]

        finally:
            connection.close()

    def search_memories(
        self,
        query: str,
        limit: int = 5,
        conversation_id: str = LEGACY_CONVERSATION_ID,
        *,
        user_id: str,
    ):
        """在指定会话的聊天历史 messages 中进行字符匹配搜索。

        检索范围严格限定在 conversation_id 内，不再查询全局消息池，
        避免把另一段对话的内容带入本轮上下文；会话属于他人时拒绝。
        """
        cid = validate_conversation_id(conversation_id)
        owner = require_user_id(user_id)

        connection = self.get_connection()

        try:
            cursor = connection.cursor()

            cursor.execute(
                "SELECT user_id FROM conversations WHERE conversation_id = ?",
                (cid,),
            )
            owner_row = cursor.fetchone()
            if owner_row is None:
                return []
            if owner_row[0] is None:
                # 迁移前的历史数据尚未归属，对任何具体用户都不可见。
                return []
            if owner_row[0] != owner:
                raise ConversationAccessDenied(cid)

            cursor.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                """,
                (cid,),
            )

            rows = cursor.fetchall()
            keywords = [
                char for char in query if char.strip()
            ]

            results = []

            for role, content, created_at in rows:
                score = sum(
                    1 for keyword in keywords
                    if keyword in content
                )

                if score > 0:
                    results.append({
                        "role": role,
                        "content": content,
                        "created_at": created_at,
                        "score": score,
                    })

            results.sort(
                key=lambda item: item["score"],
                reverse=True,
            )

            return results[:limit]

        finally:
            connection.close()

    def find_memory(self, key: str, connection=None, *, user_id: str):
        """根据 key 查找**当前用户**最新的一条长期记忆。

        connection：
        - 不传（None）：自行创建连接，用完即关（原有行为）。
        - 传入外部连接：复用它查询，且不关闭（生命周期由调用方管理）。
        """
        owner = require_user_id(user_id)
        owns_connection = connection is None

        if owns_connection:
            connection = self.get_connection()

        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT
                    id, key, value, source, confidence,
                    created_at, updated_at, category, reason
                FROM memories
                WHERE key = ? AND user_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (key, owner),
            )

            row = cursor.fetchone()

            if row is None:
                return None

            return {
                "id": row[0],
                "key": row[1],
                "value": row[2],
                "source": row[3],
                "confidence": row[4],
                "created_at": row[5],
                "updated_at": row[6],
                "category": row[7],
                "reason": row[8],
            }

        finally:
            if owns_connection:
                connection.close()

    # 检索打分前从查询中剔除的高频虚词，
    # 避免单个常见汉字就让无关记忆获得分数。
    QUERY_STOPWORD_CHARS = set(
        "的了是在你他她它们什么吗呢吧啊呀哦嗯"
        "这那个有个人和与也就都很要会对不为去说想"
    )

    @staticmethod
    def _tokenize_query(query: str):
        """把查询切分为打分词元（本轮评分改进的核心）。

        - 中文（非 ASCII）字符：逐字作为词元（原有行为，停用字剔除）。
        - ASCII 连续字母/数字：整段作为【一个】词元——只有整段片段
          真正出现在 key/value 中才计分，避免查询 "major" 时被
          "project"（含 o/r/j 等单字母）碰撞召回。
        - 单个 ASCII 字符（如 "o"）：证据过弱，不予计分。
        - 空白与标点：跳过，不计分。
        """
        keywords = []
        buffer = ""

        def flush():
            nonlocal buffer
            if buffer:
                keywords.append(buffer)
                buffer = ""

        for char in query:
            if not char.strip() or char in Memory.QUERY_STOPWORD_CHARS:
                flush()
                continue

            if char.isascii() and char.isalnum():
                buffer += char
            else:
                flush()
                keywords.append(char)

        flush()

        return [
            keyword for keyword in keywords
            if not (len(keyword) == 1 and keyword.isascii())
        ]

    def search_long_term_memory(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
        min_score: int = 2,
        *,
        user_id: str,
    ):
        """搜索长期记忆，返回匹配分数和新增字段。

        - category：传入时只检索该分类（参数化查询，不拼接 SQL）。
        - min_score：相关性阈值，低于该分数的记忆不返回。
          默认 2：单个汉字在 value 里的偶然命中（1 分）不再返回，
          需要命中 key（2 分）或多个字符才能入选。
        - 打分前会剔除查询中的高频虚词（QUERY_STOPWORD_CHARS），
          避免"的""是"等常见字给无关记忆加分。
        - 打分按词元进行（_tokenize_query）：中文逐字、ASCII 连续
          片段整体匹配，单个 ASCII 字符不计分——查询 "major" 不会
          再因单字母重合召回 "project" 等无关记忆。
        """
        owner = require_user_id(user_id)

        connection = self.get_connection()

        try:
            cursor = connection.cursor()

            if category is None:
                cursor.execute(
                    """
                    SELECT
                        id, key, value, source, confidence,
                        created_at, updated_at, category, reason
                    FROM memories
                    WHERE user_id = ?
                    ORDER BY id DESC
                    """,
                    (owner,),
                )
            else:
                cursor.execute(
                    """
                    SELECT
                        id, key, value, source, confidence,
                        created_at, updated_at, category, reason
                    FROM memories
                    WHERE category = ? AND user_id = ?
                    ORDER BY id DESC
                    """,
                    (category, owner),
                )

            rows = cursor.fetchall()
            keywords = self._tokenize_query(query)

            results = []
            scored = []  # (key_hits, memory)：key_hits 仅用于同分排序

            for row in rows:
                memory = {
                    "id": row[0],
                    "key": row[1],
                    "value": row[2],
                    "source": row[3],
                    "confidence": row[4],
                    "created_at": row[5],
                    "updated_at": row[6],
                    "category": row[7],
                    "reason": row[8],
                }

                score = 0
                key_hits = 0

                # key 匹配：每词元权重 2×词长（中文单字=2，与原行为
                # 一致；英文片段需整段出现才计分）。
                for keyword in keywords:
                    if keyword in memory["key"]:
                        score += 2 * len(keyword)
                        key_hits += 1

                # value 匹配：每词元权重 1×词长（中文单字=1）。
                for keyword in keywords:
                    if keyword in memory["value"]:
                        score += len(keyword)

                if score >= min_score:
                    memory["score"] = score
                    scored.append((key_hits, memory))

            # 同分时的二级排序：key 命中数更多者优先（key 证据比
            # value 里的字符偶合更可靠）；两者仍相同则保持原行序
            # （id 倒序），排序稳定。
            scored.sort(
                key=lambda item: (item[1]["score"], item[0]),
                reverse=True,
            )

            return [memory for _, memory in scored[:limit]]

        finally:
            connection.close()

    def get_memories_by_category(self, category: str, limit: int = 3, *, user_id: str):
        """读取**当前用户**指定分类的最近几条长期记忆（参数化查询）。

        用于意图检索的兜底：用户明确在问某一类信息
        （例如职业目标），但按 key 查询和文本匹配都没有命中时，
        直接返回该分类下最近的记忆。
        """
        owner = require_user_id(user_id)

        connection = self.get_connection()

        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT
                    id, key, value, source, confidence,
                    created_at, updated_at, category, reason
                FROM memories
                WHERE category = ? AND user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (category, owner, limit),
            )

            rows = cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "key": row[1],
                    "value": row[2],
                    "source": row[3],
                    "confidence": row[4],
                    "created_at": row[5],
                    "updated_at": row[6],
                    "category": row[7],
                    "reason": row[8],
                }
                for row in rows
            ]

        finally:
            connection.close()

    def save_memory(
        self,
        key: str,
        value: str,
        source: str = "user",
        confidence: float = 1.0,
        category: Optional[str] = None,
        reason: Optional[str] = None,
        connection=None,
        *,
        user_id: str,
    ):
        """
        保存或更新一条长期记忆。

        返回：
        add    = 新增记忆
        update = 更新已有记忆
        ignore = 内容相同，不修改

        connection：
        - 不传（None）：自行创建连接、自行 commit、用完即关（原有行为）。
        - 传入外部连接：读写复用同一个连接，且不 commit、不关闭，
          由调用方统一提交或回滚（用于批量保存的原子性）。
        """
        if not key or not key.strip():
            raise ValueError("记忆 key 不能为空")

        if not isinstance(value, str) or not value.strip():
            raise ValueError("记忆 value 不能为空")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence 必须在 0 到 1 之间")

        # 未指定分类时，根据 key 推断。
        if category is None:
            category = self.infer_category(key)

        if not category.strip():
            raise ValueError("category 不能为空")

        # 未提供原因时使用默认说明，不伪装成大模型生成的原因。
        if reason is None:
            reason = "未提供具体保存原因"

        if not reason.strip():
            reason = "未提供具体保存原因"

        owns_connection = connection is None

        owner = require_user_id(user_id)

        # 复用外部连接时，查询与写入必须落在同一个连接上，
        # 否则读不到本事务里尚未提交的改动（批内重复 key 会被误判为新增）。
        existing_memory = self.find_memory(key, connection=connection, user_id=owner)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if owns_connection:
            connection = self.get_connection()

        try:
            cursor = connection.cursor()

            if existing_memory is None:
                cursor.execute(
                    """
                    INSERT INTO memories (
                        key, value, source, confidence,
                        created_at, updated_at, category, reason, user_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        value,
                        source,
                        confidence,
                        now,
                        now,
                        category,
                        reason,
                        owner,
                    ),
                )

                if owns_connection:
                    connection.commit()

                return "add"

            # 内容相同时，保留原有 ignore 语义。
            if existing_memory["value"] == value:
                return "ignore"

            cursor.execute(
                """
                UPDATE memories
                SET
                    value = ?,
                    source = ?,
                    confidence = ?,
                    updated_at = ?,
                    category = ?,
                    reason = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    value,
                    source,
                    confidence,
                    now,
                    category,
                    reason,
                    existing_memory["id"],
                    owner,
                ),
            )

            if owns_connection:
                connection.commit()

            return "update"

        finally:
            if owns_connection:
                connection.close()

    def get_memories(self, limit: int = 100, *, user_id: str):
        """读取**当前用户**的长期记忆，包含分类和保存原因。"""
        owner = require_user_id(user_id)

        connection = self.get_connection()

        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT
                    id, key, value, source, confidence,
                    created_at, updated_at, category, reason
                FROM memories
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (owner, limit),
            )

            rows = cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "key": row[1],
                    "value": row[2],
                    "source": row[3],
                    "confidence": row[4],
                    "created_at": row[5],
                    "updated_at": row[6],
                    "category": row[7],
                    "reason": row[8],
                }
                for row in rows
            ]

        finally:
            connection.close()


if __name__ == "__main__":
    memory = Memory()

    print("Memory 模块自检")
    print("数据库字段已兼容新结构。")
    print("自 PAI-AUTH-MVP 起，私有数据读写必须显式传入 user_id，")
    print("因此本自检只验证表结构，不读取任何用户数据。")
