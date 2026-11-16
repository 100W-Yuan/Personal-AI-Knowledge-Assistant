"""FastAPI 鉴权依赖：会话解析、角色校验、同源校验。

设计要点：
- 身份只从 Cookie 里的会话令牌解析，**不接受** query/body 里的 user_id；
- 页面请求未登录 → 302 跳 /login；接口请求未登录 → 401 JSON；
- 管理接口额外要求 ``role == "admin"`` → 否则 403。
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request

from auth.service import AuthService

SESSION_COOKIE = "pai_session"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

_service = AuthService()


def get_service() -> AuthService:
    return _service


def _unauthorized(message: str = "未登录或登录已失效") -> HTTPException:
    return HTTPException(
        status_code=401,
        detail={"type": "unauthenticated", "message": message},
        headers={"WWW-Authenticate": "Cookie"},
    )


def get_session_token(request: Request) -> str | None:
    return request.cookies.get(SESSION_COOKIE)


def get_optional_user(
    request: Request,
    service: AuthService = Depends(get_service),
) -> dict | None:
    """已登录返回用户信息，未登录返回 None（用于 /auth/status 等公开接口）。"""
    token = get_session_token(request)
    if not token:
        return None
    return service.validate_session(token)


def get_current_user(
    request: Request,
    service: AuthService = Depends(get_service),
) -> dict:
    user = get_optional_user(request, service)
    if user is None:
        raise _unauthorized()
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail={"type": "forbidden", "message": "需要管理员权限"},
        )
    return user


def get_page_user(
    request: Request,
    service: AuthService = Depends(get_service),
) -> dict:
    """页面路由专用：未登录时 302 跳转登录页，而不是返回 401。"""
    user = get_optional_user(request, service)
    if user is None:
        target = "/login"
        if request.url.path not in ("/", "/login"):
            target = "/login?next=" + request.url.path
        raise HTTPException(status_code=302, headers={"Location": target})
    return user


def require_password_changed(user: dict = Depends(get_current_user)) -> dict:
    """强制改密守卫（API）：``must_change_password=1`` 时拦截业务接口。

    - 拦截发生在任何业务逻辑之前（FastAPI 先解析依赖，再进入端点函数），
      不会先调用 Agent / 访问业务库 / 检索知识库 / 消耗模型额度；
    - 对管理员同样生效（管理员被重置密码后也必须先改密）；
    - ``/auth/me``、``/auth/logout``、``/auth/change-password`` 不挂本守卫，
      保证强制改密用户始终能查询自身状态、登出与完成改密。

    判断依据是服务端数据库中经 Session 验证的用户状态（``get_current_user``
    的返回值），不依赖前端上报的任何字段。
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
    return user


def get_page_user_strict(
    request: Request,
    service: AuthService = Depends(get_service),
) -> dict:
    """页面守卫（严格版）：未登录 302 /login；强制改密 302 /change-password。

    与 ``get_page_user`` 的唯一区别：已登录但 ``must_change_password=1``
    的用户访问受保护页面时，服务端直接 302 到改密页，而不是返回页面内容。
    改密页本身（``GET /change-password``）继续使用 ``get_page_user``，
    保证强制改密用户与自愿改密的普通用户都能打开它。
    """
    user = get_optional_user(request, service)
    if user is None:
        target = "/login"
        if request.url.path not in ("/", "/login"):
            target = "/login?next=" + request.url.path
        raise HTTPException(status_code=302, headers={"Location": target})
    if user.get("must_change_password"):
        raise HTTPException(
            status_code=302,
            headers={"Location": "/change-password"},
        )
    return user


def _allowed_origins(request: Request) -> set[str]:
    """允许的来源集合：当前请求自身的 origin + 可选 PAI_PUBLIC_ORIGIN。"""
    host = request.headers.get("host") or ""
    origins: set[str] = set()
    if host:
        origins.add("%s://%s" % (request.url.scheme, host))
        # 本地与反向代理场景下 scheme 可能被代理改写，额外接受 https 变体
        origins.add("https://%s" % host)
    configured = os.environ.get("PAI_PUBLIC_ORIGIN", "")
    for item in configured.split(","):
        value = item.strip().rstrip("/")
        if value:
            origins.add(value)
    return origins


def enforce_same_origin(request: Request) -> None:
    """写请求的同源校验：Origin > Referer > 拒绝。

    只对状态变更方法生效；缺失两个来源头时**拒绝**（默认不信任）。
    """
    if request.method.upper() in SAFE_METHODS:
        return

    allowed = _allowed_origins(request)
    origin = request.headers.get("origin")
    referer = request.headers.get("referer")

    candidate = None
    if origin:
        candidate = origin.rstrip("/")
    elif referer:
        parts = urlsplit(referer)
        if parts.scheme and parts.netloc:
            candidate = "%s://%s" % (parts.scheme, parts.netloc)

    if not candidate or candidate not in allowed:
        raise HTTPException(
            status_code=403,
            detail={
                "type": "origin_rejected",
                "message": "跨站请求被拒绝：缺少或非法的 Origin/Referer",
            },
        )


def require_same_origin(request: Request) -> None:
    """作为 Depends 使用的包装（返回 None 即可）。"""
    enforce_same_origin(request)
    return None
