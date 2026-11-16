"""PAI-AUTH-MVP 认证子系统。

最小实现，只包含：用户表、Session、CLI 账号管理、路由鉴权依赖。
不包含注册、邀请码、邮件、OAuth、JWT、分布式会话、审计平台。
"""

__all__ = ["schema", "security", "service", "deps"]
