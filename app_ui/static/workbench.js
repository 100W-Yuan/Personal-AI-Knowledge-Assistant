"use strict";

/* =========================================================
   统一工作台外壳脚本（R3-UI-03）

   职责边界：
   1. 只负责 hash 路由、面板显示、iframe 的创建与显示切换、导航选中态。
   2. 不读取、不写入 iframe 内部文档，不与其内部脚本通信。
   3. 不调用任何业务接口、不执行模型调用、不访问数据库。
   4. 不修改 iframe 内部的 hash；业务页自己的 #/orders 只存在于 iframe 内。
   5. 不引入任何第三方库。

   路由表：home / ai / business / settings / help
   - ai、business 各创建一次 iframe 并常驻保留，切换时只切可见性，
     不删除 DOM、不重设 src、不重新加载，因此内部页面状态得以保留。
   - settings、help 为显式占位页，不做假功能。
   ========================================================= */

(function () {
    var ROUTES = {
        home: { pane: "home", title: "工作台" },
        ai: { pane: "frame", title: "AI 助手", src: "/" },
        business: { pane: "frame", title: "业务管理", src: "/business-ui" },
        /* R1-2-S2-K1：知识库入口。独立 iframe 复用（ensureFrame 每模块只建一次、
           不重设 src）；页面访问由服务端 get_page_user_strict 守卫，
           所有已登录角色均可见导航与可进入。 */
        knowledge: { pane: "frame", title: "知识库", src: "/knowledge-ui" },
        /* R1-2-S1：管理员入口的占位路由。真实账户管理页在后续阶段接入；
           入口是否显示由 /auth/me 的 role 决定，是否允许访问由服务端决定。 */
        admin: { pane: "admin", title: "账户管理" },
        settings: { pane: "settings", title: "系统设置" },
        help: { pane: "help", title: "帮助中心" }
    };

    var DEFAULT_ROUTE = "home";

    var view = document.getElementById("main");
    var panes = document.querySelectorAll("[data-pane]");
    var navItems = document.querySelectorAll(".nav-item[data-route]");
    var frameHost = document.getElementById("frame-host");
    var side = document.getElementById("side");
    var scrim = document.getElementById("scrim");
    var menuBtn = document.getElementById("menu-btn");
    var sysHost = document.getElementById("sys-host");
    var skipBtn = document.querySelector("[data-skip]");

    var activeRoute = null;
    var frames = {};

    /* ---------- 身份与退出（R1-2-S1） ----------
       只消费 GET /auth/me 的结果做导航展示与顶层跳转；
       权限始终由服务端守卫决定，前端判断不构成安全边界。
       fail-closed 原则：身份未确认/未知角色时，管理员入口保持隐藏。 */

    var IDENTITY_ENDPOINT = "/auth/me";
    var LOGOUT_ENDPOINT = "/auth/logout";
    var LOGIN_PAGE = "/login";
    var CHANGE_PASSWORD_PAGE = "/change-password";

    var ROLE_LABELS = { admin: "管理员", user: "普通用户" };

    var identityBox = document.getElementById("identity");
    var identityName = document.getElementById("identity-name");
    var identityRole = document.getElementById("identity-role");
    var logoutBtn = document.getElementById("logout-btn");
    var adminNavItems = document.querySelectorAll(
        '.nav-item[data-requires-role="admin"]'
    );

    function showIdentityState(text, roleKey) {
        if (!identityRole) {
            return;
        }
        identityRole.textContent = text;
        identityRole.setAttribute("data-role", roleKey);
    }

    function revealAdminNav() {
        var i;
        for (i = 0; i < adminNavItems.length; i += 1) {
            adminNavItems[i].hidden = false;
        }
    }

    function applyIdentity(user) {
        if (!identityBox) {
            return;
        }

        identityBox.hidden = false;
        if (identityName) {
            identityName.textContent = user.username || "—";
        }

        if (user.role === "admin") {
            showIdentityState(ROLE_LABELS.admin, "admin");
            revealAdminNav();
            return;
        }

        if (user.role === "user") {
            showIdentityState(ROLE_LABELS.user, "user");
            /* 管理员入口保持 hidden（初始即 hidden，不显式展示）。 */
            return;
        }

        /* 未知角色：明确报错，不默认提权，管理员入口保持隐藏。 */
        showIdentityState("未知角色，权限未确认", "error");
    }

    function loadIdentity() {
        if (!identityBox) {
            return;
        }

        fetch(IDENTITY_ENDPOINT, {
            credentials: "same-origin",
            cache: "no-store"
        })
            .then(function (response) {
                if (response.status === 401) {
                    /* 未登录/会话失效：顶层回登录页。 */
                    window.location.replace(LOGIN_PAGE);
                    return null;
                }
                if (!response.ok) {
                    throw new Error("identity-failed");
                }
                return response.json();
            })
            .then(function (payload) {
                if (!payload || !payload.user) {
                    return null;
                }
                if (payload.user.must_change_password === true) {
                    /* 强制改密：顶层跳转，绝不让改密页嵌进外壳/iframe。 */
                    window.location.replace(CHANGE_PASSWORD_PAGE);
                    return null;
                }
                applyIdentity(payload.user);
                return null;
            })
            .catch(function () {
                /* 身份确认失败（网络/服务异常）：报错并保持最小导航，
                   管理员入口保持隐藏。 */
                identityBox.hidden = false;
                showIdentityState("身份确认失败", "error");
            });
    }

    function doLogout() {
        if (!logoutBtn || logoutBtn.disabled) {
            return;
        }

        logoutBtn.disabled = true;

        fetch(LOGOUT_ENDPOINT, {
            method: "POST",
            credentials: "same-origin"
        })
            .then(function (response) {
                if (response.ok) {
                    /* 服务端已撤销会话并清除 Cookie 后才离开。 */
                    window.location.replace(LOGIN_PAGE);
                    return;
                }
                logoutBtn.disabled = false;
                showIdentityState(
                    "退出失败（HTTP " + response.status + "）",
                    "error"
                );
            })
            .catch(function () {
                logoutBtn.disabled = false;
                showIdentityState("退出失败：网络异常", "error");
            });
    }

    /* ---------- 路由解析 ---------- */

    function parseRoute() {
        var raw = window.location.hash.replace(/^#\/?/, "");
        var name = raw.split("/")[0];

        if (Object.prototype.hasOwnProperty.call(ROUTES, name)) {
            return name;
        }

        return DEFAULT_ROUTE;
    }

    /* ---------- 导航选中态 ---------- */

    function setNav(route) {
        var i;
        var item;

        for (i = 0; i < navItems.length; i += 1) {
            item = navItems[i];

            if (item.getAttribute("data-route") === route) {
                item.setAttribute("aria-current", "page");
            } else {
                item.removeAttribute("aria-current");
            }
        }
    }

    /* ---------- iframe 持久层 ----------
       每个模块只创建一次，之后一直保留在 DOM 中。
       离开模块只移除 is-active（配合 CSS 切 visibility），
       不删除节点、不重设 src，因此不会重新加载。
       仅设置 src 与 title，不触碰内部文档。 */

    function ensureFrame(route, src, title) {
        var el = frames[route];

        if (el) {
            /* 已存在：只同步标题，绝不重设 src，避免触发重新加载。 */
            el.setAttribute("title", title);
            return el;
        }

        el = document.createElement("iframe");

        el.className = "frame";
        el.setAttribute("src", src);
        el.setAttribute("title", title);
        el.setAttribute("loading", "eager");
        el.setAttribute("data-frame", route);

        frameHost.appendChild(el);
        frames[route] = el;

        return el;
    }

    function activateFrame(route) {
        var key;

        for (key in frames) {
            if (Object.prototype.hasOwnProperty.call(frames, key)) {
                if (key === route) {
                    frames[key].classList.add("is-active");
                } else {
                    frames[key].classList.remove("is-active");
                }
            }
        }
    }

    function hideFrames() {
        var key;

        for (key in frames) {
            if (Object.prototype.hasOwnProperty.call(frames, key)) {
                frames[key].classList.remove("is-active");
            }
        }
    }

    /* ---------- 渲染 ---------- */

    function apply(route) {
        var cfg;
        var i;

        if (activeRoute === route) {
            return;
        }

        activeRoute = route;
        cfg = ROUTES[route];

        for (i = 0; i < panes.length; i += 1) {
            if (panes[i].getAttribute("data-pane") === cfg.pane) {
                panes[i].classList.add("is-on");
            } else {
                panes[i].classList.remove("is-on");
            }
        }

        if (cfg.pane === "frame") {
            ensureFrame(route, cfg.src, cfg.title);
            activateFrame(route);
        } else {
            hideFrames();
        }

        setNav(route);
        document.title = cfg.title + " · MY AI 工作台";

        if (view) {
            view.scrollTop = 0;
        }

        closeSide();
    }

    /* ---------- 移动端抽屉 ---------- */

    function openSide() {
        if (!side || !menuBtn) {
            return;
        }

        side.classList.add("is-open");
        scrim.classList.add("is-on");
        menuBtn.setAttribute("aria-expanded", "true");
    }

    function closeSide() {
        if (!side || !menuBtn) {
            return;
        }

        side.classList.remove("is-open");
        scrim.classList.remove("is-on");
        menuBtn.setAttribute("aria-expanded", "false");
    }

    function toggleSide() {
        if (side.classList.contains("is-open")) {
            closeSide();
        } else {
            openSide();
        }
    }

    /* ---------- 初始化 ---------- */

    function init() {
        /* 地址栏展示真实访问地址，不使用任何写死的值。 */
        if (sysHost && window.location.host) {
            sysHost.textContent = window.location.host;
        }

        /* 首次进入且没有 hash 时补上默认路由，保证刷新后地址与视图一致。 */
        if (!window.location.hash) {
            window.history.replaceState(null, "", "#/" + DEFAULT_ROUTE);
        }

        apply(parseRoute());

        window.addEventListener("hashchange", function () {
            apply(parseRoute());
        });

        if (logoutBtn) {
            logoutBtn.addEventListener("click", doLogout);
        }

        /* 启动即确认身份：填充用户区、控制管理员入口、处理强制改密与登录态。 */
        loadIdentity();

        if (menuBtn) {
            menuBtn.addEventListener("click", toggleSide);
        }

        if (scrim) {
            scrim.addEventListener("click", closeSide);
        }

        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape") {
                closeSide();
            }
        });

        window.addEventListener("resize", function () {
            if (window.innerWidth > 960) {
                closeSide();
            }
        });

        if (skipBtn && view) {
            skipBtn.addEventListener("click", function () {
                view.focus();
            });
        }
    }

    init();
}());
