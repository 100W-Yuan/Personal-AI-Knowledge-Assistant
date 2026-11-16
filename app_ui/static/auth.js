/* =========================================================
 * Personal AI · 登录页面脚本
 * ---------------------------------------------------------
 * 认证后端已真实接入（Argon2id 密码校验 + HttpOnly 会话 Cookie）。
 * 前端只负责两件事：
 *   1. 提交前的格式校验（不替代后端身份校验）；
 *   2. 把后端返回的状态码翻译成可读提示。
 *
 * 三条硬约束（与实现保持一致）：
 *   1. 不伪造登录成功：只有后端返回 200 才跳转，前端不写任何登录标记。
 *   2. 不保存密码：密码只存在内存中的表单值里，不写入浏览器端任何持久化存储。
 *   3. 不存在注册入口：公开注册已彻底关闭，账号只能由管理员通过 CLI 创建。
 * ========================================================= */

(function () {
    "use strict";

    var STATUS_ENDPOINT = "/auth/status";
    var LOGIN_ENDPOINT = "/auth/login";

    /* 与 HTML 中 minlength / maxlength 保持一致，改一处必须同步另一处。
     * 密码下限与后端 auth/security.py 的 PASSWORD_MIN_LENGTH 一致（10）。 */
    var LIMITS = {
        username: { min: 3, max: 32 },
        password: { min: 10, max: 128 }
    };

    var TEXT = {
        checkingTitle: "正在检查登录状态…",
        checkingNote: "正在与本地认证服务确认当前会话。",

        onlineTitle: "认证服务可用",
        onlineNote: "凭据由本机后端校验，密码不会离开这台机器。",

        signedInTitle: "当前已登录",
        signedInNote: "如需切换账号，直接在下方输入另一个账号的用户名与密码即可。",

        probeFailedTitle: "认证服务状态未知",
        probeFailedNote: "无法读取 GET /auth/status；仍可尝试登录，失败原因以后端返回为准。",

        loginFailedTitle: "登录未成功",
        badCredentials: "用户名或密码不正确。",
        accountDisabled: "该账号已被禁用，请联系管理员。",
        rateLimited: "登录失败次数过多，请 15 分钟后再试。",
        originRejected: "请求来源校验失败：请通过本机地址打开登录页后再提交。",
        networkError: "无法连接认证服务，请确认服务已启动后重试。",
        unexpected: "登录未完成，请稍后重试。"
    };

    var state = {
        probed: false
    };

    /* ---------------------------------------------------------
     * 小工具
     * ------------------------------------------------------- */

    function byId(id) {
        return document.getElementById(id);
    }

    function each(list, fn) {
        Array.prototype.forEach.call(list || [], fn);
    }

    function setText(node, value) {
        if (node) {
            node.textContent = value;
        }
    }

    function strlen(value) {
        /* 按字符数而非 UTF-16 码元计，避免表情/生僻字被算成两个字符。 */
        return Array.from(String(value)).length;
    }

    /* 登录成功后的落点：只接受站内绝对路径，阻断开放重定向。
     * 拒绝 "//evil.com"、"http://…"、反斜杠变体。
     * R1-2-S1：默认落点改为统一工作台；合法的站内 next 参数优先。
     * "/" 仍是 AI 助手独立页面，保留可直达，不做重定向。 */
    function nextTarget() {
        var raw = "";
        try {
            raw = new URLSearchParams(window.location.search).get("next") || "";
        } catch (err) {
            raw = "";
        }
        if (raw.charAt(0) === "/" && raw.charAt(1) !== "/" && raw.indexOf("\\") === -1) {
            return raw;
        }
        return "/workbench";
    }

    /* ---------------------------------------------------------
     * 状态区（顶部）：如实反映当前登录态
     * ------------------------------------------------------- */

    function renderStatusBox(tone, title, note) {
        var box = byId("auth-status");
        if (!box) {
            return;
        }
        box.setAttribute("data-tone", tone);
        setText(byId("auth-status-title"), title);
        setText(byId("auth-status-text"), note);
    }

    function applyStatusPayload(payload) {
        state.probed = true;

        if (payload && payload.authenticated === true && payload.user) {
            var name = payload.user.username || "";
            var role = payload.user.role === "admin" ? "管理员" : "普通用户";
            renderStatusBox(
                "info",
                TEXT.signedInTitle,
                "已登录为 " + name + "（" + role + "）。" + TEXT.signedInNote
            );
            return;
        }

        renderStatusBox("info", TEXT.onlineTitle, TEXT.onlineNote);
    }

    function probeStatus() {
        if (typeof window.fetch !== "function") {
            renderStatusBox("warn", TEXT.probeFailedTitle, TEXT.probeFailedNote);
            return;
        }

        window
            .fetch(STATUS_ENDPOINT, {
                method: "GET",
                headers: { Accept: "application/json" },
                credentials: "same-origin",
                cache: "no-store"
            })
            .then(function (response) {
                if (!response.ok) {
                    throw new Error("HTTP " + response.status);
                }
                return response.json();
            })
            .then(applyStatusPayload)
            .catch(function () {
                state.probed = true;
                renderStatusBox("warn", TEXT.probeFailedTitle, TEXT.probeFailedNote);
            });
    }

    /* ---------------------------------------------------------
     * 密码显示 / 隐藏
     * ------------------------------------------------------- */

    function bindReveal() {
        each(document.querySelectorAll("[data-reveal]"), function (button) {
            var target = byId(button.getAttribute("data-reveal"));
            if (!target) {
                return;
            }

            button.addEventListener("click", function () {
                var showing = button.getAttribute("aria-pressed") === "true";
                var next = !showing;

                target.setAttribute("type", next ? "text" : "password");
                button.setAttribute("aria-pressed", next ? "true" : "false");
                button.setAttribute("aria-label", next ? "隐藏密码" : "显示密码");
                button.setAttribute("title", next ? "隐藏密码" : "显示密码");

                /* 切换后把光标放回输入框尾部，避免焦点留在按钮上连点。 */
                try {
                    var end = target.value.length;
                    target.focus();
                    target.setSelectionRange(end, end);
                } catch (err) {
                    target.focus();
                }
            });
        });
    }

    /* ---------------------------------------------------------
     * 字段级校验
     * ------------------------------------------------------- */

    function fieldWrap(input) {
        return input ? input.closest(".field") : null;
    }

    function errNode(input) {
        return input ? byId(input.id + "-err") : null;
    }

    function clearFieldError(input) {
        if (!input) {
            return;
        }
        var wrap = fieldWrap(input);
        var node = errNode(input);

        if (wrap) {
            wrap.classList.remove("is-invalid");
        }
        if (node) {
            node.hidden = true;
            node.textContent = "";
        }
        input.setAttribute("aria-invalid", "false");
    }

    function showFieldError(input, message) {
        if (!input) {
            return;
        }
        var wrap = fieldWrap(input);
        var node = errNode(input);

        if (wrap) {
            wrap.classList.add("is-invalid");
        }
        if (node) {
            node.textContent = message;
            node.hidden = false;
        }
        input.setAttribute("aria-invalid", "true");
    }

    /* 返回该字段的第一条错误文案；无错误返回空串。 */
    function checkField(input, rule) {
        if (!input) {
            return "";
        }

        var raw = input.value;
        var value = rule.trim === false ? raw : raw.trim();

        if (!value) {
            return rule.required;
        }

        if (rule.noSpace && /\s/.test(value)) {
            return rule.noSpace;
        }

        var size = strlen(value);
        if (size < rule.min || size > rule.max) {
            return rule.range;
        }

        return "";
    }

    function validateOne(input, rule) {
        var message = checkField(input, rule);
        if (message) {
            showFieldError(input, message);
            return false;
        }
        clearFieldError(input);
        return true;
    }

    function bindValidation(input, rule) {
        if (!input) {
            return;
        }

        input.addEventListener("blur", function () {
            if (input.value.trim()) {
                validateOne(input, rule);
            }
        });

        input.addEventListener("input", function () {
            clearFieldError(input);
            hideAlert();
        });
    }

    /* ---------------------------------------------------------
     * 表单级提示区
     * ------------------------------------------------------- */

    function hideAlert() {
        var box = byId("login-alert");
        if (box) {
            box.hidden = true;
        }
    }

    function showAlert(title, message) {
        var box = byId("login-alert");
        if (!box) {
            return;
        }
        setText(byId("login-alert-title"), title);
        setText(byId("login-alert-text"), message);
        box.hidden = false;
    }

    /* ---------------------------------------------------------
     * 提交按钮加载态
     * ------------------------------------------------------- */

    function setLoading(button, loading) {
        if (!button) {
            return;
        }
        button.classList.toggle("is-loading", !!loading);
        button.disabled = !!loading;
        button.setAttribute("aria-busy", loading ? "true" : "false");
    }

    function focusFirstInvalid(inputs) {
        for (var i = 0; i < inputs.length; i += 1) {
            if (inputs[i].getAttribute("aria-invalid") === "true") {
                inputs[i].focus();
                return;
            }
        }
    }

    /* ---------------------------------------------------------
     * 登录页
     * ------------------------------------------------------- */

    /* 把后端响应翻译成提示文案；返回空串表示成功。 */
    function describeFailure(status, body) {
        var detail = (body && body.detail) || {};
        var code = (body && body.error) || detail.type || "";

        if (status === 401) {
            return TEXT.badCredentials;
        }
        if (status === 429) {
            return TEXT.rateLimited;
        }
        if (status === 403) {
            if (code === "origin_rejected") {
                return TEXT.originRejected;
            }
            if (code === "account_disabled") {
                return TEXT.accountDisabled;
            }
            return TEXT.unexpected;
        }
        if (status === 422) {
            return "用户名或密码格式不正确。";
        }
        return TEXT.unexpected + "（HTTP " + status + "）";
    }

    function setupLogin() {
        var form = byId("login-form");
        var button = byId("login-submit");
        if (!form || !button) {
            return;
        }

        var username = byId("login-username");
        var password = byId("login-password");

        var rules = {
            username: {
                required: "请输入用户名",
                range: "用户名需为 " + LIMITS.username.min + "–" + LIMITS.username.max + " 个字符",
                noSpace: "用户名不能包含空格",
                min: LIMITS.username.min,
                max: LIMITS.username.max
            },
            password: {
                required: "请输入密码",
                range: "密码需为 " + LIMITS.password.min + "–" + LIMITS.password.max + " 个字符",
                min: LIMITS.password.min,
                max: LIMITS.password.max,
                trim: false
            }
        };

        bindValidation(username, rules.username);
        bindValidation(password, rules.password);

        form.addEventListener("submit", function (event) {
            /* 始终由脚本提交，避免浏览器原生表单把密码写进 URL。 */
            event.preventDefault();
            hideAlert();

            var okUser = validateOne(username, rules.username);
            var okPass = validateOne(password, rules.password);

            if (!okUser || !okPass) {
                focusFirstInvalid([username, password]);
                return;
            }

            if (typeof window.fetch !== "function") {
                showAlert(TEXT.loginFailedTitle, TEXT.unexpected);
                return;
            }

            setLoading(button, true);

            window
                .fetch(LOGIN_ENDPOINT, {
                    method: "POST",
                    credentials: "same-origin",
                    cache: "no-store",
                    headers: {
                        "Content-Type": "application/json",
                        Accept: "application/json"
                    },
                    body: JSON.stringify({
                        username: username.value.trim(),
                        password: password.value
                    })
                })
                .then(function (response) {
                    return response
                        .json()
                        .catch(function () {
                            return {};
                        })
                        .then(function (body) {
                            return { status: response.status, ok: response.ok, body: body };
                        });
                })
                .then(function (result) {
                    if (result.ok && result.status === 200) {
                        /* 只有后端明确返回 200 才跳转；清空密码后再离开。 */
                        password.value = "";
                        window.location.replace(nextTarget());
                        return;
                    }
                    setLoading(button, false);
                    showAlert(
                        TEXT.loginFailedTitle,
                        describeFailure(result.status, result.body)
                    );
                })
                .catch(function () {
                    setLoading(button, false);
                    showAlert(TEXT.loginFailedTitle, TEXT.networkError);
                });
        });
    }

    /* ---------------------------------------------------------
     * 启动
     * ------------------------------------------------------- */

    function init() {
        bindReveal();
        setupLogin();
        renderStatusBox("warn", TEXT.checkingTitle, TEXT.checkingNote);
        probeStatus();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
