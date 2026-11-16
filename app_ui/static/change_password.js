/* =========================================================
   change_password.js · Personal AI 改密页脚本（R1-1-FIX-01）

   边界与信任模型
   --------------
   - 本脚本只负责表单交互与跳转；**是否允许使用业务功能完全由服务端
     会话状态决定**（各业务接口的 password_change_required 守卫），
     前端状态一律不作为可信依据。
   - 页面加载即读取 GET /auth/me：
       * 401        → 回登录页；
       * must_change_password = true  → 强制模式（隐藏"返回工作台"）；
       * must_change_password = false → 自愿模式（显示"返回工作台"）。
   - 提交 POST /auth/change-password：
       * 200 → 改密成功，服务端已保留当前会话、撤销其它会话并清除
               强制标记 → 进入工作台；
       * 401 → 当前密码不正确；
       * 400 → 服务端校验失败（长度等），展示后端 message；
       * 其它 → 通用失败提示。
   - 不写任何 localStorage / cookie；密码不出现在 URL 或日志。
   ========================================================= */

(function () {
    "use strict";

    var ME_ENDPOINT = "/auth/me";
    var CHANGE_ENDPOINT = "/auth/change-password";
    var LOGOUT_ENDPOINT = "/auth/logout";
    var LOGIN_PAGE = "/login";
    var WORKBENCH_PAGE = "/workbench";
    var MIN_PASSWORD_LENGTH = 10;
    var MAX_PASSWORD_LENGTH = 128;

    var form = document.getElementById("cp-form");
    var submitButton = document.getElementById("cp-submit");
    var alertBox = document.getElementById("cp-alert");
    var alertText = document.getElementById("cp-alert-text");
    var modeBox = document.getElementById("cp-mode");
    var modeTitle = document.getElementById("cp-mode-title");
    var modeText = document.getElementById("cp-mode-text");
    var subtitle = document.getElementById("cp-subtitle");
    var backLink = document.getElementById("cp-back");
    var logoutLink = document.getElementById("cp-logout");
    var currentInput = document.getElementById("cp-current");
    var newInput = document.getElementById("cp-new");
    var confirmInput = document.getElementById("cp-confirm");

    function showError(message) {
        alertText.textContent = message;
        alertBox.hidden = false;
    }

    function clearError() {
        alertBox.hidden = true;
        alertText.textContent = "";
    }

    function setFieldError(fieldId, errorId, message) {
        var field = document.getElementById(fieldId);
        var error = document.getElementById(errorId);
        if (!field || !error) {
            return;
        }
        if (message) {
            error.textContent = message;
            error.hidden = false;
            field.classList.add("is-invalid");
            field.setAttribute("aria-invalid", "true");
        } else {
            error.hidden = true;
            error.textContent = "";
            field.classList.remove("is-invalid");
            field.setAttribute("aria-invalid", "false");
        }
    }

    function clearFieldErrors() {
        setFieldError("field-current", "cp-current-err", "");
        setFieldError("field-new", "cp-new-err", "");
        setFieldError("field-confirm", "cp-confirm-err", "");
    }

    function enterForcedMode() {
        subtitle.textContent = "首次登录需要先设置新密码";
        modeTitle.textContent = "首次登录 · 请先修改密码";
        modeText.textContent =
            "管理员设置了临时密码。修改完成前，所有业务功能在服务端保持锁定。";
        modeBox.hidden = false;
        if (backLink) {
            backLink.hidden = true;
        }
    }

    function enterVoluntaryMode() {
        subtitle.textContent = "定期更换密码有助于保护账号安全";
        if (backLink) {
            backLink.hidden = false;
        }
    }

    function loadAccountState() {
        fetch(ME_ENDPOINT, { credentials: "same-origin" })
            .then(function (response) {
                if (response.status === 401) {
                    window.location.replace(LOGIN_PAGE);
                    return null;
                }
                if (!response.ok) {
                    throw new Error("me-failed");
                }
                return response.json();
            })
            .then(function (payload) {
                if (!payload || !payload.user) {
                    return;
                }
                if (payload.user.must_change_password === true) {
                    enterForcedMode();
                } else {
                    enterVoluntaryMode();
                }
            })
            .catch(function () {
                // /auth/me 探测失败不阻断表单：提交以后端判定为准。
                subtitle.textContent = "修改登录密码";
            });
    }

    function validate() {
        clearFieldErrors();
        clearError();

        var ok = true;
        var current = currentInput.value || "";
        var next = newInput.value || "";
        var confirm = confirmInput.value || "";

        if (!current) {
            setFieldError("field-current", "cp-current-err", "请输入当前密码");
            ok = false;
        }
        if (next.length < MIN_PASSWORD_LENGTH) {
            setFieldError("field-new", "cp-new-err",
                "新密码至少 " + MIN_PASSWORD_LENGTH + " 位");
            ok = false;
        } else if (next.length > MAX_PASSWORD_LENGTH) {
            setFieldError("field-new", "cp-new-err",
                "新密码不能超过 " + MAX_PASSWORD_LENGTH + " 位");
            ok = false;
        }
        if (confirm !== next || !confirm) {
            setFieldError("field-confirm", "cp-confirm-err",
                "两次输入的新密码不一致");
            ok = false;
        } else if (next === current) {
            setFieldError("field-new", "cp-new-err",
                "新密码不能与当前密码相同");
            ok = false;
        }
        return ok;
    }

    function setBusy(busy) {
        submitButton.disabled = busy;
        submitButton.classList.toggle("is-loading", busy);
    }

    function submitChange(event) {
        event.preventDefault();
        if (!validate()) {
            return;
        }

        setBusy(true);
        fetch(CHANGE_ENDPOINT, {
            method: "POST",
            credentials: "same-origin",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                current_password: currentInput.value,
                new_password: newInput.value,
            }),
        })
            .then(function (response) {
                if (response.ok) {
                    // 改密成功：服务端已保留当前会话并清除强制标记。
                    window.location.replace(WORKBENCH_PAGE);
                    return null;
                }
                return response.json().then(function (payload) {
                    var message =
                        (payload && (payload.message || payload.error)) || "";
                    if (response.status === 401) {
                        if (payload && payload.error === "unauthenticated") {
                            window.location.replace(LOGIN_PAGE);
                            return null;
                        }
                        showError("当前密码不正确，请重新输入。");
                    } else if (response.status === 400) {
                        showError(message || "新密码不满足要求，请调整后重试。");
                    } else {
                        showError(message || "修改失败，请稍后重试。");
                    }
                    return null;
                });
            })
            .catch(function () {
                showError("网络异常，请确认本地服务正在运行后重试。");
            })
            .then(function () {
                setBusy(false);
            });
    }

    function submitLogout(event) {
        event.preventDefault();
        fetch(LOGOUT_ENDPOINT, {
            method: "POST",
            credentials: "same-origin",
        })
            .then(function () {
                window.location.replace(LOGIN_PAGE);
            })
            .catch(function () {
                window.location.replace(LOGIN_PAGE);
            });
    }

    if (form) {
        form.addEventListener("submit", submitChange);
    }
    if (logoutLink) {
        logoutLink.addEventListener("click", submitLogout);
    }

    loadAccountState();
})();
