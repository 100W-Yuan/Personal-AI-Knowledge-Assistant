/* R1-2-S2-K3A：知识库视图脚本（只读功能 + 管理员上传）。
   数据接口仅两个（同一源，均已登录可读）：
     - GET  /knowledge/documents  文档列表（with_index=false 默认；=true 仅读取已有索引状态）
     - GET  /knowledge/search     知识检索（仅用户提交时发起）
   写接口仅一个，且仅管理员可用：
     - POST /knowledge/documents  上传文档（multipart/form-data，字段名 file）
   本页没有删除、覆盖、批量操作或索引重建：不上传以外任何写操作。
   渲染安全：后端返回的文件名、片段正文、上下文与错误文本一律作为纯文本
   节点写入（textContent），不拼接标记，也不执行任何后端返回的内容。
   权限：前端角色判断只用于 UX；服务端 require_admin 是最终判定方。 */

(function () {
    "use strict";

    var IDENTITY_ENDPOINT = "/auth/me";
    var LOGIN_PAGE = "/login";
    var CHANGE_PASSWORD_PAGE = "/change-password";
    var DOCUMENTS_ENDPOINT = "/knowledge/documents";
    var SEARCH_ENDPOINT = "/knowledge/search";
    var UPLOAD_ENDPOINT = "/knowledge/documents";
    var DEFAULT_TOP_K = 3;
    var MIN_TOP_K = 1;
    var MAX_TOP_K = 10;

    // 上传前端预检参数（与后端 ALLOWED_KNOWLEDGE_SUFFIXES / MAX_KNOWLEDGE_FILE_BYTES 对齐；
    // 前端预检只为少发无效请求，后端校验才是最终判定）。
    var UPLOAD_SUFFIXES = [".md", ".txt", ".pdf", ".docx"];
    var UPLOAD_MAX_BYTES = 10 * 1024 * 1024;
    var UPLOAD_NOTICE =
        "文件已保存。上传成功不代表文档已经完成解析和索引，实际状态请查看索引状态。";

    // ---- DOM ----
    var identityStatus = document.getElementById("identity-status");
    var docCount = document.getElementById("doc-count");
    var docState = document.getElementById("doc-state");
    var docTableWrap = document.getElementById("doc-table-wrap");
    var docRows = document.getElementById("doc-rows");
    var refreshBtn = document.getElementById("refresh-docs");
    var checkIndexBtn = document.getElementById("check-index");
    var searchForm = document.getElementById("search-form");
    var searchInput = document.getElementById("search-input");
    var searchTopk = document.getElementById("search-topk");
    var searchState = document.getElementById("search-state");
    var searchResults = document.getElementById("search-results");

    // 管理员上传区
    var uploadSection = document.getElementById("upload-section");
    var uploadInput = document.getElementById("upload-input");
    var uploadSubmit = document.getElementById("upload-submit");
    var uploadFileInfo = document.getElementById("upload-file-info");
    var uploadState = document.getElementById("upload-state");
    var uploadResult = document.getElementById("upload-result");

    // 是否已展示索引状态列：刷新列表时保持一致，默认不读取索引。
    var indexVisible = false;
    var loadingDocs = false;
    var searching = false;
    // 当前会话是否为管理员（仅用于显示上传区；服务端权限才是最终边界）。
    var isAdmin = false;
    var uploading = false;

    // ============================================================
    // 通用：请求与错误描述
    // ============================================================

    function detailOf(data) {
        return data && data.detail && typeof data.detail === "object" ? data.detail : null;
    }

    function describeHttpError(status, data) {
        var detail = detailOf(data);
        var type = detail && detail.type ? String(detail.type) : "";
        var message = detail && detail.message ? String(detail.message) : "";

        if (status === 401) {
            return "登录状态已失效，请重新登录";
        }
        if (status === 403 && type === "password_change_required") {
            return "需要先修改初始密码";
        }
        if (status === 503 && type === "knowledge_unavailable") {
            return "知识库服务暂不可用" + (message ? "：" + message : "");
        }
        if (status === 400 || status === 422) {
            return "请求参数有误" + (message ? "：" + message : "");
        }
        if (status === 500) {
            return "服务端错误" + (message ? "：" + message : "");
        }
        return "请求失败（HTTP " + status + "）" + (message ? "：" + message : "");
    }

    /* 统一请求：返回 {ok:true, data} 或 {ok:false, error}。
       401 / 强制改密按 K1 语义顶层跳转，不把会话问题伪装成业务错误。 */
    function requestJSON(url) {
        return fetch(url, {
            credentials: "same-origin",
            headers: { "Accept": "application/json" }
        }).then(function (res) {
            if (res.status === 401) {
                window.location.replace(LOGIN_PAGE);
                return { ok: false, error: "登录状态已失效，正在跳转登录页…" };
            }
            return res.text().then(function (bodyText) {
                var data = null;
                if (bodyText) {
                    try {
                        data = JSON.parse(bodyText);
                    } catch (e) {
                        data = null;
                    }
                }
                var detail = detailOf(data);
                if (
                    res.status === 403 &&
                    detail &&
                    String(detail.type || "") === "password_change_required"
                ) {
                    window.location.replace(CHANGE_PASSWORD_PAGE);
                    return { ok: false, error: "需要先修改初始密码，正在跳转…" };
                }
                if (!res.ok) {
                    return { ok: false, error: describeHttpError(res.status, data) };
                }
                if (data === null) {
                    return { ok: false, error: "服务端返回的内容不是有效 JSON" };
                }
                return { ok: true, data: data };
            });
        }).catch(function () {
            return { ok: false, error: "网络异常，无法连接服务端" };
        });
    }

    // ============================================================
    // 身份（沿用 K1：/auth/me；未知角色 fail-closed）
    // ============================================================

    function setIdentity(text, stateClass) {
        if (!identityStatus) return;
        identityStatus.textContent = text;
        identityStatus.className = "identity-status" + (stateClass ? " " + stateClass : "");
    }

    function loadIdentity() {
        return fetch(IDENTITY_ENDPOINT, { credentials: "same-origin" })
            .then(function (res) {
                if (res.status === 401) {
                    window.location.replace(LOGIN_PAGE);
                    return null;
                }
                if (!res.ok) {
                    setIdentity("身份确认失败", "is-error");
                    applyRole(null);
                    return null;
                }
                return res.json().then(function (data) {
                    var u = data && data.user ? data.user : data;
                    if (u && u.must_change_password === true) {
                        window.location.replace(CHANGE_PASSWORD_PAGE);
                        return null;
                    }
                    var role = u && u.role;
                    applyRole(role);
                    if (role === "admin") {
                        setIdentity("已登录：" + displayName(u) + "（管理员）", "is-ok");
                    } else if (role === "user") {
                        setIdentity("已登录：" + displayName(u) + "（普通用户）", "is-ok");
                    } else {
                        // 未知角色 fail-closed：不给出管理员身份，也不开放任何额外能力。
                        setIdentity("已登录：未知角色（权限未确认）", "is-error");
                    }
                    return null;
                });
            })
            .catch(function () {
                setIdentity("身份确认失败", "is-error");
                applyRole(null);
            });
    }

    function displayName(u) {
        return u && u.username ? String(u.username) : "未知用户";
    }

    /* 角色 → 上传区可见性。fail-closed：只有明确的 admin 才显示；
       user / 未知角色 / 身份检查失败一律保持隐藏。 */
    function applyRole(role) {
        isAdmin = role === "admin";
        if (uploadSection) {
            uploadSection.hidden = !isAdmin;
        }
        if (!isAdmin && uploadInput) {
            // 普通用户即使通过脚本强行调用也不会带上已选文件。
            uploadInput.value = "";
        }
        if (!isAdmin && uploadSubmit) {
            uploadSubmit.disabled = true;
        }
        if (!isAdmin && uploadResult) {
            uploadResult.hidden = true;
        }
    }

    // ============================================================
    // 文档列表
    // ============================================================

    function setDocState(text, kind) {
        if (!docState) return;
        docState.textContent = text;
        docState.className = "placeholder" + (kind ? " is-" + kind : "");
        docState.hidden = false;
    }

    function setIndexColumnsVisible(visible) {
        indexVisible = visible;
        var cols = document.querySelectorAll("[data-index-col]");
        for (var i = 0; i < cols.length; i += 1) {
            cols[i].hidden = !visible;
        }
    }

    function formatBytes(size) {
        var n = typeof size === "number" ? size : Number(size);
        if (!isFinite(n) || n < 0) return "未知";
        if (n < 1024) return n + " B";
        if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
        return (n / (1024 * 1024)).toFixed(1) + " MB";
    }

    function textOrUnknown(value) {
        if (value === null || value === undefined || value === "") return "未知";
        return String(value);
    }

    function parseStatusInfo(status) {
        if (status === "indexed") return { text: "已索引", cls: "status-pill is-indexed" };
        if (status === "not_indexed") return { text: "未索引", cls: "status-pill is-not-indexed" };
        // unknown 与任何未识别取值一律显示「未知」，不推断为已索引。
        return { text: "未知", cls: "status-pill is-unknown" };
    }

    function chunkCountText(value) {
        if (typeof value === "number" && isFinite(value) && value >= 0) {
            return String(Math.floor(value));
        }
        return "未知";
    }

    function cell(text, extraClass) {
        var td = document.createElement("td");
        td.textContent = text;
        if (extraClass) td.className = extraClass;
        return td;
    }

    function buildRow(doc, withIndex) {
        var tr = document.createElement("tr");
        // document_id 只作为行标识保存，绝不拼成任何路径。
        tr.setAttribute("data-document-id", doc && doc.document_id ? String(doc.document_id) : "");

        tr.appendChild(cell(textOrUnknown(doc && doc.filename)));
        tr.appendChild(cell(textOrUnknown(doc && doc.file_type)));

        var sizeCell = cell(formatBytes(doc ? doc.file_size : null), "cell-num");
        tr.appendChild(sizeCell);

        tr.appendChild(cell(textOrUnknown(doc && doc.updated_at)));

        if (withIndex) {
            var info = parseStatusInfo(doc ? doc.parse_status : null);
            var statusTd = document.createElement("td");
            var pill = document.createElement("span");
            pill.className = info.cls;
            pill.textContent = info.text;
            statusTd.appendChild(pill);
            tr.appendChild(statusTd);

            tr.appendChild(cell(chunkCountText(doc ? doc.chunk_count : null), "cell-num"));
        }
        return tr;
    }

    function renderDocuments(data, withIndex) {
        var documents = Array.isArray(data.documents) ? data.documents : [];
        var count = typeof data.count === "number" ? data.count : documents.length;

        if (docCount) docCount.textContent = "共 " + count + " 个文档";

        if (docTableWrap) docTableWrap.hidden = true;
        if (docRows) {
            while (docRows.firstChild) {
                docRows.removeChild(docRows.firstChild);
            }
        }

        if (data.directory_exists === false) {
            setDocState("知识库目录不存在或尚未创建，因此没有可列出的文档。", "empty");
            return;
        }

        if (documents.length === 0) {
            setDocState("知识库暂无文档（0 个）。", "empty");
            return;
        }

        setIndexColumnsVisible(withIndex);

        if (docRows) {
            for (var i = 0; i < documents.length; i += 1) {
                docRows.appendChild(buildRow(documents[i] || {}, withIndex));
            }
        }
        if (docTableWrap) docTableWrap.hidden = false;
        setDocState("已加载 " + count + " 个文档。", "ok");
    }

    /* 加载文档列表。
       withIndex=false：默认视图，任何情况下都不读取索引；
       withIndex=true ：仅在用户主动点击「查看索引状态」时使用，
                        只读取已有索引状态，不触发索引构建。 */
    function loadDocuments(withIndex) {
        if (loadingDocs) return Promise.resolve();
        loadingDocs = true;
        if (refreshBtn) refreshBtn.disabled = true;
        if (checkIndexBtn) checkIndexBtn.disabled = true;

        if (docTableWrap) docTableWrap.hidden = true;
        setDocState(
            withIndex ? "正在读取已有索引状态…" : "正在加载文档列表…",
            "loading"
        );

        var url = DOCUMENTS_ENDPOINT + "?with_index=" + (withIndex ? "true" : "false");

        return requestJSON(url).then(function (result) {
            loadingDocs = false;
            if (refreshBtn) refreshBtn.disabled = false;
            if (checkIndexBtn) checkIndexBtn.disabled = false;

            if (!result.ok) {
                if (docCount) docCount.textContent = "—";
                setDocState("文档列表加载失败：" + result.error, "error");
                return;
            }
            var data = result.data;
            if (!data || data.ok !== true) {
                var detail = data && data.error ? String(data.error) : "响应格式异常";
                setDocState("文档列表加载失败：" + detail, "error");
                return;
            }
            renderDocuments(data, withIndex);
        });
    }

    // ============================================================
    // 知识搜索
    // ============================================================

    function setSearchState(text, kind) {
        if (!searchState) return;
        searchState.textContent = text;
        searchState.className = "placeholder" + (kind ? " is-" + kind : "");
        searchState.hidden = false;
    }

    function clampTopK(value) {
        var n = parseInt(value, 10);
        if (!isFinite(n)) return DEFAULT_TOP_K;
        if (n < MIN_TOP_K) return MIN_TOP_K;
        if (n > MAX_TOP_K) return MAX_TOP_K;
        return n;
    }

    function formatScore(score) {
        // 显示原始数值的合理小数格式；不换算成百分比（取值范围未经确认）。
        var n = Number(score);
        if (!isFinite(n)) return "—";
        return n.toFixed(4);
    }

    function describeSearchFailure(data) {
        var stage = data && data.stage ? String(data.stage) : "";
        var detail = data && data.error ? String(data.error) : "";
        var prefix;
        if (stage === "input") {
            prefix = "搜索输入有误，请检查搜索问题";
        } else if (stage === "init") {
            prefix = "知识库初始化失败";
        } else if (stage === "search") {
            prefix = "搜索失败";
        } else {
            prefix = "搜索未完成";
        }
        return prefix + (detail ? "：" + detail : "");
    }

    function clearResults() {
        if (!searchResults) return;
        while (searchResults.firstChild) {
            searchResults.removeChild(searchResults.firstChild);
        }
        searchResults.hidden = true;
    }

    function buildResultCard(item, index) {
        var article = document.createElement("article");
        article.className = "result";

        var head = document.createElement("div");
        head.className = "result__head";

        var source = document.createElement("span");
        source.className = "result__source";
        source.textContent = textOrUnknown(item && item.source);

        var chunkId = document.createElement("span");
        chunkId.className = "result__chunk";
        chunkId.textContent = "片段 " + textOrUnknown(item && item.chunk_id);

        var score = document.createElement("span");
        score.className = "result__score";
        score.textContent = "相似度 " + formatScore(item ? item.score : null);

        var order = document.createElement("span");
        order.className = "result__chunk";
        order.textContent = "#" + (index + 1);

        head.appendChild(order);
        head.appendChild(source);
        head.appendChild(chunkId);
        head.appendChild(score);

        var body = document.createElement("p");
        body.className = "result__chunk-text";
        body.textContent =
            item && item.chunk ? String(item.chunk) : "（片段内容为空）";

        article.appendChild(head);
        article.appendChild(body);
        return article;
    }

    function buildContextBlock(context) {
        var details = document.createElement("details");
        details.className = "context";
        // 默认折叠：不设置 open 属性。

        var summary = document.createElement("summary");
        summary.textContent = "查看拼接上下文（默认折叠）";

        var pre = document.createElement("pre");
        pre.className = "context__text";
        pre.textContent = String(context);

        details.appendChild(summary);
        details.appendChild(pre);
        return details;
    }

    function renderSearchResults(data) {
        var results = Array.isArray(data.results) ? data.results : [];
        clearResults();

        if (results.length === 0) {
            setSearchState("未检索到相关内容（0 条命中）。", "empty");
            return;
        }

        setSearchState("共 " + results.length + " 条结果。", "ok");
        if (!searchResults) return;

        for (var i = 0; i < results.length; i += 1) {
            searchResults.appendChild(buildResultCard(results[i] || {}, i));
        }
        if (data.context) {
            searchResults.appendChild(buildContextBlock(data.context));
        }
        searchResults.hidden = false;
    }

    /* 提交搜索。只有本函数会发起检索，且只由表单提交事件调用，
       页面加载不会自动检索。 */
    function submitSearch() {
        if (searching) return;

        var raw = searchInput && searchInput.value ? searchInput.value : "";
        var query = raw.trim();
        if (!query) {
            clearResults();
            setSearchState("请先输入搜索内容，再点击「搜索」；空输入不会发送请求。", "warn");
            return;
        }

        var topK = clampTopK(searchTopk ? searchTopk.value : DEFAULT_TOP_K);

        searching = true;
        if (searchForm) {
            var button = searchForm.querySelector("button[type='submit']");
            if (button) button.disabled = true;
        }
        clearResults();
        setSearchState("正在检索…", "loading");

        var url =
            SEARCH_ENDPOINT +
            "?query=" + encodeURIComponent(query) +
            "&top_k=" + topK;

        return requestJSON(url).then(function (result) {
            searching = false;
            if (searchForm) {
                var btn = searchForm.querySelector("button[type='submit']");
                if (btn) btn.disabled = false;
            }

            if (!result.ok) {
                setSearchState("搜索失败：" + result.error, "error");
                return;
            }
            var data = result.data;
            if (!data) {
                setSearchState("搜索失败：响应为空", "error");
                return;
            }
            if (data.ok !== true) {
                // HTTP 200 + ok:false 的业务失败（stage=input/init/search）。
                setSearchState(describeSearchFailure(data), "error");
                return;
            }
            renderSearchResults(data);
        });
    }

    // ============================================================
    // 管理员上传（K3A）
    // ============================================================

    function setUploadState(text, kind) {
        if (!uploadState) return;
        uploadState.textContent = text;
        uploadState.className = "placeholder" + (kind ? " is-" + kind : "");
        uploadState.hidden = false;
    }

    function fileSuffix(name) {
        var text = String(name || "");
        var dot = text.lastIndexOf(".");
        return dot >= 0 ? text.slice(dot).toLowerCase() : "";
    }

    function selectedFile() {
        if (!uploadInput || !uploadInput.files || uploadInput.files.length === 0) {
            return null;
        }
        return uploadInput.files[0];
    }

    function updateUploadFileInfo() {
        var file = selectedFile();
        if (!file) {
            if (uploadFileInfo) uploadFileInfo.textContent = "尚未选择文件。";
            if (uploadSubmit) uploadSubmit.disabled = true;
            return;
        }
        if (uploadFileInfo) {
            uploadFileInfo.textContent =
                "已选择：" + file.name + "（" + formatBytes(file.size) + "）";
        }
        if (uploadSubmit) uploadSubmit.disabled = uploading;
    }

    function resetUploadResult() {
        if (!uploadResult) return;
        while (uploadResult.firstChild) {
            uploadResult.removeChild(uploadResult.firstChild);
        }
        uploadResult.hidden = true;
    }

    function appendMeta(list, label, value) {
        var dt = document.createElement("dt");
        dt.textContent = label;
        var dd = document.createElement("dd");
        dd.textContent = value;
        list.appendChild(dt);
        list.appendChild(dd);
    }

    function renderUploadSuccess(data) {
        resetUploadResult();
        if (!uploadResult) return;

        var list = document.createElement("dl");
        list.className = "upload-result__meta";
        appendMeta(list, "文件名", textOrUnknown(data && data.filename));
        appendMeta(list, "大小", formatBytes(data ? data.bytes : null));
        appendMeta(list, "文档标识", textOrUnknown(data && data.document_id));

        var notice = document.createElement("p");
        notice.className = "upload-result__notice";
        notice.textContent = UPLOAD_NOTICE;

        uploadResult.appendChild(list);
        uploadResult.appendChild(notice);
        uploadResult.hidden = false;
    }

    /* 上传错误映射：先看 HTTP 状态，再读后端 detail.type 与 detail.message，
       不单靠状态码推测原因。 */
    function describeUploadError(status, data) {
        var detail = detailOf(data);
        var type = detail && detail.type ? String(detail.type) : "";
        var message = detail && detail.message ? String(detail.message) : "";

        if (status === 400) {
            if (type === "empty_file") return "文件内容为空，无法上传";
            if (type === "unsupported_format") {
                return "文件格式不支持（仅 .md / .txt / .pdf / .docx）";
            }
            if (type === "invalid_filename") {
                return "文件名不合法" + (message ? "：" + message : "");
            }
            return "文件不合法" + (message ? "：" + message : "");
        }
        if (status === 403) {
            if (type === "password_change_required") {
                return "需要先修改初始密码后才能上传";
            }
            return "权限不足：只有管理员可以上传文档";
        }
        if (status === 409) {
            if (type === "already_exists") {
                return "同名文件已存在，未覆盖" + (message ? "：" + message : "");
            }
            if (type === "duplicate_content") {
                var existing = detail && detail.existing_filename
                    ? String(detail.existing_filename)
                    : "";
                return "内容重复：知识库中已有相同内容的文件"
                    + (existing ? "（" + existing + "）" : "");
            }
            return "冲突：文件已存在或内容重复";
        }
        if (status === 413) return "文件过大：单文件上限 10 MiB";
        if (status === 500) return "服务端写入失败" + (message ? "：" + message : "");
        return "上传失败（HTTP " + status + "）" + (message ? "：" + message : "");
    }

    /* 只在管理员点击「上传」时调用；页面加载不会自动选择或提交文件。 */
    function submitUpload() {
        if (uploading) return;

        if (!isAdmin) {
            // 防御性兜底：正常流程下非管理员看不到上传区。
            setUploadState("权限不足：只有管理员可以上传文档。", "error");
            return;
        }

        var file = selectedFile();
        if (!file) {
            setUploadState("请先选择要上传的文件。", "warn");
            return;
        }

        var suffix = fileSuffix(file.name);
        if (UPLOAD_SUFFIXES.indexOf(suffix) === -1) {
            setUploadState("文件格式不支持（仅 .md / .txt / .pdf / .docx）。", "error");
            return;
        }
        if (file.size === 0) {
            setUploadState("文件内容为空，无法上传。", "error");
            return;
        }
        if (file.size > UPLOAD_MAX_BYTES) {
            setUploadState("文件过大：单文件上限 10 MiB。", "error");
            return;
        }

        var form = new FormData();
        // 字段名必须是 file；不手动设置请求头（交给浏览器生成 multipart 边界）。
        form.append("file", file, file.name);

        uploading = true;
        if (uploadSubmit) uploadSubmit.disabled = true;
        resetUploadResult();
        setUploadState("正在上传…", "loading");

        return fetch(UPLOAD_ENDPOINT, {
            method: "POST",
            body: form,
            credentials: "same-origin"
        }).then(function (res) {
            return res.text().then(function (bodyText) {
                var data = null;
                if (bodyText) {
                    try {
                        data = JSON.parse(bodyText);
                    } catch (e) {
                        data = null;
                    }
                }
                uploading = false;

                if (res.status === 401) {
                    window.location.replace(LOGIN_PAGE);
                    setUploadState("登录状态已失效，正在跳转登录页…", "error");
                    return;
                }
                var detail = detailOf(data);
                if (
                    res.status === 403 &&
                    detail &&
                    String(detail.type || "") === "password_change_required"
                ) {
                    window.location.replace(CHANGE_PASSWORD_PAGE);
                    setUploadState("需要先修改初始密码，正在跳转…", "error");
                    return;
                }
                if (!res.ok) {
                    setUploadState(describeUploadError(res.status, data), "error");
                    if (uploadSubmit) uploadSubmit.disabled = false;
                    return;
                }
                if (!data || data.ok !== true) {
                    setUploadState("上传失败：服务端返回内容异常", "error");
                    if (uploadSubmit) uploadSubmit.disabled = false;
                    return;
                }

                renderUploadSuccess(data);
                setUploadState("上传成功。", "ok");
                // 上传成功 → 按默认 with_index=false 刷新列表；不触发检索或索引构建。
                if (uploadInput) uploadInput.value = "";
                updateUploadFileInfo();
                loadDocuments(false);
            });
        }).catch(function () {
            uploading = false;
            setUploadState("网络请求失败：无法连接服务端，请重试。", "error");
            if (uploadSubmit) uploadSubmit.disabled = false;
        });
    }

    // ============================================================
    // 初始化
    // ============================================================

    function init() {
        if (refreshBtn) {
            refreshBtn.addEventListener("click", function () {
                loadDocuments(indexVisible);
            });
        }
        if (checkIndexBtn) {
            checkIndexBtn.addEventListener("click", function () {
                loadDocuments(true);
            });
        }
        if (searchForm) {
            searchForm.addEventListener("submit", function (event) {
                event.preventDefault();
                submitSearch();
            });
        }

        // 上传：只有用户选择文件 / 点击上传才发生；init 不触发任何上传。
        if (uploadInput) {
            uploadInput.addEventListener("change", updateUploadFileInfo);
        }
        if (uploadSubmit) {
            uploadSubmit.addEventListener("click", submitUpload);
        }

        loadIdentity();
        // 页面初始化只加载一次列表（with_index=false），不请求索引、不检索。
        loadDocuments(false);
    }

    init();
})();
