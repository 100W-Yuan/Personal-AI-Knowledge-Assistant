"use strict";

/*
 * 业务管理前端（B4-05B）
 *
 * 约束：
 * 1. 只调用现有 13 个业务接口，不新增业务规则。
 * 2. 不使用 innerHTML 拼接任何用户输入或后端数据。
 * 3. 不使用 eval、不动态执行脚本。
 * 4. POST 不自动重试；提交期间禁用按钮。
 * 5. 提交失败保留表单内容。
 */

(function () {

    // =========================================================
    // 常量与状态
    // =========================================================

    var API_PREFIX = "/business";
    var CHILD_PAGE_SIZE = 100;
    var ID_PATTERN = /^[A-Za-z0-9_-]+$/;

    // 仅用于配色，不参与校验；状态白名单以页面 select 选项为准。
    var STATUS_TONE = {
        "待支付": "tone-warn",
        "待发货": "tone-warn",
        "待处理": "tone-warn",
        "已支付": "tone-info",
        "已发货": "tone-info",
        "处理中": "tone-info",
        "支付成功": "tone-success",
        "已完成": "tone-success",
        "支付失败": "tone-danger",
        "已取消": "tone-muted",
        "已退款": "tone-muted",
        "已关闭": "tone-muted"
    };

    var state = {
        dbAvailable: true,
        orderId: null,
        shipmentId: null,
        orders: {
            page: 1,
            pageSize: 20,
            total: 0,
            requestId: 0,
            filters: {
                status: "",
                customer_id: "",
                created_from: "",
                created_to: ""
            }
        }
    };

    // =========================================================
    // DOM 引用
    // =========================================================

    var banner = document.getElementById("global-banner");
    var dbIndicator = document.getElementById("db-indicator");
    var toast = document.getElementById("toast");
    var navDetail = document.getElementById("nav-detail");
    var navEvents = document.getElementById("nav-events");

    var viewOrders = document.getElementById("view-orders");
    var viewDetail = document.getElementById("view-detail");
    var viewEvents = document.getElementById("view-events");

    var orderForm = document.getElementById("order-form");
    var orderFilter = document.getElementById("order-filter");
    var ordersTable = document.getElementById("orders-table");
    var ordersTotal = document.getElementById("orders-total");
    var ordersTotalTop = document.getElementById("orders-total-top");
    var ordersPageInfo = document.getElementById("orders-page-info");
    var ordersPrev = document.getElementById("orders-prev");
    var ordersNext = document.getElementById("orders-next");
    var ordersPageSize = document.getElementById("orders-page-size");

    var detailBody = document.getElementById("detail-body");
    var orderSummary = document.getElementById("order-summary");

    var paymentsTable = document.getElementById("payments-table");
    var paymentsMeta = document.getElementById("payments-meta");
    var shipmentsTable = document.getElementById("shipments-table");
    var shipmentsMeta = document.getElementById("shipments-meta");
    var aftersalesTable = document.getElementById("aftersales-table");
    var aftersalesMeta = document.getElementById("aftersales-meta");
    var workordersTable = document.getElementById("workorders-table");
    var workordersMeta = document.getElementById("workorders-meta");

    var paymentForm = document.getElementById("payment-form");
    var shipmentForm = document.getElementById("shipment-form");
    var aftersaleForm = document.getElementById("aftersale-form");
    var workorderForm = document.getElementById("workorder-form");

    var eventsTitle = document.getElementById("events-title");
    var eventsBack = document.getElementById("events-back");
    var shipmentSummary = document.getElementById("shipment-summary");
    var eventsTimeline = document.getElementById("events-timeline");
    var eventsMeta = document.getElementById("events-meta");
    var eventForm = document.getElementById("event-form");

    // =========================================================
    // 通用工具
    // =========================================================

    function el(tag, className, text) {
        var node = document.createElement(tag);
        if (className) {
            node.className = className;
        }
        if (text !== undefined && text !== null) {
            node.textContent = String(text);
        }
        return node;
    }

    function displayValue(value) {
        if (value === null || value === undefined || value === "") {
            return "—";
        }
        return String(value);
    }

    function formatAmount(value) {
        var number = Number(value);
        if (!isFinite(number)) {
            return displayValue(value);
        }
        return number.toFixed(2);
    }

    function boolText(value) {
        return value ? "是" : "否";
    }

    function statusBadge(value) {
        var text = displayValue(value);
        var tone = STATUS_TONE[text] || "tone-neutral";
        return el("span", "badge " + tone, text);
    }

    function nowLocalValue() {
        var now = new Date();
        function pad(number) {
            return String(number).padStart(2, "0");
        }
        return now.getFullYear()
            + "-" + pad(now.getMonth() + 1)
            + "-" + pad(now.getDate())
            + "T" + pad(now.getHours())
            + ":" + pad(now.getMinutes())
            + ":" + pad(now.getSeconds());
    }

    function readRaw(form, name) {
        var field = form.elements.namedItem(name);
        return field ? field.value : "";
    }

    function readTrimmed(form, name) {
        return readRaw(form, name).trim();
    }

    function readChecked(form, name) {
        var field = form.elements.namedItem(name);
        return !!(field && field.checked);
    }

    // =========================================================
    // API 层
    // =========================================================

    function ApiError(status, info) {
        this.name = "ApiError";
        this.message = info.message;
        this.status = status;
        this.kind = info.kind;
        this.fields = info.fields || {};
    }
    ApiError.prototype = Object.create(Error.prototype);

    function normalizeError(status, body) {
        var detail = body ? body.detail : null;

        if (Array.isArray(detail)) {
            var fields = {};
            var parts = [];

            detail.forEach(function (item) {
                var loc = Array.isArray(item.loc) ? item.loc : [];
                var field = loc.length > 1 ? String(loc[loc.length - 1]) : "";
                var message = item.msg || "参数不合法";

                if (field) {
                    fields[field] = message;
                    parts.push(field + "：" + message);
                } else {
                    parts.push(message);
                }
            });

            return {
                kind: "validation",
                message: parts.length ? parts.join("；") : "请求参数不合法",
                fields: fields
            };
        }

        if (detail && typeof detail === "object") {
            return {
                kind: detail.type || "error",
                message: detail.message || "请求失败",
                fields: {}
            };
        }

        if (typeof detail === "string") {
            return { kind: "error", message: detail, fields: {} };
        }

        return {
            kind: "http",
            message: "请求失败（HTTP " + status + "）",
            fields: {}
        };
    }

    function request(path, options) {
        return fetch(path, options).then(function (response) {
            return response.text().then(function (text) {
                var body = null;

                if (text) {
                    try {
                        body = JSON.parse(text);
                    } catch (error) {
                        body = null;
                    }
                }

                if (response.ok) {
                    setDbAvailable(true);
                    return body ? body.data : null;
                }

                if (response.status === 503) {
                    setDbAvailable(false);
                }

                throw new ApiError(response.status, normalizeError(response.status, body));
            });
        }, function () {
            throw new ApiError(0, {
                kind: "network_error",
                message: "无法连接后端服务，请确认服务已启动。",
                fields: {}
            });
        });
    }

    function getJson(path, params) {
        var search = new URLSearchParams();

        if (params) {
            Object.keys(params).forEach(function (key) {
                var value = params[key];
                if (value !== undefined && value !== null && value !== "") {
                    search.append(key, value);
                }
            });
        }

        var query = search.toString();
        return request(path + (query ? "?" + query : ""), { method: "GET" });
    }

    function postJson(path, payload) {
        return request(path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
    }

    // =========================================================
    // 全局提示与数据库可用性
    // =========================================================

    var toastTimer = null;

    function showToast(message, isError) {
        toast.textContent = message;
        toast.className = "toast " + (isError ? "toast-error" : "toast-success");
        toast.hidden = false;

        if (toastTimer) {
            clearTimeout(toastTimer);
        }

        toastTimer = setTimeout(function () {
            toast.hidden = true;
        }, 2600);
    }

    function setDbAvailable(available) {
        if (state.dbAvailable === available) {
            return;
        }

        state.dbAvailable = available;

        if (available) {
            banner.hidden = true;
            banner.textContent = "";
            dbIndicator.textContent = "服务正常";
            dbIndicator.className = "db-indicator db-ok";
        } else {
            banner.textContent = "业务数据库不可用，写入功能已禁用。请检查后端启动日志后刷新页面。";
            banner.hidden = false;
            dbIndicator.textContent = "服务不可用";
            dbIndicator.className = "db-indicator db-bad";
        }

        document.querySelectorAll("form button[type=submit]").forEach(function (button) {
            button.disabled = !available;
        });
    }

    // =========================================================
    // 表格与空状态
    // =========================================================

    function renderTable(container, headers, rows, rowBuilder, emptyText) {
        container.replaceChildren();

        if (!rows || rows.length === 0) {
            container.appendChild(el("p", "empty-state", emptyText));
            return;
        }

        var table = el("table", "data-table");
        var thead = el("thead");
        var headRow = el("tr");

        headers.forEach(function (header) {
            headRow.appendChild(el("th", null, header));
        });

        thead.appendChild(headRow);
        table.appendChild(thead);

        var tbody = el("tbody");
        rows.forEach(function (row) {
            tbody.appendChild(rowBuilder(row));
        });
        table.appendChild(tbody);
        container.appendChild(table);
    }

    function renderLoading(container, text) {
        container.replaceChildren(el("p", "empty-state", text));
    }

    function rowCell(text, className) {
        return el("td", className || null, displayValue(text));
    }

    // =========================================================
    // 视图切换与路由
    // =========================================================

    function showView(name) {
        viewOrders.hidden = name !== "orders";
        viewDetail.hidden = name !== "detail";
        viewEvents.hidden = name !== "events";
    }

    function parseHash() {
        var raw = window.location.hash.replace(/^#/, "");
        var parts = raw.split("/").filter(function (part) {
            return part !== "";
        }).map(function (part) {
            try {
                return decodeURIComponent(part);
            } catch (error) {
                return part;
            }
        });

        if (parts.length === 5
            && parts[0] === "orders"
            && parts[2] === "shipments"
            && parts[4] === "events") {
            return { name: "events", orderId: parts[1], shipmentId: parts[3] };
        }

        if (parts.length === 2 && parts[0] === "orders") {
            return { name: "detail", orderId: parts[1] };
        }

        return { name: "orders" };
    }

    function navigate(hash) {
        if (window.location.hash === hash) {
            route();
        } else {
            window.location.hash = hash;
        }
    }

    function route() {
        var parsed = parseHash();

        navDetail.hidden = parsed.name === "orders";
        navEvents.hidden = parsed.name !== "events";

        if (parsed.name === "events") {
            navDetail.textContent = parsed.orderId;
            navEvents.textContent = parsed.shipmentId;
            showView("events");
            loadEventsView(parsed.orderId, parsed.shipmentId);
            return;
        }

        if (parsed.name === "detail") {
            navDetail.textContent = parsed.orderId;
            delete navEvents.textContent;
            navEvents.hidden = true;
            showView("detail");
            loadDetail(parsed.orderId);
            return;
        }

        showView("orders");
        loadOrders();
    }

    // =========================================================
    // 视图 1：订单列表
    // =========================================================

    function renderOrderRow(order) {
        var row = el("tr");

        row.appendChild(rowCell(order.order_id, "cell-id"));
        row.appendChild(rowCell(order.customer_id));
        row.appendChild(el("td")).appendChild(statusBadge(order.status));
        row.appendChild(el("td", "cell-amount", formatAmount(order.amount)));
        row.appendChild(rowCell(order.created_at, "cell-time"));
        row.appendChild(rowCell(order.paid_at, "cell-time"));

        var actions = el("td", "cell-actions");
        var button = el("button", "btn btn-small", "查看");
        button.type = "button";
        button.addEventListener("click", function () {
            navigate("#/orders/" + encodeURIComponent(order.order_id));
        });
        actions.appendChild(button);
        row.appendChild(actions);

        return row;
    }

    function renderPager(data) {
        var totalPages = Math.max(1, Math.ceil(data.total / data.page_size));

        ordersTotal.textContent = "共 " + data.total + " 条";
        ordersTotalTop.textContent = String(data.total);
        ordersPageInfo.textContent = data.page + " / " + totalPages;
        ordersPrev.disabled = data.page <= 1;
        ordersNext.disabled = data.page >= totalPages;
    }

    function loadOrders() {
        var requestId = ++state.orders.requestId;
        renderLoading(ordersTable, "正在加载订单…");

        getJson(API_PREFIX + "/orders", {
            status: state.orders.filters.status,
            customer_id: state.orders.filters.customer_id,
            created_from: state.orders.filters.created_from,
            created_to: state.orders.filters.created_to,
            page: state.orders.page,
            page_size: state.orders.pageSize
        }).then(function (data) {
            if (requestId !== state.orders.requestId) {
                return;
            }

            var totalPages = Math.max(1, Math.ceil(data.total / data.page_size));

            if (data.total > 0 && data.page > totalPages) {
                state.orders.page = totalPages;
                loadOrders();
                return;
            }

            state.orders.total = data.total;
            renderTable(
                ordersTable,
                ["订单编号", "客户编号", "状态", "金额", "创建时间", "支付时间", "操作"],
                data.items,
                renderOrderRow,
                "还没有订单。点击右上角「新建订单」开始录入。"
            );
            renderPager(data);
        }).catch(function (error) {
            if (requestId !== state.orders.requestId) {
                return;
            }

            ordersTable.replaceChildren(
                el("p", "empty-state empty-state-error", error.message)
            );
            ordersTotal.textContent = "共 0 条";
            ordersTotalTop.textContent = "0";
            ordersPageInfo.textContent = "1 / 1";
            ordersPrev.disabled = true;
            ordersNext.disabled = true;
        });
    }

    // =========================================================
    // 视图 2：订单详情
    // =========================================================

    function summaryItem(label, valueNode) {
        var item = el("div", "summary-item");
        item.appendChild(el("span", "summary-label", label));

        if (valueNode instanceof Node) {
            item.appendChild(valueNode);
        } else {
            item.appendChild(el("span", "summary-value", displayValue(valueNode)));
        }

        return item;
    }

    function renderOrderSummary(order) {
        orderSummary.replaceChildren();
        detailBody.classList.remove("is-missing");

        var grid = el("div", "summary-grid");

        grid.appendChild(summaryItem("订单编号", order.order_id));
        grid.appendChild(summaryItem("客户编号", order.customer_id));
        grid.appendChild(summaryItem("订单状态", statusBadge(order.status)));
        grid.appendChild(summaryItem("金额", formatAmount(order.amount)));
        grid.appendChild(summaryItem("创建时间", order.created_at));
        grid.appendChild(summaryItem("支付时间", order.paid_at));

        orderSummary.appendChild(grid);
    }

    function renderMissingOrder(orderId) {
        orderSummary.replaceChildren();
        detailBody.classList.add("is-missing");

        var card = el("div", "error-card");
        card.appendChild(el("h2", "error-title", "订单编号不存在"));
        card.appendChild(el("p", "error-text", "未找到订单 " + orderId + "，请返回列表确认编号。"));

        var actions = el("div", "form-actions");
        var back = el("a", "btn btn-primary", "返回订单列表");
        back.href = "#/orders";
        actions.appendChild(back);
        card.appendChild(actions);

        orderSummary.appendChild(card);

        [paymentsTable, shipmentsTable, aftersalesTable, workordersTable].forEach(function (node) {
            node.replaceChildren();
        });
        [paymentsMeta, shipmentsMeta, aftersalesMeta, workordersMeta].forEach(function (node) {
            node.textContent = "";
        });
    }

    function renderDetailError(error) {
        orderSummary.replaceChildren();

        var card = el("div", "error-card");
        card.appendChild(el("h2", "error-title", "订单加载失败"));
        card.appendChild(el("p", "error-text", error.message));

        var actions = el("div", "form-actions");
        var retry = el("button", "btn btn-primary", "重试");
        retry.type = "button";
        retry.addEventListener("click", function () {
            loadDetail(state.orderId);
        });
        actions.appendChild(retry);
        card.appendChild(actions);

        orderSummary.appendChild(card);
    }

    function loadChildSection(config) {
        config.container.replaceChildren(el("p", "empty-state", "正在加载…"));
        config.meta.textContent = "";

        getJson(config.path, { page: 1, page_size: CHILD_PAGE_SIZE }).then(function (data) {
            renderTable(
                config.container,
                config.headers,
                data.items,
                config.rowBuilder,
                config.emptyText
            );

            var meta = "共 " + data.total + " 条";
            if (data.total > data.items.length) {
                meta += "（当前显示前 " + data.items.length + " 条）";
            }
            config.meta.textContent = meta;
        }).catch(function (error) {
            config.container.replaceChildren(
                el("p", "empty-state empty-state-error", error.message)
            );
        });
    }

    function renderDetailLoadError(container, error) {
        container.replaceChildren(
            el("p", "empty-state empty-state-error", error.message)
        );
    }

    function paymentRow(item) {
        var row = el("tr");
        row.appendChild(rowCell(item.payment_id, "cell-id"));
        row.appendChild(rowCell(item.method));
        row.appendChild(el("td", "cell-amount", formatAmount(item.amount)));
        row.appendChild(el("td")).appendChild(statusBadge(item.status));
        row.appendChild(rowCell(item.paid_at, "cell-time"));
        return row;
    }

    function shipmentRow(item) {
        var row = el("tr");
        row.appendChild(rowCell(item.shipment_id, "cell-id"));
        row.appendChild(rowCell(item.carrier));
        row.appendChild(el("td", null, displayValue(item.status)));
        row.appendChild(el("td", null, boolText(item.at_hub)));
        row.appendChild(el("td", null, boolText(item.departed_hub)));
        row.appendChild(rowCell(item.t_last, "cell-time"));

        var actions = el("td", "cell-actions");
        var button = el("button", "btn btn-small", "查看事件");
        button.type = "button";
        button.addEventListener("click", function () {
            navigate(
                "#/orders/" + encodeURIComponent(state.orderId)
                + "/shipments/" + encodeURIComponent(item.shipment_id)
                + "/events"
            );
        });
        actions.appendChild(button);
        row.appendChild(actions);

        return row;
    }

    function aftersaleRow(item) {
        var row = el("tr");
        row.appendChild(rowCell(item.aftersale_id, "cell-id"));
        row.appendChild(el("td", null, displayValue(item.type)));
        row.appendChild(el("td")).appendChild(statusBadge(item.status));
        row.appendChild(rowCell(item.created_at, "cell-time"));
        return row;
    }

    function workorderRow(item) {
        var row = el("tr");
        row.appendChild(rowCell(item.workorder_id, "cell-id"));
        row.appendChild(el("td", null, displayValue(item.type)));
        row.appendChild(el("td")).appendChild(statusBadge(item.status));
        row.appendChild(el("td", null, displayValue(item.assigned_to)));
        row.appendChild(rowCell(item.created_at, "cell-time"));
        return row;
    }

    function fillHiddenOrderId(orderId) {
        [paymentForm, shipmentForm, aftersaleForm, workorderForm].forEach(function (form) {
            var field = form.elements.namedItem("order_id");
            if (field) {
                field.value = orderId;
            }
        });
    }

    function loadDetail(orderId) {
        state.orderId = orderId;
        document.getElementById("detail-title").textContent = "订单详情 · " + orderId;
        fillHiddenOrderId(orderId);
        orderSummary.replaceChildren(el("p", "empty-state", "正在加载订单…"));

        getJson(API_PREFIX + "/orders/" + encodeURIComponent(orderId)).then(function (order) {
            renderOrderSummary(order);

            var base = API_PREFIX + "/orders/" + encodeURIComponent(orderId);

            loadChildSection({
                path: base + "/payments",
                container: paymentsTable,
                meta: paymentsMeta,
                headers: ["支付编号", "支付方式", "金额", "状态", "支付时间"],
                rowBuilder: paymentRow,
                emptyText: "该订单暂无支付记录。"
            });

            loadChildSection({
                path: base + "/shipments",
                container: shipmentsTable,
                meta: shipmentsMeta,
                headers: ["物流编号", "承运商", "状态", "在转运中心", "已离开", "最近更新", "操作"],
                rowBuilder: shipmentRow,
                emptyText: "该订单暂无物流记录。"
            });

            loadChildSection({
                path: base + "/aftersales",
                container: aftersalesTable,
                meta: aftersalesMeta,
                headers: ["售后编号", "类型", "状态", "创建时间"],
                rowBuilder: aftersaleRow,
                emptyText: "该订单暂无售后记录。"
            });

            loadChildSection({
                path: base + "/workorders",
                container: workordersTable,
                meta: workordersMeta,
                headers: ["工单编号", "类型", "状态", "负责人", "创建时间"],
                rowBuilder: workorderRow,
                emptyText: "该订单暂无客服工单。"
            });
        }).catch(function (error) {
            if (error.status === 404) {
                renderMissingOrder(orderId);
                return;
            }

            renderDetailError(error);
            [paymentsTable, shipmentsTable, aftersalesTable, workordersTable].forEach(function (node) {
                renderDetailLoadError(node, error);
            });
        });
    }

    // =========================================================
    // 视图 3：物流事件
    // =========================================================

    function renderShipmentSummary(orderId, shipmentId) {
        shipmentSummary.replaceChildren();

        var grid = el("div", "summary-grid");
        grid.appendChild(summaryItem("物流编号", shipmentId));

        var orderLink = el("a", "inline-link", orderId);
        orderLink.href = "#/orders/" + encodeURIComponent(orderId);
        grid.appendChild(summaryItem("所属订单", orderLink));

        shipmentSummary.appendChild(grid);
    }

    function renderTimeline(items) {
        eventsTimeline.replaceChildren();

        if (!items || items.length === 0) {
            var empty = el("li", "empty-state", "该物流记录暂无事件。");
            eventsTimeline.appendChild(empty);
            return;
        }

        items.forEach(function (item) {
            var node = el("li", "timeline-item");

            node.appendChild(el("div", "timeline-time", displayValue(item.event_time)));

            var body = el("div", "timeline-body");
            var head = el("div", "timeline-head");

            head.appendChild(el("span", "timeline-type", displayValue(item.event_type)));

            if (item.is_exception) {
                head.appendChild(el("span", "badge tone-danger", "异常"));
            }
            if (item.is_real_advance) {
                head.appendChild(el("span", "badge tone-info", "真实提前"));
            }

            body.appendChild(head);

            if (item.detail) {
                body.appendChild(el("p", "timeline-detail", item.detail));
            }

            body.appendChild(el("p", "timeline-id", "事件编号：" + displayValue(item.event_id)));

            node.appendChild(body);
            eventsTimeline.appendChild(node);
        });
    }

    function loadEventsView(orderId, shipmentId) {
        state.orderId = orderId;
        state.shipmentId = shipmentId;

        eventsTitle.textContent = "物流事件 · " + shipmentId;
        eventsBack.href = "#/orders/" + encodeURIComponent(orderId);

        var field = eventForm.elements.namedItem("shipment_id");
        if (field) {
            field.value = shipmentId;
        }

        renderShipmentSummary(orderId, shipmentId);
        eventsTimeline.replaceChildren(el("li", "empty-state", "正在加载事件…"));
        eventsMeta.textContent = "";

        getJson(
            API_PREFIX + "/shipments/" + encodeURIComponent(shipmentId) + "/events",
            { page: 1, page_size: CHILD_PAGE_SIZE }
        ).then(function (data) {
            renderTimeline(data.items);

            var meta = "共 " + data.total + " 条事件（按事件时间正序）";
            if (data.total > data.items.length) {
                meta += "，当前显示前 " + data.items.length + " 条";
            }
            eventsMeta.textContent = meta;
        }).catch(function (error) {
            eventsTimeline.replaceChildren(
                el("li", "empty-state empty-state-error", error.message)
            );
            eventsMeta.textContent = "";
        });
    }

    // =========================================================
    // 表单通用行为
    // =========================================================

    function setFormError(form, message) {
        var box = form.querySelector(".form-error");
        if (!box) {
            return;
        }
        box.textContent = message;
        box.hidden = false;
    }

    function clearFormError(form) {
        var box = form.querySelector(".form-error");
        if (!box) {
            return;
        }
        box.textContent = "";
        box.hidden = true;
    }

    function clearInvalidFields(form) {
        form.querySelectorAll(".is-invalid").forEach(function (node) {
            node.classList.remove("is-invalid");
        });
    }

    function markField(form, name) {
        var field = form.elements.namedItem(name);
        if (field && field.classList) {
            field.classList.add("is-invalid");
        }
    }

    function applyFieldHints(form, error) {
        Object.keys(error.fields || {}).forEach(function (name) {
            markField(form, name);
        });

        if (error.kind === "duplicate_record") {
            var idField = form.querySelector('input[name$="_id"]');
            if (idField) {
                idField.classList.add("is-invalid");
            }
        }

        if (error.kind === "invalid_reference") {
            markField(form, "order_id");
            markField(form, "shipment_id");
        }
    }

    function handleSubmit(form, buildPayload, onSuccess) {
        var button = form.querySelector("button[type=submit]");

        if (!state.dbAvailable) {
            setFormError(form, "业务数据库不可用，暂时无法提交。");
            return;
        }

        if (form.dataset.submitting === "1") {
            return;
        }

        form.dataset.submitting = "1";
        button.disabled = true;

        var originalLabel = button.textContent;
        button.textContent = "提交中…";

        clearFormError(form);
        clearInvalidFields(form);

        var built;

        try {
            built = buildPayload();
        } catch (error) {
            setFormError(form, "表单读取失败：" + error.message);
            form.dataset.submitting = "0";
            button.disabled = !state.dbAvailable;
            button.textContent = originalLabel;
            return;
        }

        if (built.problem) {
            setFormError(form, built.problem);
            form.dataset.submitting = "0";
            button.disabled = !state.dbAvailable;
            button.textContent = originalLabel;
            return;
        }

        postJson(form.dataset.action, built.payload).then(function () {
            showToast("创建成功", false);
            form.reset();
            applyDefaultTimes(form);
            form.hidden = true;
            return onSuccess();
        }).catch(function (error) {
            // 失败时保留用户已填内容，仅提示错误。
            setFormError(form, error.message);
            applyFieldHints(form, error);
            showToast(error.message, true);
        }).then(function () {
            form.dataset.submitting = "0";
            button.disabled = !state.dbAvailable;
            button.textContent = originalLabel;
        });
    }

    // =========================================================
    // 六个表单的载荷构造
    // =========================================================

    function buildOrderPayload() {
        var orderId = readTrimmed(orderForm, "order_id");
        var customerId = readTrimmed(orderForm, "customer_id");
        var createdAt = readRaw(orderForm, "created_at");
        var amountRaw = readRaw(orderForm, "amount");
        var paidAt = readRaw(orderForm, "paid_at");

        if (!orderId) {
            return { problem: "请填写订单编号。" };
        }
        if (!ID_PATTERN.test(orderId)) {
            return { problem: "订单编号只能包含字母、数字、下划线和中划线。" };
        }
        if (!customerId) {
            return { problem: "请填写客户编号。" };
        }
        if (!createdAt) {
            return { problem: "请填写创建时间。" };
        }
        if (amountRaw === "") {
            return { problem: "请填写金额。" };
        }

        var amount = Number(amountRaw);
        if (!isFinite(amount) || amount < 0) {
            return { problem: "金额必须是不小于 0 的数字。" };
        }

        return {
            payload: {
                order_id: orderId,
                customer_id: customerId,
                status: readRaw(orderForm, "status"),
                created_at: createdAt,
                amount: amount,
                paid_at: paidAt === "" ? null : paidAt
            }
        };
    }

    function buildPaymentPayload() {
        var paymentId = readTrimmed(paymentForm, "payment_id");
        var orderId = readTrimmed(paymentForm, "order_id") || state.orderId;
        var method = readTrimmed(paymentForm, "method");
        var amountRaw = readRaw(paymentForm, "amount");
        var paidAt = readRaw(paymentForm, "paid_at");

        if (!paymentId) {
            return { problem: "请填写支付编号。" };
        }
        if (!method) {
            return { problem: "请填写支付方式。" };
        }
        if (amountRaw === "") {
            return { problem: "请填写金额。" };
        }

        var amount = Number(amountRaw);
        if (!isFinite(amount) || amount < 0) {
            return { problem: "金额必须是不小于 0 的数字。" };
        }

        return {
            payload: {
                payment_id: paymentId,
                order_id: orderId,
                paid_at: paidAt === "" ? null : paidAt,
                method: method,
                amount: amount,
                status: readRaw(paymentForm, "status")
            }
        };
    }

    function buildShipmentPayload() {
        var shipmentId = readTrimmed(shipmentForm, "shipment_id");
        var orderId = readTrimmed(shipmentForm, "order_id") || state.orderId;
        var carrier = readTrimmed(shipmentForm, "carrier");
        var status = readTrimmed(shipmentForm, "status");
        var tLast = readRaw(shipmentForm, "t_last");

        if (!shipmentId) {
            return { problem: "请填写物流编号。" };
        }
        if (!ID_PATTERN.test(shipmentId)) {
            return { problem: "物流编号只能包含字母、数字、下划线和中划线。" };
        }
        if (!carrier) {
            return { problem: "请填写承运商。" };
        }
        if (!status) {
            return { problem: "请填写物流状态。" };
        }

        return {
            payload: {
                shipment_id: shipmentId,
                order_id: orderId,
                carrier: carrier,
                status: status,
                at_hub: readChecked(shipmentForm, "at_hub"),
                departed_hub: readChecked(shipmentForm, "departed_hub"),
                t_last: tLast === "" ? null : tLast
            }
        };
    }

    function buildAftersalePayload() {
        var aftersaleId = readTrimmed(aftersaleForm, "aftersale_id");
        var orderId = readTrimmed(aftersaleForm, "order_id") || state.orderId;
        var type = readTrimmed(aftersaleForm, "type");
        var createdAt = readRaw(aftersaleForm, "created_at");

        if (!aftersaleId) {
            return { problem: "请填写售后编号。" };
        }
        if (!type) {
            return { problem: "请填写售后类型。" };
        }
        if (!createdAt) {
            return { problem: "请填写创建时间。" };
        }

        return {
            payload: {
                aftersale_id: aftersaleId,
                order_id: orderId,
                type: type,
                status: readRaw(aftersaleForm, "status"),
                created_at: createdAt
            }
        };
    }

    function buildWorkorderPayload() {
        var workorderId = readTrimmed(workorderForm, "workorder_id");
        var orderId = readTrimmed(workorderForm, "order_id") || state.orderId;
        var type = readTrimmed(workorderForm, "type");
        var createdAt = readRaw(workorderForm, "created_at");
        var assignedTo = readTrimmed(workorderForm, "assigned_to");

        if (!workorderId) {
            return { problem: "请填写工单编号。" };
        }
        if (!type) {
            return { problem: "请填写工单类型。" };
        }
        if (!createdAt) {
            return { problem: "请填写创建时间。" };
        }

        return {
            payload: {
                workorder_id: workorderId,
                order_id: orderId,
                type: type,
                status: readRaw(workorderForm, "status"),
                created_at: createdAt,
                assigned_to: assignedTo === "" ? null : assignedTo
            }
        };
    }

    function buildEventPayload() {
        var eventId = readTrimmed(eventForm, "event_id");
        var shipmentId = readTrimmed(eventForm, "shipment_id") || state.shipmentId;
        var eventType = readTrimmed(eventForm, "event_type");
        var eventTime = readRaw(eventForm, "event_time");
        var detail = readRaw(eventForm, "detail").trim();

        if (!eventId) {
            return { problem: "请填写事件编号。" };
        }
        if (!eventType) {
            return { problem: "请填写事件类型。" };
        }
        if (!eventTime) {
            return { problem: "请填写事件时间。" };
        }

        return {
            payload: {
                event_id: eventId,
                shipment_id: shipmentId,
                event_type: eventType,
                event_time: eventTime,
                detail: detail,
                is_real_advance: readChecked(eventForm, "is_real_advance"),
                is_exception: readChecked(eventForm, "is_exception")
            }
        };
    }

    // =========================================================
    // 默认时间与事件绑定
    // =========================================================

    var DEFAULT_TIME_FIELDS = {
        "order-form": ["created_at"],
        "payment-form": ["paid_at"],
        "aftersale-form": ["created_at"],
        "workorder-form": ["created_at"],
        "event-form": ["event_time"]
    };

    function applyDefaultTimes(form) {
        var names = DEFAULT_TIME_FIELDS[form.id];
        if (!names) {
            return;
        }

        var now = nowLocalValue();

        names.forEach(function (name) {
            var field = form.elements.namedItem(name);
            if (field && !field.value) {
                field.value = now;
            }
        });
    }

    function bindToggle(buttonId, formId) {
        var button = document.getElementById(buttonId);
        var form = document.getElementById(formId);

        button.addEventListener("click", function () {
            if (!state.dbAvailable) {
                showToast("业务数据库不可用，暂时无法录入。", true);
                return;
            }

            form.hidden = !form.hidden;

            if (!form.hidden) {
                applyDefaultTimes(form);
                var firstField = form.querySelector("input:not([type=hidden]), select, textarea");
                if (firstField) {
                    firstField.focus();
                }
            }
        });
    }

    function bindEvents() {
        orderFilter.addEventListener("submit", function (event) {
            event.preventDefault();
            state.orders.filters.status = readRaw(orderFilter, "status");
            state.orders.filters.customer_id = readTrimmed(orderFilter, "customer_id");
            state.orders.filters.created_from = readRaw(orderFilter, "created_from");
            state.orders.filters.created_to = readRaw(orderFilter, "created_to");
            state.orders.page = 1;
            loadOrders();
        });

        document.getElementById("order-filter-reset").addEventListener("click", function () {
            orderFilter.reset();
            state.orders.filters.status = "";
            state.orders.filters.customer_id = "";
            state.orders.filters.created_from = "";
            state.orders.filters.created_to = "";
            state.orders.page = 1;
            loadOrders();
        });

        ordersPageSize.addEventListener("change", function () {
            state.orders.pageSize = Number(ordersPageSize.value) || 20;
            state.orders.page = 1;
            loadOrders();
        });

        ordersPrev.addEventListener("click", function () {
            if (state.orders.page > 1) {
                state.orders.page -= 1;
                loadOrders();
            }
        });

        ordersNext.addEventListener("click", function () {
            state.orders.page += 1;
            loadOrders();
        });

        bindToggle("toggle-order-form", "order-form");
        bindToggle("toggle-payment-form", "payment-form");
        bindToggle("toggle-shipment-form", "shipment-form");
        bindToggle("toggle-aftersale-form", "aftersale-form");
        bindToggle("toggle-workorder-form", "workorder-form");
        bindToggle("toggle-event-form", "event-form");

        document.querySelectorAll("[data-close-form]").forEach(function (button) {
            button.addEventListener("click", function () {
                var form = document.getElementById(button.dataset.closeForm);
                form.hidden = true;
                clearFormError(form);
                clearInvalidFields(form);
            });
        });

        document.querySelectorAll("form").forEach(function (form) {
            form.addEventListener("input", function (event) {
                if (event.target.classList) {
                    event.target.classList.remove("is-invalid");
                }
            });
        });

        orderForm.addEventListener("submit", function (event) {
            event.preventDefault();
            handleSubmit(orderForm, buildOrderPayload, function () {
                state.orders.page = 1;
                return loadOrders();
            });
        });

        paymentForm.addEventListener("submit", function (event) {
            event.preventDefault();
            handleSubmit(paymentForm, buildPaymentPayload, function () {
                return refreshDetailSections();
            });
        });

        shipmentForm.addEventListener("submit", function (event) {
            event.preventDefault();
            handleSubmit(shipmentForm, buildShipmentPayload, function () {
                return refreshDetailSections();
            });
        });

        aftersaleForm.addEventListener("submit", function (event) {
            event.preventDefault();
            handleSubmit(aftersaleForm, buildAftersalePayload, function () {
                return refreshDetailSections();
            });
        });

        workorderForm.addEventListener("submit", function (event) {
            event.preventDefault();
            handleSubmit(workorderForm, buildWorkorderPayload, function () {
                return refreshDetailSections();
            });
        });

        eventForm.addEventListener("submit", function (event) {
            event.preventDefault();
            handleSubmit(eventForm, buildEventPayload, function () {
                return loadEventsView(state.orderId, state.shipmentId);
            });
        });
    }

    function refreshDetailSections() {
        var orderId = state.orderId;
        var base = API_PREFIX + "/orders/" + encodeURIComponent(orderId);

        loadChildSection({
            path: base + "/payments",
            container: paymentsTable,
            meta: paymentsMeta,
            headers: ["支付编号", "支付方式", "金额", "状态", "支付时间"],
            rowBuilder: paymentRow,
            emptyText: "该订单暂无支付记录。"
        });
        loadChildSection({
            path: base + "/shipments",
            container: shipmentsTable,
            meta: shipmentsMeta,
            headers: ["物流编号", "承运商", "状态", "在转运中心", "已离开", "最近更新", "操作"],
            rowBuilder: shipmentRow,
            emptyText: "该订单暂无物流记录。"
        });
        loadChildSection({
            path: base + "/aftersales",
            container: aftersalesTable,
            meta: aftersalesMeta,
            headers: ["售后编号", "类型", "状态", "创建时间"],
            rowBuilder: aftersaleRow,
            emptyText: "该订单暂无售后记录。"
        });
        loadChildSection({
            path: base + "/workorders",
            container: workordersTable,
            meta: workordersMeta,
            headers: ["工单编号", "类型", "状态", "负责人", "创建时间"],
            rowBuilder: workorderRow,
            emptyText: "该订单暂无客服工单。"
        });

        return Promise.resolve();
    }

    // =========================================================
    // 启动
    // =========================================================

    function init() {
        bindEvents();

        document.querySelectorAll("form").forEach(function (form) {
            applyDefaultTimes(form);
        });

        applyDefaultTimes(orderForm);

        ordersPageSize.value = String(state.orders.pageSize);

        window.addEventListener("hashchange", route);

        if (!window.location.hash) {
            window.location.hash = "#/orders";
        } else {
            route();
        }
    }

    init();

})();
