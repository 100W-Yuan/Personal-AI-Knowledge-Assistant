const messages = document.getElementById("messages");
const input = document.getElementById("input");
const fileInput = document.getElementById("image-file");
const sendButton = document.getElementById("send-button");
const imageButton = document.getElementById("image-button");
const previewArea = document.getElementById("preview-area");
const previewImage = document.getElementById("preview-image");
const previewName = document.getElementById("preview-name");
const inputStatus = document.getElementById("input-status");
const newConversationButton = document.getElementById(
    "new-conversation-button"
);

let selectedFile = null;
let previewUrl = null;
let isSending = false;

// 聊天记录中的图片使用独立的 Object URL：
// 与输入框预览 URL 分离，避免预览被释放后历史图片裂图。
const messageImageUrls = [];

// ========================================================
// 会话标识（阶段 1）
//
// conversation_id 只标识本地会话，不承担身份认证。
// 首次打开生成 UUID 并写入 localStorage，刷新后沿用同一个 ID，
// 新建对话时替换为新 UUID，旧会话消息仍保留在服务端。
// ========================================================

const CONVERSATION_STORAGE_KEY = "my_ai_current_conversation_id";

const CONVERSATION_ID_PATTERN = new RegExp(
    "^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-" +
    "[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    "i"
);

let currentConversationId = null;

// 已渲染过的服务端 message_id，用于恢复历史时去重。
const renderedMessageIds = new Set();

function generateConversationId() {
    if (
        typeof window.crypto === "undefined" ||
        typeof window.crypto.randomUUID !== "function"
    ) {
        throw new Error("浏览器不支持 crypto.randomUUID，无法生成会话 ID");
    }

    return window.crypto.randomUUID();
}

function readStoredConversationId() {
    try {
        const stored = window.localStorage.getItem(CONVERSATION_STORAGE_KEY);

        if (stored && CONVERSATION_ID_PATTERN.test(stored)) {
            return stored;
        }
    } catch (error) {
        // localStorage 被禁用时退化为内存态会话，不影响本次会话使用。
        return null;
    }

    return null;
}

function storeConversationId(conversationId) {
    try {
        window.localStorage.setItem(
            CONVERSATION_STORAGE_KEY,
            conversationId
        );
    } catch (error) {
        // 存储不可用不阻塞聊天。
        return;
    }
}

function setCurrentConversationId(conversationId) {
    currentConversationId = conversationId;
    storeConversationId(conversationId);
}

function ensureConversationId() {
    if (currentConversationId) {
        return currentConversationId;
    }

    const stored = readStoredConversationId();

    if (stored) {
        currentConversationId = stored;
        return stored;
    }

    const created = generateConversationId();

    currentConversationId = created;
    storeConversationId(created);

    return created;
}

function handleImageSelect(event) {
    const file = event.target.files && event.target.files[0];

    if (!file) {
        return;
    }

    if (!["image/png", "image/jpeg"].includes(file.type)) {
        setStatus("只支持 PNG 和 JPEG 图片。", true);
        fileInput.value = "";
        return;
    }

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    selectedFile = file;
    previewUrl = URL.createObjectURL(file);

    previewImage.src = previewUrl;
    previewName.textContent = file.name;
    previewArea.classList.add("visible");

    setStatus("", false);
}

function removeImage() {
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    selectedFile = null;
    fileInput.value = "";
    previewImage.removeAttribute("src");
    previewName.textContent = "";
    previewArea.classList.remove("visible");
}

function setStatus(text, isError) {
    inputStatus.textContent = text;
    inputStatus.className = isError
        ? "input-status error"
        : "input-status";
}

function setSendingState(sending) {
    isSending = sending;
    sendButton.disabled = sending;
    imageButton.disabled = sending;
    sendButton.textContent = sending ? "发送中..." : "发送";
}

function appendUserMessage(message, file) {
    const wrapper = document.createElement("div");
    wrapper.className = "message user";

    const content = document.createElement("div");
    content.className = "user-content";

    if (file) {
        // 为聊天记录单独创建 URL，不复用输入框预览的 previewUrl。
        const messageImageUrl = URL.createObjectURL(file);
        messageImageUrls.push(messageImageUrl);

        const image = document.createElement("img");
        image.className = "message-image";
        image.src = messageImageUrl;
        image.alt = "用户发送的图片";
        content.appendChild(image);
    }

    if (message) {
        const text = document.createElement("div");
        text.textContent = message;
        content.appendChild(text);
    }

    if (file && !message) {
        const text = document.createElement("div");
        text.textContent = "请描述这张图片的内容";
        content.appendChild(text);
    }

    wrapper.appendChild(content);
    messages.appendChild(wrapper);
    scrollToBottom();
}

function addMessage(text) {
    const wrapper = document.createElement("div");
    wrapper.className = "message";

    const content = document.createElement("div");
    content.className = "assistant-content";
    content.textContent = text;

    wrapper.appendChild(content);
    messages.appendChild(wrapper);
    scrollToBottom();
}

function scrollToBottom() {
    messages.scrollTop = messages.scrollHeight;
}

// ========================================================
// 历史消息恢复
// ========================================================

function clearMessagesView() {
    while (messages.firstChild) {
        messages.removeChild(messages.firstChild);
    }

    renderedMessageIds.clear();
}

function renderHistoryMessage(message) {
    if (!message || typeof message !== "object") {
        return;
    }

    const role = message.role;
    const content = typeof message.content === "string"
        ? message.content
        : "";

    if (!content) {
        return;
    }

    // 服务端带 message_id 时按 id 去重，避免恢复与实时渲染重复添加。
    if (typeof message.id === "number") {
        if (renderedMessageIds.has(message.id)) {
            return;
        }
        renderedMessageIds.add(message.id);
    }

    if (role === "user") {
        const wrapper = document.createElement("div");
        wrapper.className = "message user";

        const body = document.createElement("div");
        body.className = "user-content";

        const text = document.createElement("div");
        text.textContent = content;

        body.appendChild(text);
        wrapper.appendChild(body);
        messages.appendChild(wrapper);
    } else {
        addMessage("AI：" + content);
        return;
    }

    scrollToBottom();
}

async function restoreConversation(conversationId) {
    try {
        const response = await fetch(
            "/conversations/" +
            encodeURIComponent(conversationId) +
            "/messages"
        );

        // 合法但不存在的会话返回空数组；这里也容忍空会话语义。
        if (!response.ok) {
            if (response.status >= 400 && response.status < 500) {
                clearMessagesView();
                addMessage("AI：你好，我是你的 AI Agent。");
                return;
            }

            throw new Error("HTTP " + response.status);
        }

        const data = await response.json();
        const list = Array.isArray(data.messages) ? data.messages : [];

        clearMessagesView();

        if (list.length === 0) {
            addMessage("AI：你好，我是你的 AI Agent。");
            return;
        }

        list.forEach(renderHistoryMessage);
        setStatus("", false);

    } catch (error) {
        setStatus("历史消息加载失败：" + error.message, true);
    }
}

function startNewConversation() {
    if (isSending) {
        return;
    }

    let conversationId;

    try {
        conversationId = generateConversationId();
    } catch (error) {
        setStatus("无法新建对话：" + error.message, true);
        return;
    }

    setCurrentConversationId(conversationId);

    removeImage();
    input.value = "";
    clearMessagesView();
    addMessage("AI：你好，我是你的 AI Agent。");
    setStatus("已开始新对话。", false);
    input.focus();
}

async function sendMessage() {
    if (isSending) {
        return;
    }

    const message = input.value.trim();
    const file = selectedFile;

    if (!message && !file) {
        return;
    }

    setSendingState(true);
    setStatus(
        file ? "正在上传并分析图片..." : "正在发送消息...",
        false
    );

    let conversationId = null;

    try {
        conversationId = ensureConversationId();
    } catch (error) {
        // 无法生成会话 ID 时不发送请求，避免消息落到默认 legacy 会话。
        setSendingState(false);
        setStatus("无法初始化会话：" + error.message, true);
        return;
    }

    try {
        let response;

        if (file) {
            // 图片 + 文字：调用现有图片理解接口
            const formData = new FormData();
            formData.append("file", file);
            formData.append("question", message);

            response = await fetch("/analyze-image", {
                method: "POST",
                body: formData
            });

        } else {
            // 纯文字：调用现有聊天接口，并显式绑定当前会话
            response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    message: message,
                    conversation_id: conversationId
                })
            });
        }

        let data = null;
        let parsedAsJson = true;

        try {
            data = await response.json();
        } catch (error) {
            parsedAsJson = false;
        }

        // 非 JSON 响应（例如纯文本 500）：保留 HTTP 状态码，避免只剩
        // “无法解析”这类无信息量提示；不展示服务端堆栈或敏感内容。
        if (!parsedAsJson) {
            const prefix = file ? "图片请求失败" : "请求失败";
            throw new Error(
                `${prefix}（HTTP ${response.status}）：响应不是有效的 JSON`
            );
        }

        if (!response.ok) {
            let detail = data.detail || "请求失败";

            if (typeof detail === "object") {
                detail = detail.message || detail.type || "请求失败";
            }

            throw new Error(detail);
        }

        if (data.success === false) {
            let errorMessage = data.answer || "处理失败";

            if (data.error && typeof data.error === "object") {
                errorMessage =
                    data.error.message ||
                    data.error.type ||
                    errorMessage;
            }

            throw new Error(errorMessage);
        }

        // 服务端回传的会话 ID 与本地不一致时，以服务端为准并落盘，
        // 保证后续消息与历史恢复都落在同一个会话上。
        const returnedConversationId = (
            data && typeof data.conversation_id === "string"
        ) ? data.conversation_id : null;

        let conversationSwitched = false;

        if (
            returnedConversationId &&
            returnedConversationId !== conversationId
        ) {
            setCurrentConversationId(returnedConversationId);
            conversationSwitched = true;
        }

        // 请求成功后才将消息放入聊天记录（图片 URL 由内部独立创建）
        appendUserMessage(message, file);

        const answer = data.answer || "暂时没有生成回复。";
        addMessage("AI：" + answer);

        // 成功后才清空输入内容
        input.value = "";

        if (file) {
            removeImage();
        }

        setStatus(
            conversationSwitched
                ? "会话已切换到服务端返回的 ID。"
                : "",
            false
        );

    } catch (error) {
        // 失败时不清空文字和图片，方便用户重试
        setStatus("发送失败：" + error.message + "，可以修改后重试。", true);

    } finally {
        setSendingState(false);
    }
}

// Enter 发送，Shift + Enter 换行
input.addEventListener("keydown", function(event) {
    // 中文输入法正在组合候选词时，不触发发送
    if (event.isComposing || event.keyCode === 229) {
        return;
    }

    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

// 新建对话：生成本地新 ID 并清空界面，旧会话消息仍保留在服务端
if (newConversationButton) {
    newConversationButton.addEventListener("click", startNewConversation);
}

// 初始化：沿用 localStorage 中的会话 ID，并恢复该会话的历史消息。
// 恢复只读接口，不触发任何模型调用。
(async function initConversation() {
    let conversationId;

    try {
        conversationId = ensureConversationId();
    } catch (error) {
        setStatus("无法初始化会话：" + error.message, true);
        return;
    }

    await restoreConversation(conversationId);
}());

// 页面离开时分别释放输入框预览 URL 与聊天记录中的图片 URL
window.addEventListener("beforeunload", function() {
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    messageImageUrls.forEach(function(url) {
        URL.revokeObjectURL(url);
    });
    messageImageUrls.length = 0;
});
