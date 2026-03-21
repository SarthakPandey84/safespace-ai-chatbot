"use strict";

const API_BASE_URL = "http://localhost:8000";

const API_ENDPOINTS = {
    newSession:   `${API_BASE_URL}/session/new`,
    chat:         `${API_BASE_URL}/chat`,
    clearSession: (id) => `${API_BASE_URL}/session/${id}`,
};

const SESSION_STORAGE_KEY = "safespace_session_id";

const EMOTION_EMOJI_MAP = {
    anxious:     "😰",
    sad:         "😔",
    angry:       "😤",
    hopeful:     "🌱",
    lonely:      "🌙",
    overwhelmed: "🌊",
    confused:    "🌀",
    numb:        "❄️",
    grateful:    "🌸",
    fearful:     "🫧",
    ashamed:     "🍂",
    frustrated:  "🌩️",
    neutral:     "✦",
};

const AppState = {
    sessionId:    null,
    isLoading:    false,
    messageCount: 0,
    lastEmotion:  "neutral",
};

const DOM = {
    messageThread:    document.getElementById("message-thread"),
    messageInput:     document.getElementById("message-input"),
    sendBtn:          document.getElementById("send-btn"),
    typingIndicator:  document.getElementById("typing-indicator"),
    sessionIdDisplay: document.getElementById("session-id-display"),
    charCount:        document.getElementById("char-count"),
    charCounter:      document.querySelector(".char-counter"),
    emotionBadge:     document.getElementById("emotion-badge"),
    emotionIcon:      document.getElementById("emotion-icon"),
    emotionText:      document.getElementById("emotion-text"),
    aiStatus:         document.getElementById("ai-status-text"),
    statusDot:        document.getElementById("status-dot"),
    btnNewChat:       document.getElementById("btn-new-chat"),
    toast:            document.getElementById("toast"),
};


async function initializeSession() {
    const existingSession = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (existingSession) {
        AppState.sessionId = existingSession;
        console.info(`[SafeSpace] Resumed session: ${existingSession.substring(0, 8)}...`);
        return existingSession;
    }

    try {
        const response = await fetch(API_ENDPOINTS.newSession, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
        });

        if (!response.ok) {
            throw new Error(`Session init failed: HTTP ${response.status}`);
        }

        const data = await response.json();
        AppState.sessionId = data.session_id;
        sessionStorage.setItem(SESSION_STORAGE_KEY, data.session_id);
        console.info(`[SafeSpace] New session created: ${data.session_id.substring(0, 8)}...`);
        return data.session_id;

    } catch (error) {
        console.error("[SafeSpace] Failed to initialize session:", error);
        const fallbackId = crypto.randomUUID();
        AppState.sessionId = fallbackId;
        sessionStorage.setItem(SESSION_STORAGE_KEY, fallbackId);
        showToast("⚠️ Running in offline mode. Some features may be limited.");
        return fallbackId;
    }
}

async function sendMessage(message) {
    const payload = {
        session_id: AppState.sessionId,
        message:    message,
    };

    const response = await fetch(API_ENDPOINTS.chat, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: HTTP ${response.status}`);
    }

    return response.json();
}

async function startNewConversation() {
    if (AppState.sessionId) {
        fetch(API_ENDPOINTS.clearSession(AppState.sessionId), {
            method: "DELETE",
        }).catch(err => console.warn("[SafeSpace] Failed to clear session on backend:", err));
    }

    sessionStorage.removeItem(SESSION_STORAGE_KEY);
    AppState.sessionId    = null;
    AppState.messageCount = 0;
    AppState.lastEmotion  = "neutral";

    DOM.messageThread.innerHTML = "";
    resetEmotionBadge();

    await initializeSession();
    updateSessionDisplay();
    renderWelcomeMessage();

    showToast("✦ New conversation started. Your privacy is protected.");
}


function renderWelcomeMessage() {
    const welcomeHTML = `
        <div class="message-row ai-row">
            <div class="message-avatar" aria-hidden="true">🌿</div>
            <div>
                <div class="message-bubble welcome" role="article">
                    Welcome. This is your safe space — a quiet corner to think,
                    feel, and share without judgment. Whatever brought you here
                    today, I'm here to listen. You can share as little or as much
                    as you'd like.
                    <br/><br/>
                    <em>Your messages are anonymized before being processed. No personal
                    details are ever stored.</em>
                </div>
                <div class="message-meta">${formatTimestamp(new Date())}</div>
            </div>
        </div>
    `;
    DOM.messageThread.insertAdjacentHTML("beforeend", welcomeHTML);
    scrollToBottom();
}

function renderUserMessage(text) {
    const safeText   = escapeHTML(text);
    const messageRow = document.createElement("div");
    messageRow.className = "message-row user-row";
    messageRow.setAttribute("role", "article");
    messageRow.setAttribute("aria-label", "Your message");

    messageRow.innerHTML = `
        <div class="message-avatar" aria-hidden="true">💬</div>
        <div>
            <div class="message-bubble user-bubble">${safeText}</div>
            <div class="message-meta">${formatTimestamp(new Date())}</div>
        </div>
    `;

    DOM.messageThread.appendChild(messageRow);
    scrollToBottom();
}

function renderAIMessage(text, emotion, piiDetected, latencyMs) {
    const safeText   = escapeHTML(text);
    const messageRow = document.createElement("div");
    messageRow.className = "message-row ai-row";
    messageRow.setAttribute("role", "article");
    messageRow.setAttribute("aria-label", "SafeSpace AI response");

    const piiBadge = piiDetected
        ? `<span class="pii-badge" aria-label="Personal information was removed from your message" title="Personal details were anonymized before processing">
               🔒 Personal details removed
           </span>`
        : "";

    const metaText = [
        formatTimestamp(new Date()),
        latencyMs ? `${latencyMs}ms` : null,
    ].filter(Boolean).join(" · ");

    messageRow.innerHTML = `
        <div class="message-avatar" aria-hidden="true">🌿</div>
        <div>
            <div class="message-bubble ai-bubble">
                ${safeText}
                ${piiBadge}
            </div>
            <div class="message-meta">${metaText}</div>
        </div>
    `;

    DOM.messageThread.appendChild(messageRow);
    scrollToBottom();
}

function renderErrorMessage(errorMessage) {
    const messageRow = document.createElement("div");
    messageRow.className = "message-row ai-row";
    messageRow.setAttribute("role", "alert");

    messageRow.innerHTML = `
        <div class="message-avatar" aria-hidden="true">🌿</div>
        <div>
            <div class="message-bubble ai-bubble" style="border-left: 3px solid var(--color-accent);">
                I'm having a little difficulty right now — please give me a moment
                and try again. I'm still here. 💚
                <br/><small style="color: var(--color-text-muted); font-size: 0.75rem;">
                    Technical detail: ${escapeHTML(errorMessage)}
                </small>
            </div>
            <div class="message-meta">${formatTimestamp(new Date())}</div>
        </div>
    `;

    DOM.messageThread.appendChild(messageRow);
    scrollToBottom();
}

function updateEmotionBadge(emotion) {
    const safeEmotion = emotion || "neutral";
    const emoji       = EMOTION_EMOJI_MAP[safeEmotion] || "✦";

    DOM.emotionBadge.className  = `emotion-badge emotion-${safeEmotion}`;
    DOM.emotionIcon.textContent = emoji;
    DOM.emotionText.textContent = safeEmotion;
    AppState.lastEmotion        = safeEmotion;
}

function resetEmotionBadge() {
    DOM.emotionBadge.className  = "emotion-badge";
    DOM.emotionIcon.textContent = "✦";
    DOM.emotionText.textContent = "Listening...";
}

function setLoadingState(isLoading) {
    AppState.isLoading         = isLoading;
    DOM.typingIndicator.hidden = !isLoading;
    DOM.sendBtn.disabled       = isLoading || !DOM.messageInput.value.trim();
    DOM.messageInput.disabled  = isLoading;

    if (isLoading) {
        DOM.statusDot.classList.add("thinking");
        DOM.aiStatus.textContent = "Listening and reflecting...";
    } else {
        DOM.statusDot.classList.remove("thinking");
        DOM.aiStatus.textContent = "Ready to listen";
    }

    if (isLoading) scrollToBottom();
}

function updateSessionDisplay() {
    if (!AppState.sessionId) return;
    const id = AppState.sessionId;
    DOM.sessionIdDisplay.textContent = `${id.substring(0, 8)}...${id.slice(-4)}`;
    DOM.sessionIdDisplay.title       = `Full session ID: ${id}`;
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        DOM.messageThread.scrollTop = DOM.messageThread.scrollHeight;
    });
}

function showToast(message, duration = 3000) {
    DOM.toast.textContent = message;
    DOM.toast.hidden      = false;
    DOM.toast.offsetHeight;
    DOM.toast.classList.add("visible");

    setTimeout(() => {
        DOM.toast.classList.remove("visible");
        setTimeout(() => { DOM.toast.hidden = true; }, 300);
    }, duration);
}

function autoResizeTextarea() {
    const input        = DOM.messageInput;
    input.style.height = "auto";
    const newHeight    = Math.min(input.scrollHeight, 120);
    input.style.height = `${newHeight}px`;
}

function updateCharCounter() {
    const count = DOM.messageInput.value.length;
    const max   = 2000;
    DOM.charCount.textContent = count;

    DOM.charCounter.classList.remove("warning", "danger");
    if (count > max * 0.9)       DOM.charCounter.classList.add("danger");
    else if (count > max * 0.75) DOM.charCounter.classList.add("warning");
}


function escapeHTML(str) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}

function formatTimestamp(date) {
    return date.toLocaleTimeString("en-IN", {
        hour:   "numeric",
        minute: "2-digit",
        hour12: true,
    });
}


async function handleSendMessage() {
    const rawMessage = DOM.messageInput.value.trim();

    if (!rawMessage || AppState.isLoading) return;

    if (!AppState.sessionId) {
        showToast("⚠️ Session not ready. Please wait a moment.");
        return;
    }

    DOM.messageInput.value = "";
    autoResizeTextarea();
    updateCharCounter();
    DOM.sendBtn.disabled = true;

    renderUserMessage(rawMessage);
    AppState.messageCount++;

    setLoadingState(true);

    try {
        const data = await sendMessage(rawMessage);

        renderAIMessage(
            data.response,
            data.detected_emotion,
            data.pii_was_detected,
            data.latency_ms
        );

        updateEmotionBadge(data.detected_emotion);

    } catch (error) {
        console.error("[SafeSpace] Message send failed:", error);
        renderErrorMessage(error.message);
    } finally {
        setLoadingState(false);
        DOM.messageInput.focus();
    }
}


DOM.sendBtn.addEventListener("click", handleSendMessage);

DOM.messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        handleSendMessage();
    }
});

DOM.messageInput.addEventListener("input", () => {
    autoResizeTextarea();
    updateCharCounter();
    DOM.sendBtn.disabled = !DOM.messageInput.value.trim() || AppState.isLoading;
});

DOM.btnNewChat.addEventListener("click", async () => {
    const confirmed = window.confirm(
        "Start a new conversation? The current conversation will end."
    );
    if (confirmed) {
        await startNewConversation();
    }
});


document.addEventListener("DOMContentLoaded", async () => {
    console.info("[SafeSpace] Initializing application...");

    try {
        await initializeSession();
        updateSessionDisplay();
        renderWelcomeMessage();
        DOM.messageInput.focus();
        console.info("[SafeSpace] Application ready.");
    } catch (error) {
        console.error("[SafeSpace] Initialization failed:", error);
        showToast("⚠️ Could not connect to SafeSpace. Please refresh the page.", 6000);
    }
});

const style = document.createElement('style');
style.textContent = `
    .pii-badge {
        display: inline-block;
        margin-top: 8px;
        font-size: 0.7rem;
        color: var(--color-text-muted);
        background: rgba(58, 107, 139, 0.08);
        border: 1px solid rgba(58, 107, 139, 0.18);
        border-radius: 100px;
        padding: 2px 8px;
    }
`;
document.head.appendChild(style);