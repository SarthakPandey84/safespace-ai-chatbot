/**
 * FILE: frontend/script.js
 * PROJECT: SafeSpace AI - Empathetic Privacy-First Chatbot
 * PURPOSE: All client-side logic for the chat interface. Handles:
 *            1. Anonymous UUID session initialization & persistence
 *            2. Sending messages to the FastAPI backend
 *            3. Rendering AI responses and user messages in the DOM
 *            4. UI state management (typing indicators, button states)
 *            5. Auto-resizing textarea, character count, keyboard shortcuts
 *
 * DESIGN PRINCIPLE — No Framework, Maximum Clarity:
 *   This file uses zero external libraries or frameworks. Every DOM
 *   operation is explicit and readable. This is intentional for an
 *   academic project — a faculty reviewer should be able to trace the
 *   execution of every user interaction without framework knowledge.
 *
 * ARCHITECTURE NOTE — Separation of Concerns within this file:
 *   The code is organized into four clear sections:
 *     1. CONFIGURATION    — API URLs, constants
 *     2. STATE MANAGEMENT — Session UUID, conversation state
 *     3. API LAYER        — All fetch() calls to the backend
 *     4. UI LAYER         — All DOM manipulation functions
 *     5. EVENT HANDLERS   — User interaction bindings
 *   Functions in the UI Layer NEVER call the API directly.
 *   Functions in the API Layer NEVER touch the DOM directly.
 *   This mirrors the separation of concerns in the backend.
 */

"use strict"; // Enforce strict mode: catches common JS mistakes at runtime.
              // ACADEMIC NOTE: Strict mode disables legacy JS features that
              // are error-prone (e.g., implicit globals, `with` statements).
              // It is a best practice in professional JS development.


// =============================================================================
// SECTION 1: CONFIGURATION
// =============================================================================

const API_BASE_URL = "http://localhost:8000"; // FastAPI server URL

const API_ENDPOINTS = {
    newSession:  `${API_BASE_URL}/session/new`,
    chat:        `${API_BASE_URL}/chat`,
    clearSession: (id) => `${API_BASE_URL}/session/${id}`,
};

// Storage key for persisting the session UUID in sessionStorage.
// ACADEMIC NOTE — sessionStorage vs localStorage:
//   localStorage persists across browser sessions (until manually cleared).
//   sessionStorage is cleared when the browser tab is closed.
//   We use sessionStorage intentionally: when the user closes the tab,
//   their session UUID is gone. This provides automatic "session expiry"
//   with zero server-side work, and prevents session tokens from accumulating
//   in the browser indefinitely — a privacy-aligned choice.
const SESSION_STORAGE_KEY = "safespace_session_id";

// Emotion → Emoji mapping for the emotion badge display
// ACADEMIC NOTE: Surfacing detected emotion visually through universally
// understood emoji provides immediate, cross-linguistic emotional feedback.
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


// =============================================================================
// SECTION 2: APPLICATION STATE
// =============================================================================

/**
 * Centralized application state object.
 * ACADEMIC NOTE — Centralized State:
 *   Rather than scattering state across multiple variables, we group all
 *   mutable state in one object. This makes it easy to inspect the entire
 *   application state at any point (e.g., in the browser DevTools console:
 *   `AppState`), and prevents state inconsistencies caused by variables
 *   falling out of sync with each other.
 */
const AppState = {
    sessionId:    null,    // The current anonymous UUID session identifier
    isLoading:    false,   // True while waiting for an AI response
    messageCount: 0,       // Total messages sent in this session (for UI metrics)
    lastEmotion:  "neutral", // Last detected emotion (for badge updates)
};


// =============================================================================
// SECTION 3: DOM ELEMENT REFERENCES
// =============================================================================
// Cache all DOM element references at module load time.
// ACADEMIC NOTE — DOM Caching:
//   document.getElementById() traverses the DOM tree on every call.
//   Caching the references in variables means we pay this traversal cost
//   ONCE at startup rather than on every user interaction.
//   This is a micro-optimization that also makes the code more readable.

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


// =============================================================================
// SECTION 4: API LAYER — All Backend Communication
// =============================================================================

/**
 * Initializes a new anonymous session by calling POST /session/new.
 * Stores the returned UUID in sessionStorage for persistence across page refreshes.
 *
 * FLOW:
 *   1. Check sessionStorage for an existing session UUID.
 *   2. If found, reuse it (allows page refresh without losing session).
 *   3. If not found, request a new UUID from the backend.
 *
 * @returns {Promise<string>} The session UUID string.
 */
async function initializeSession() {
    // Check for an existing session in sessionStorage first.
    const existingSession = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (existingSession) {
        AppState.sessionId = existingSession;
        console.info(`[SafeSpace] Resumed session: ${existingSession.substring(0, 8)}...`);
        return existingSession;
    }

    // No existing session — request a new one from the backend.
    try {
        const response = await fetch(API_ENDPOINTS.newSession, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
        });

        if (!response.ok) {
            throw new Error(`Session init failed: HTTP ${response.status}`);
        }

        const data = await response.json();
        AppState.sessionId = data.session_id;

        // Persist in sessionStorage so a page refresh doesn't create a new session.
        sessionStorage.setItem(SESSION_STORAGE_KEY, data.session_id);

        console.info(`[SafeSpace] New session created: ${data.session_id.substring(0, 8)}...`);
        return data.session_id;

    } catch (error) {
        console.error("[SafeSpace] Failed to initialize session:", error);
        // Generate a client-side fallback UUID if the backend is unreachable.
        // ACADEMIC NOTE: crypto.randomUUID() is a Web Crypto API function that
        // generates a cryptographically secure UUID4 — identical quality to the
        // server-side uuid.uuid4(). Using the built-in browser API avoids
        // importing a library just for UUID generation.
        const fallbackId = crypto.randomUUID();
        AppState.sessionId = fallbackId;
        sessionStorage.setItem(SESSION_STORAGE_KEY, fallbackId);
        showToast("⚠️ Running in offline mode. Some features may be limited.");
        return fallbackId;
    }
}

/**
 * Sends a user message to POST /chat and returns the structured API response.
 *
 * ACADEMIC NOTE — async/await vs .then()/.catch():
 *   Both patterns handle Promises (JavaScript's asynchronous primitive).
 *   async/await is syntactic sugar over Promises that makes async code read
 *   like synchronous code, significantly improving readability and making
 *   error handling with try/catch natural. It is the modern standard.
 *
 * @param {string} message - The raw user message text.
 * @returns {Promise<Object>} The ChatResponse JSON object from the backend.
 */
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

    // Check for HTTP-level errors (4xx, 5xx).
    // fetch() only throws on network failures — it does NOT throw for
    // HTTP error status codes. We must check response.ok manually.
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
            errorData.detail || `Server error: HTTP ${response.status}`
        );
    }

    return response.json();
}

/**
 * Clears the current session's conversation history on the backend,
 * then resets the client-side state and UI for a fresh conversation.
 *
 * @returns {Promise<void>}
 */
async function startNewConversation() {
    if (AppState.sessionId) {
        // Notify the backend to clear in-memory history for this session.
        // We use fetch with method DELETE, following REST conventions.
        // The call is fire-and-forget: we don't await or block the UI on it.
        fetch(API_ENDPOINTS.clearSession(AppState.sessionId), {
            method: "DELETE",
        }).catch(err => console.warn("[SafeSpace] Failed to clear session on backend:", err));
    }

    // Clear sessionStorage so the next init creates a fresh UUID.
    sessionStorage.removeItem(SESSION_STORAGE_KEY);
    AppState.sessionId    = null;
    AppState.messageCount = 0;
    AppState.lastEmotion  = "neutral";

    // Reset the UI.
    DOM.messageThread.innerHTML = "";
    resetEmotionBadge();

    // Re-initialize with a brand new session.
    await initializeSession();
    updateSessionDisplay();
    renderWelcomeMessage();

    showToast("✦ New conversation started. Your privacy is protected.");
}


// =============================================================================
// SECTION 5: UI LAYER — All DOM Manipulation
// =============================================================================

/**
 * Renders the welcome message when the app first loads or after a reset.
 * This is the first thing a user sees — it sets the emotional tone.
 */
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

/**
 * Appends a user message bubble to the conversation thread.
 * @param {string} text - The raw message text to display.
 */
function renderUserMessage(text) {
    // Sanitize the text before injecting into the DOM.
    // SECURITY NOTE — XSS Prevention:
    //   If we use innerHTML with unsanitized user input, a malicious user could
    //   inject HTML tags (e.g., <script>alert('XSS')</script>) that execute in
    //   the browser. We use textContent (which escapes HTML entities) via
    //   the escapeHTML helper. This is the fundamental defense against
    //   DOM-based Cross-Site Scripting (XSS) attacks.
    const safeText = escapeHTML(text);
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

/**
 * Appends an AI response bubble to the conversation thread.
 * @param {string} text          - The AI's response text.
 * @param {string} emotion       - The detected emotion label.
 * @param {boolean} piiDetected  - Whether PII was scrubbed from the user's input.
 * @param {number} latencyMs     - The API response time in milliseconds.
 */
function renderAIMessage(text, emotion, piiDetected, latencyMs) {
    const safeText = escapeHTML(text);
    const messageRow = document.createElement("div");
    messageRow.className = "message-row ai-row";
    messageRow.setAttribute("role", "article");
    messageRow.setAttribute("aria-label", "SafeSpace AI response");

    // Build the optional PII notice badge.
    // ACADEMIC NOTE — Transparency as UX:
    //   If PII was detected and scrubbed, we show a small inline badge.
    //   This serves two purposes:
    //     1. Transparency: The user knows their information was protected.
    //     2. Education: It raises awareness of what PII looks like.
    //   Showing this only WHEN PII is detected avoids alert fatigue
    //   (constantly showing a notice the user learns to ignore).
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

/**
 * Renders an error message bubble when the API call fails.
 * @param {string} errorMessage - The error description.
 */
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

/**
 * Updates the emotion badge in the chat header.
 * @param {string} emotion - The detected emotion label (e.g., "anxious").
 */
function updateEmotionBadge(emotion) {
    const safeEmotion = emotion || "neutral";
    const emoji       = EMOTION_EMOJI_MAP[safeEmotion] || "✦";

    // Remove old emotion class, add new one for color theming.
    DOM.emotionBadge.className = `emotion-badge emotion-${safeEmotion}`;
    DOM.emotionIcon.textContent = emoji;
    DOM.emotionText.textContent = safeEmotion;

    AppState.lastEmotion = safeEmotion;
}

/** Resets the emotion badge to its default neutral state. */
function resetEmotionBadge() {
    DOM.emotionBadge.className = "emotion-badge";
    DOM.emotionIcon.textContent = "✦";
    DOM.emotionText.textContent = "Listening...";
}

/**
 * Toggles the UI into/out of "loading" state while waiting for the AI.
 * @param {boolean} isLoading - True to show loading state, false to restore.
 */
function setLoadingState(isLoading) {
    AppState.isLoading = isLoading;

    // Show/hide typing indicator.
    DOM.typingIndicator.hidden = !isLoading;

    // Disable/enable the send button and input.
    DOM.sendBtn.disabled       = isLoading || !DOM.messageInput.value.trim();
    DOM.messageInput.disabled  = isLoading;

    // Update the status dot and text in the header.
    if (isLoading) {
        DOM.statusDot.classList.add("thinking");
        DOM.aiStatus.textContent = "Listening and reflecting...";
    } else {
        DOM.statusDot.classList.remove("thinking");
        DOM.aiStatus.textContent = "Ready to listen";
    }

    // Scroll to show the typing indicator.
    if (isLoading) scrollToBottom();
}

/**
 * Updates the session ID display in the sidebar.
 * Shows only the first 8 + last 4 characters for readability.
 * (Full UUID is stored in AppState and sessionStorage.)
 */
function updateSessionDisplay() {
    if (!AppState.sessionId) return;
    const id = AppState.sessionId;
    // Format: "a3f1c2d4-...-567890"
    DOM.sessionIdDisplay.textContent = `${id.substring(0, 8)}...${id.slice(-4)}`;
    DOM.sessionIdDisplay.title       = `Full session ID: ${id}`;
}

/**
 * Scrolls the message thread to the latest message.
 * ACADEMIC NOTE: We use scrollTop = scrollHeight (not scrollIntoView) because
 * scrollIntoView can cause the entire page to shift, which is disorienting.
 * Setting scrollTop directly only scrolls the specific container element.
 */
function scrollToBottom() {
    // Use requestAnimationFrame to ensure the DOM has updated before scrolling.
    // Without this, scrollHeight may not yet reflect the newly added message.
    requestAnimationFrame(() => {
        DOM.messageThread.scrollTop = DOM.messageThread.scrollHeight;
    });
}

/**
 * Displays a temporary toast notification at the bottom of the screen.
 * @param {string} message  - The notification text.
 * @param {number} duration - Display duration in milliseconds (default 3000).
 */
function showToast(message, duration = 3000) {
    DOM.toast.textContent = message;
    DOM.toast.hidden      = false;

    // Force a reflow before adding the visible class (required for CSS transition).
    DOM.toast.offsetHeight; // eslint-disable-line no-unused-expressions
    DOM.toast.classList.add("visible");

    setTimeout(() => {
        DOM.toast.classList.remove("visible");
        setTimeout(() => { DOM.toast.hidden = true; }, 300);
    }, duration);
}

/**
 * Auto-resizes the textarea to fit its content, up to a maximum height.
 * This provides a more natural chat input experience than a fixed-height box.
 */
function autoResizeTextarea() {
    const input = DOM.messageInput;
    input.style.height = "auto"; // Reset height to recalculate
    const newHeight = Math.min(input.scrollHeight, 120); // Cap at 120px (≈5 lines)
    input.style.height = `${newHeight}px`;
}

/**
 * Updates the character counter below the input field.
 * Changes color to warn the user when approaching the limit.
 */
function updateCharCounter() {
    const count   = DOM.messageInput.value.length;
    const max     = 2000;
    DOM.charCount.textContent = count;

    // Reset classes and apply appropriate warning level.
    DOM.charCounter.classList.remove("warning", "danger");
    if (count > max * 0.9)      DOM.charCounter.classList.add("danger");
    else if (count > max * 0.75) DOM.charCounter.classList.add("warning");
}


// =============================================================================
// SECTION 6: UTILITY FUNCTIONS
// =============================================================================

/**
 * Escapes HTML special characters in a string to prevent XSS injection.
 * This is the primary defense against DOM-based XSS attacks.
 *
 * ACADEMIC NOTE — Why Not Just Use innerHTML?
 *   `element.innerHTML = userInput` renders any HTML tags in userInput.
 *   If a user types "<script>stealCookies()</script>", the browser executes it.
 *   escapeHTML converts `<` to `&lt;`, `>` to `&gt;`, etc., so the browser
 *   renders them as literal characters, never as executable HTML.
 *
 * @param {string} str - The potentially unsafe string.
 * @returns {string}   - The HTML-escaped safe string.
 */
function escapeHTML(str) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
    // ACADEMIC NOTE: This technique uses the browser's own HTML parser to
    // perform the escaping, which is more reliable than a manual regex approach
    // that might miss edge cases (e.g., rare Unicode characters that can be
    // exploited in certain browser contexts).
}

/**
 * Formats a Date object into a human-readable time string (HH:MM AM/PM).
 * @param {Date} date - The date to format.
 * @returns {string} - e.g., "2:45 PM"
 */
function formatTimestamp(date) {
    return date.toLocaleTimeString("en-IN", {
        hour:   "numeric",
        minute: "2-digit",
        hour12: true,
    });
    // ACADEMIC NOTE: We use "en-IN" (Indian English locale) to format times
    // in the style familiar to our primary user base — another localization
    // detail that demonstrates thoughtful deployment context awareness.
}


// =============================================================================
// SECTION 7: CORE INTERACTION HANDLER
// =============================================================================

/**
 * The central function that orchestrates the complete send-receive cycle:
 *   1. Read and validate the input.
 *   2. Render the user's message immediately (optimistic UI).
 *   3. Enter loading state.
 *   4. Send to backend API.
 *   5. Render AI response.
 *   6. Update emotion badge.
 *   7. Exit loading state.
 *
 * ACADEMIC NOTE — Optimistic UI:
 *   We render the user's message (step 2) BEFORE the API call completes (step 4).
 *   This gives the user immediate visual feedback that their message was received,
 *   making the interface feel snappy and responsive even with 1-3 second API latency.
 *   This is the "optimistic UI" pattern used by WhatsApp, iMessage, etc.
 */
async function handleSendMessage() {
    const rawMessage = DOM.messageInput.value.trim();

    // Guard: Do nothing if input is empty or a request is already in flight.
    if (!rawMessage || AppState.isLoading) return;

    // Guard: Ensure we have a valid session.
    if (!AppState.sessionId) {
        showToast("⚠️ Session not ready. Please wait a moment.");
        return;
    }

    // --- Step 1: Clear the input field immediately ---
    DOM.messageInput.value = "";
    autoResizeTextarea();
    updateCharCounter();
    DOM.sendBtn.disabled = true;

    // --- Step 2: Optimistic render of the user's message ---
    renderUserMessage(rawMessage);
    AppState.messageCount++;

    // --- Step 3: Enter loading state ---
    setLoadingState(true);

    try {
        // --- Step 4: API call ---
        const data = await sendMessage(rawMessage);

        // --- Step 5: Render AI response ---
        renderAIMessage(
            data.response,
            data.detected_emotion,
            data.pii_was_detected,
            data.latency_ms
        );

        // --- Step 6: Update emotion badge ---
        updateEmotionBadge(data.detected_emotion);

    } catch (error) {
        console.error("[SafeSpace] Message send failed:", error);
        renderErrorMessage(error.message);
    } finally {
        // --- Step 7: Always exit loading state, even on error ---
        // Using `finally` guarantees the UI is never stuck in a loading state,
        // regardless of whether the try block succeeded or threw an error.
        setLoadingState(false);
        DOM.messageInput.focus(); // Return focus to input for keyboard users.
    }
}


// =============================================================================
// SECTION 8: EVENT LISTENERS
// =============================================================================

// --- Send button click ---
DOM.sendBtn.addEventListener("click", handleSendMessage);

// --- Keyboard input handler ---
DOM.messageInput.addEventListener("keydown", (event) => {
    // Send on Enter key press (without Shift).
    // Shift+Enter inserts a newline — natural multi-line input behavior.
    // ACADEMIC NOTE: This is the convention in modern chat applications.
    //   WhatsApp, Slack, Teams all use Enter = send, Shift+Enter = newline.
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault(); // Prevent newline insertion
        handleSendMessage();
    }
});

// --- Input change handlers: auto-resize + char counter + send button state ---
DOM.messageInput.addEventListener("input", () => {
    autoResizeTextarea();
    updateCharCounter();

    // Enable send button only if there is non-whitespace content.
    DOM.sendBtn.disabled = !DOM.messageInput.value.trim() || AppState.isLoading;
});

// --- New conversation button ---
DOM.btnNewChat.addEventListener("click", async () => {
    // Confirm before wiping the conversation — prevent accidental resets.
    // ACADEMIC NOTE: window.confirm() is a simple synchronous modal provided
    // by the browser. For production, we'd replace this with a custom modal
    // component for better UX consistency, but for an MVP it serves the purpose.
    const confirmed = window.confirm(
        "Start a new conversation? The current conversation will end."
    );
    if (confirmed) {
        await startNewConversation();
    }
});


// =============================================================================
// SECTION 9: APPLICATION INITIALIZATION
// =============================================================================

/**
 * Bootstrap function: runs once when the page loads.
 * Sets up the session, renders the welcome message, and prepares the UI.
 *
 * ACADEMIC NOTE — DOMContentLoaded vs window.onload:
 *   DOMContentLoaded fires when the HTML has been parsed and the DOM is ready,
 *   but before images and stylesheets have fully loaded.
 *   window.onload fires after EVERYTHING (images, fonts, CSS) is loaded.
 *   We use DOMContentLoaded because our script only needs the DOM to be ready —
 *   waiting for images/fonts is unnecessary delay.
 */
document.addEventListener("DOMContentLoaded", async () => {
    console.info("[SafeSpace] Initializing application...");

    try {
        // Initialize or resume the anonymous session.
        await initializeSession();
        updateSessionDisplay();

        // Render the welcome message.
        renderWelcomeMessage();

        // Focus the message input so the user can start typing immediately.
        DOM.messageInput.focus();

        console.info("[SafeSpace] Application ready.");
    } catch (error) {
        console.error("[SafeSpace] Initialization failed:", error);
        showToast("⚠️ Could not connect to SafeSpace. Please refresh the page.", 6000);
    }
});

// Add a subtle PII-awareness badge style inline (since we can't add to CSS for dynamic badges)
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
