/* Miya AI — frontend client */

(function () {
  "use strict";

  /* ── Configuration ──────────────────────────────────────────────────── */
  const API_BASE = location.origin;
  const WS_BASE  = API_BASE.replace(/^http/, "ws");
  const MAX_RECONNECT = 8;
  const RECONNECT_BASE_MS = 1000;

  /* ── State ──────────────────────────────────────────────────────────── */
  let sessionId   = localStorage.getItem("miya_session") || _newSessionId();
  let ws          = null;
  let reconnects  = 0;
  let streaming   = false;
  let streamBuf   = "";

  /* ── DOM refs ───────────────────────────────────────────────────────── */
  const $chat      = document.getElementById("chat");
  const $input     = document.getElementById("msg-input");
  const $send      = document.getElementById("btn-send");
  const $dot       = document.getElementById("status-dot");
  const $status    = document.getElementById("status-text");
  const $theme     = document.getElementById("btn-theme");
  const $newSess   = document.getElementById("btn-new-session");
  const $fileInput = document.getElementById("file-input");
  const $dropZone  = document.getElementById("drop-overlay");

  /* ── Helpers ────────────────────────────────────────────────────────── */
  function _newSessionId() {
    const id = "sess_" + crypto.randomUUID().replace(/-/g, "").slice(0, 12);
    localStorage.setItem("miya_session", id);
    return id;
  }

  /** Optional reply_language for voice-style sessions (localStorage or <html data-reply-language>). */
  function _replyLanguagePayload() {
    let rl = (localStorage.getItem("miya_reply_language") || "").trim();
    if (!rl) {
      rl = (document.documentElement.getAttribute("data-reply-language") || "").trim();
    }
    if (!rl || rl.toLowerCase() === "auto") return {};
    return { reply_language: rl };
  }

  function _escHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function _renderMarkdown(raw) {
    let html = _escHtml(raw);
    // code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, function (_, lang, code) {
      return '<pre><code class="lang-' + _escHtml(lang) + '">' + code + "</code></pre>";
    });
    // inline code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    // bold
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    // italic
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
    // line breaks
    html = html.replace(/\n/g, "<br>");
    return html;
  }

  /* ── Message rendering ─────────────────────────────────────────────── */
  function addMessage(role, content, meta) {
    const msg = document.createElement("div");
    msg.className = "msg " + role;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "U" : "M";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = _renderMarkdown(content);

    msg.appendChild(avatar);
    msg.appendChild(bubble);

    if (meta) {
      const metaEl = document.createElement("div");
      metaEl.className = "meta";
      if (meta.agent_used) metaEl.innerHTML += "<span>Agent: <strong>" + _escHtml(meta.agent_used) + "</strong></span>";
      if (meta.tools_used && meta.tools_used.length) metaEl.innerHTML += "<span>Tools: " + meta.tools_used.map(_escHtml).join(", ") + "</span>";
      if (meta.execution_time_ms) metaEl.innerHTML += "<span>" + meta.execution_time_ms + " ms</span>";
      if (meta.confidence) metaEl.innerHTML += "<span>" + (meta.confidence * 100).toFixed(0) + "% conf</span>";
      bubble.appendChild(metaEl);
    }

    $chat.appendChild(msg);
    $chat.scrollTop = $chat.scrollHeight;
    return bubble;
  }

  function addTypingIndicator() {
    const msg = document.createElement("div");
    msg.className = "msg assistant";
    msg.id = "typing-indicator";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "M";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';

    msg.appendChild(avatar);
    msg.appendChild(bubble);
    $chat.appendChild(msg);
    $chat.scrollTop = $chat.scrollHeight;
  }

  function removeTypingIndicator() {
    const el = document.getElementById("typing-indicator");
    if (el) el.remove();
  }

  /* ── WebSocket ─────────────────────────────────────────────────────── */
  function connectWs() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

    const url = WS_BASE + "/ws/" + sessionId;
    ws = new WebSocket(url);

    ws.onopen = function () {
      reconnects = 0;
      $dot.classList.add("connected");
      $status.textContent = "Connected";
    };

    ws.onclose = function () {
      $dot.classList.remove("connected");
      $status.textContent = "Disconnected";
      _scheduleReconnect();
    };

    ws.onerror = function () {
      $dot.classList.remove("connected");
      $status.textContent = "Error";
    };

    ws.onmessage = function (evt) {
      var data;
      try { data = JSON.parse(evt.data); } catch (_) { return; }

      switch (data.type) {
        case "ping":
          ws.send(JSON.stringify({ type: "pong" }));
          break;

        case "status":
          if (data.status === "processing") addTypingIndicator();
          break;

        case "stream_start":
          removeTypingIndicator();
          streaming = true;
          streamBuf = "";
          addMessage("assistant", "");
          break;

        case "token":
          if (streaming) {
            streamBuf += data.content;
            var bubbles = $chat.querySelectorAll(".msg.assistant .bubble");
            var last = bubbles[bubbles.length - 1];
            if (last) last.innerHTML = _renderMarkdown(streamBuf);
            $chat.scrollTop = $chat.scrollHeight;
          }
          break;

        case "stream_end":
          streaming = false;
          break;

        case "response":
          removeTypingIndicator();
          addMessage("assistant", data.response || "", {
            agent_used: data.agent_used,
            tools_used: data.tools_used,
            execution_time_ms: data.execution_time_ms,
            confidence: data.confidence,
          });
          break;

        case "error":
          removeTypingIndicator();
          addMessage("assistant", "**Error:** " + (data.detail || "Unknown error"));
          break;
      }
    };
  }

  function _scheduleReconnect() {
    if (reconnects >= MAX_RECONNECT) {
      $status.textContent = "Offline";
      return;
    }
    var delay = RECONNECT_BASE_MS * Math.pow(2, reconnects);
    reconnects++;
    $status.textContent = "Reconnecting …";
    setTimeout(connectWs, delay);
  }

  /* ── Send message ──────────────────────────────────────────────────── */
  function sendMessage() {
    var text = $input.value.trim();
    if (!text) return;

    addMessage("user", text);
    $input.value = "";
    _autoResize();

    /* stream:false → orchestrator.process (agent chat()) — ishonchli.
       stream:true → token-by-token; backend chat_stream ishlatadi. */
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify(
          Object.assign({ type: "chat", message: text, stream: false }, _replyLanguagePayload())
        )
      );
    } else {
      _sendViaHttp(text);
    }
  }

  async function _sendViaHttp(text) {
    addTypingIndicator();
    try {
      var chatBody = Object.assign(
        { message: text, session_id: sessionId },
        _replyLanguagePayload()
      );
      var resp = await fetch(API_BASE + "/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(chatBody),
      });
      var data = await resp.json();
      removeTypingIndicator();
      if (resp.ok) {
        addMessage("assistant", data.response || "", {
          agent_used: data.agent_used,
          tools_used: data.tools_used,
          execution_time_ms: data.execution_time_ms,
          confidence: data.confidence,
        });
      } else {
        addMessage("assistant", "**Error " + resp.status + ":** " + (data.detail || data.error || ""));
      }
    } catch (e) {
      removeTypingIndicator();
      addMessage("assistant", "**Connection error:** " + e.message);
    }
  }

  /* ── File upload ───────────────────────────────────────────────────── */
  async function uploadFile(file) {
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) {
      addMessage("assistant", "File exceeds 10 MB limit.");
      return;
    }
    var form = new FormData();
    form.append("file", file);
    try {
      var resp = await fetch(API_BASE + "/api/v1/upload", { method: "POST", body: form });
      var data = await resp.json();
      if (resp.ok) {
        addMessage("assistant", "Uploaded **" + _escHtml(data.filename) + "** (" + data.size_bytes + " bytes)");
      } else {
        addMessage("assistant", "Upload failed: " + (data.detail || data.error || ""));
      }
    } catch (e) {
      addMessage("assistant", "Upload error: " + e.message);
    }
  }

  /* ── Auto-resize textarea ──────────────────────────────────────────── */
  function _autoResize() {
    $input.style.height = "auto";
    $input.style.height = Math.min($input.scrollHeight, 160) + "px";
  }

  /* ── Theme toggle ──────────────────────────────────────────────────── */
  function toggleTheme() {
    var html = document.documentElement;
    var next = html.getAttribute("data-theme") === "dark" ? "light" : "dark";
    html.setAttribute("data-theme", next);
    localStorage.setItem("miya_theme", next);
  }

  (function restoreTheme() {
    var saved = localStorage.getItem("miya_theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
  })();

  /* ── Drag & Drop ───────────────────────────────────────────────────── */
  var dragCounter = 0;
  document.addEventListener("dragenter", function (e) { e.preventDefault(); dragCounter++; $dropZone.classList.add("active"); });
  document.addEventListener("dragleave", function (e) { e.preventDefault(); dragCounter--; if (dragCounter <= 0) { dragCounter = 0; $dropZone.classList.remove("active"); } });
  document.addEventListener("dragover",  function (e) { e.preventDefault(); });
  document.addEventListener("drop", function (e) {
    e.preventDefault();
    dragCounter = 0;
    $dropZone.classList.remove("active");
    var files = e.dataTransfer.files;
    if (files.length > 0) uploadFile(files[0]);
  });

  /* ── Event wiring ──────────────────────────────────────────────────── */
  $send.addEventListener("click", sendMessage);
  $input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  $input.addEventListener("input", _autoResize);
  $theme.addEventListener("click", toggleTheme);
  $newSess.addEventListener("click", function () {
    sessionId = _newSessionId();
    $chat.innerHTML = "";
    if (ws) ws.close();
    connectWs();
  });
  $fileInput.addEventListener("change", function () {
    if ($fileInput.files.length) uploadFile($fileInput.files[0]);
    $fileInput.value = "";
  });

  /* ── Boot ───────────────────────────────────────────────────────────── */
  connectWs();
})();
