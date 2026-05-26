/* ===================================================================
   Ask AI — conversational front-end for the FoundryOps Copilot.
   Sends the running conversation to POST /api/ask and renders answers
   with a light typewriter reveal. Stateless backend; we keep history here.
   =================================================================== */
(function () {
    'use strict';

    var messagesEl = document.getElementById('ask-messages');
    var form = document.getElementById('ask-form');
    var input = document.getElementById('ask-input');
    var sendBtn = document.getElementById('ask-send');
    var statusEl = document.getElementById('ask-status');
    var clearLink = document.getElementById('ask-clear');

    if (!messagesEl || !form || !input) return;

    // Conversation history sent to the API (user/assistant turns only).
    var history = [];
    var busy = false;

    // ---- helpers ----------------------------------------------------
    function escapeHtml(s) {
        return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    // Minimal, safe markdown: escape first, then **bold**, `code`, hyphen bullets, paragraphs.
    function formatAnswer(text) {
        var safe = escapeHtml(text.trim());
        safe = safe.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');

        var blocks = safe.split(/\n{2,}/);
        return blocks.map(function (block) {
            var lines = block.split('\n');
            var isList = lines.every(function (l) { return /^\s*[-*]\s+/.test(l) || l.trim() === ''; })
                && lines.some(function (l) { return /^\s*[-*]\s+/.test(l); });
            if (isList) {
                var items = lines
                    .filter(function (l) { return /^\s*[-*]\s+/.test(l); })
                    .map(function (l) { return '<li>' + l.replace(/^\s*[-*]\s+/, '') + '</li>'; });
                return '<ul>' + items.join('') + '</ul>';
            }
            return '<p>' + block.replace(/\n/g, '<br>') + '</p>';
        }).join('');
    }

    function scrollToBottom() {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function addBubble(role, html) {
        var wrap = document.createElement('div');
        wrap.className = 'ask-bubble ' + role;
        var body = document.createElement('div');
        body.className = 'ask-bubble-body';
        body.innerHTML = html;
        wrap.appendChild(body);
        messagesEl.appendChild(wrap);
        scrollToBottom();
        return body;
    }

    function addTyping() {
        var wrap = document.createElement('div');
        wrap.className = 'ask-bubble assistant';
        wrap.id = 'ask-typing';
        wrap.innerHTML = '<div class="ask-bubble-body"><div class="ask-dots"><span></span><span></span><span></span></div></div>';
        messagesEl.appendChild(wrap);
        scrollToBottom();
    }

    function removeTyping() {
        var t = document.getElementById('ask-typing');
        if (t) t.remove();
    }

    // Reveal the answer progressively for a chat-like feel, then swap to the
    // formatted version. Crawl over cleaned plain text so raw markdown (** , - )
    // never flashes on screen.
    function typewriter(bodyEl, fullText) {
        var html = formatAnswer(fullText);
        var plainFull = fullText.replace(/\*\*/g, '').replace(/`/g, '').replace(/^\s*[-*]\s+/gm, '• ');
        var words = plainFull.split(/(\s+)/);
        // For long answers, render instantly to avoid a slow crawl.
        if (words.length > 140) { bodyEl.innerHTML = html; scrollToBottom(); return; }
        bodyEl.innerHTML = '';
        var plain = '';
        var i = 0;
        var timer = setInterval(function () {
            plain += words[i];
            bodyEl.textContent = plain;
            scrollToBottom();
            i++;
            if (i >= words.length) {
                clearInterval(timer);
                bodyEl.innerHTML = html;  // swap to formatted version at the end
                scrollToBottom();
            }
        }, 18);
    }

    function setBusy(state) {
        busy = state;
        sendBtn.disabled = state;
        input.disabled = state;
        statusEl.textContent = state ? 'Thinking…' : '';
    }

    function autoGrow() {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 160) + 'px';
    }

    // ---- send -------------------------------------------------------
    function send(question) {
        if (busy) return;
        question = (question || '').trim();
        if (!question) return;

        addBubble('user', '<p>' + escapeHtml(question) + '</p>');
        history.push({ role: 'user', content: question });
        input.value = '';
        autoGrow();
        setBusy(true);
        addTyping();

        fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: history })
        })
            .then(function (res) { return res.json().then(function (d) { return { ok: res.ok, data: d }; }); })
            .then(function (r) {
                removeTyping();
                var answer = (r.data && r.data.answer) || 'Sorry — no answer came back.';
                var body = addBubble('assistant', '');
                if (r.ok) {
                    history.push({ role: 'assistant', content: answer });
                    typewriter(body, answer);
                    if (r.data.latency_ms) {
                        statusEl.textContent = 'Answered in ' + Math.round(r.data.latency_ms) + ' ms';
                        setTimeout(function () { statusEl.textContent = ''; }, 4000);
                    }
                } else {
                    body.classList.add('ask-error');
                    body.innerHTML = formatAnswer(answer);
                }
            })
            .catch(function () {
                removeTyping();
                var body = addBubble('assistant', '');
                body.classList.add('ask-error');
                body.innerHTML = '<p>Couldn\'t reach the assistant. Check your connection and try again.</p>';
            })
            .finally(function () { setBusy(false); input.focus(); });
    }

    // ---- events -----------------------------------------------------
    form.addEventListener('submit', function (e) {
        e.preventDefault();
        send(input.value);
    });

    input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            send(input.value);
        }
    });

    input.addEventListener('input', autoGrow);

    document.querySelectorAll('.ask-chip').forEach(function (chip) {
        chip.addEventListener('click', function () {
            send(chip.getAttribute('data-q'));
        });
    });

    if (clearLink) {
        clearLink.addEventListener('click', function (e) {
            e.preventDefault();
            history = [];
            // Keep only the first (welcome) bubble.
            var bubbles = messagesEl.querySelectorAll('.ask-bubble');
            bubbles.forEach(function (b, idx) { if (idx > 0) b.remove(); });
            statusEl.textContent = '';
            input.focus();
        });
    }
})();
