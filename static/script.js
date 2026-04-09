function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function renderMarkdown(text) {
    const escaped = escapeHtml(text || "");
    const lines = escaped.split(/\r?\n/);
    let html = "";
    let inList = false;

    for (const line of lines) {
        const trimmed = line.trim();
        const isListItem = /^(\d+\.\s+|[-*]\s+)/.test(trimmed);

        if (isListItem && !inList) {
            html += "<ul>";
            inList = true;
        }
        if (!isListItem && inList) {
            html += "</ul>";
            inList = false;
        }

        const withBold = line.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
        if (isListItem) {
            const itemText = withBold.replace(/^(\d+\.\s+|[-*]\s+)/, "");
            html += `<li>${itemText}</li>`;
        } else if (trimmed === "") {
            html += "<br>";
        } else {
            html += `<p>${withBold}</p>`;
        }
    }

    if (inList) {
        html += "</ul>";
    }

    return html;
}

function appendMessage(chat, role, text) {
    const message = document.createElement("div");
    message.className = role;
    if (role === "bot") {
        message.innerHTML = renderMarkdown(text);
    } else {
        message.textContent = text;
    }
    chat.appendChild(message);
}

async function send() {
    const input = document.getElementById("question");
    const q = input.value.trim();
    const chat = document.getElementById("chatBox");

    if (!q) {
        return;
    }

    appendMessage(chat, "user", q);
    input.value = "";

    const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q })
    });

    const data = await res.json();
    appendMessage(chat, "bot", data.answer || "No response received.");
    chat.scrollTop = chat.scrollHeight;
}
