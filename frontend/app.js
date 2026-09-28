const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sendButton = document.getElementById("sendButton");
const chatMessages = document.getElementById("chatMessages");

const API_URL = "https://swarnaskychatbot-production.up.railway.app/chat";


// ---------------------------------------------------------
// Escape HTML
// ---------------------------------------------------------

function escapeHTML(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


// ---------------------------------------------------------
// Format assistant response safely
// ---------------------------------------------------------

function formatAssistantMessage(text) {
    if (!text) {
        return "";
    }

    let formatted = escapeHTML(String(text));

    // Convert HTML-style line breaks to new lines
    formatted = formatted.replace(/&lt;br\s*\/?&gt;/gi, "\n");

    // Bold Markdown: **text**
    formatted = formatted.replace(
        /\*\*(.+?)\*\*/g,
        "<strong>$1</strong>"
    );

    // Convert Markdown table
    const lines = formatted.split("\n");

    const isTableLine = (line) => {
        return line.trim().startsWith("|") &&
               line.trim().endsWith("|");
    };

    const isSeparatorLine = (line) => {
        return /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$/.test(
            line
        );
    };

    const output = [];

    let i = 0;

    while (i < lines.length) {

        // ---------------------------------------------
        // Markdown table
        // ---------------------------------------------

        if (
            isTableLine(lines[i]) &&
            i + 1 < lines.length &&
            isSeparatorLine(lines[i + 1])
        ) {
            const headerCells = lines[i]
                .split("|")
                .slice(1, -1)
                .map(cell => cell.trim());

            i += 2;

            const rows = [];

            while (i < lines.length && isTableLine(lines[i])) {
                const cells = lines[i]
                    .split("|")
                    .slice(1, -1)
                    .map(cell => cell.trim());

                rows.push(cells);
                i++;
            }

            let table = `
                <div class="response-table-wrapper">
                    <table class="response-table">
                        <thead>
                            <tr>
            `;

            for (const cell of headerCells) {
                table += `<th>${cell}</th>`;
            }

            table += `
                            </tr>
                        </thead>
                        <tbody>
            `;

            for (const row of rows) {
                table += "<tr>";

                for (let j = 0; j < headerCells.length; j++) {
                    table += `<td>${row[j] || ""}</td>`;
                }

                table += "</tr>";
            }

            table += `
                        </tbody>
                    </table>
                </div>
            `;

            output.push(table);

            continue;
        }

        output.push(lines[i]);

        i++;
    }

    formatted = output.join("\n");

    // ---------------------------------------------
    // Bullet lists
    // ---------------------------------------------

    formatted = formatted.replace(
        /^[ \t]*[-*]\s+(.+)$/gm,
        '<div class="response-bullet">• $1</div>'
    );

    // ---------------------------------------------
    // Numbered lists
    // ---------------------------------------------

    formatted = formatted.replace(
        /^[ \t]*(\d+)\.\s+(.+)$/gm,
        '<div class="response-number">$1. $2</div>'
    );

    // ---------------------------------------------
    // Preserve line breaks
    // ---------------------------------------------

    formatted = formatted.replace(/\n/g, "<br>");

    return formatted;
}


// ---------------------------------------------------------
// Add chat message
// ---------------------------------------------------------

function addMessage(role, text) {
    const message = document.createElement("div");
    message.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "You" : "S";

    const content = document.createElement("div");
    content.className = "message-content";

    const name = document.createElement("div");
    name.className = "message-name";
    name.textContent = role === "user"
        ? "You"
        : "Swarnasky AI";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    if (role === "assistant") {
        bubble.innerHTML = formatAssistantMessage(text);
    } else {
        bubble.textContent = text;
    }

    content.appendChild(name);
    content.appendChild(bubble);

    message.appendChild(avatar);
    message.appendChild(content);

    chatMessages.appendChild(message);

    scrollToBottom();

    return message;
}


// ---------------------------------------------------------
// Typing indicator
// ---------------------------------------------------------

function addTypingMessage() {
    const message = document.createElement("div");

    message.className = "message assistant";
    message.id = "typingMessage";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "S";

    const content = document.createElement("div");
    content.className = "message-content";

    const name = document.createElement("div");
    name.className = "message-name";
    name.textContent = "Swarnasky AI";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    const typing = document.createElement("div");
    typing.className = "typing";

    for (let i = 0; i < 3; i++) {
        const dot = document.createElement("span");
        typing.appendChild(dot);
    }

    bubble.appendChild(typing);

    content.appendChild(name);
    content.appendChild(bubble);

    message.appendChild(avatar);
    message.appendChild(content);

    chatMessages.appendChild(message);

    scrollToBottom();
}


function removeTypingMessage() {
    const typingMessage =
        document.getElementById("typingMessage");

    if (typingMessage) {
        typingMessage.remove();
    }
}


// ---------------------------------------------------------
// Scroll
// ---------------------------------------------------------

function scrollToBottom() {
    chatMessages.scrollTop =
        chatMessages.scrollHeight;
}


// ---------------------------------------------------------
// Send message
// ---------------------------------------------------------

async function sendMessage(question) {

    addMessage("user", question);

    addTypingMessage();

    questionInput.disabled = true;
    sendButton.disabled = true;

    try {

        const response = await fetch(API_URL, {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                question: question
            })
        });

        let data;

        try {
            data = await response.json();
        } catch {
            data = {};
        }

        removeTypingMessage();

        if (!response.ok) {

            const errorMessage =
                data.detail ||
                "Something went wrong while processing your question.";

            addMessage(
                "assistant",
                errorMessage
            );

            return;
        }

        addMessage(
            "assistant",
            data.answer ||
            "I couldn't generate an answer."
        );

    } catch (error) {

        removeTypingMessage();

        addMessage(
            "assistant",
            "Unable to connect to the Swarnasky server. Please make sure the FastAPI server is running."
        );

        console.error(
            "Chat request failed:",
            error
        );

    } finally {

        questionInput.disabled = false;
        sendButton.disabled = false;

        questionInput.focus();
    }
}


// ---------------------------------------------------------
// Form submit
// ---------------------------------------------------------

chatForm.addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();

        const question =
            questionInput.value.trim();

        if (!question) {
            return;
        }

        questionInput.value = "";

        await sendMessage(question);
    }
);


// ---------------------------------------------------------
// Initial focus
// ---------------------------------------------------------

questionInput.focus();