const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const chatBox = document.getElementById("chatBox");
const typing = document.getElementById("typing");


function addMessage(message, type) {

    const messageDiv = document.createElement("div");

    messageDiv.classList.add("message");

    if (type === "user") {

        messageDiv.classList.add("user-message");

        messageDiv.innerHTML = `
            <div class="message-content">
                <p>${escapeHTML(message)}</p>
            </div>
        `;

    } else {

        messageDiv.classList.add("bot-message");

        messageDiv.innerHTML = `
            <div class="avatar">✨</div>

            <div class="message-content">
                <p>${escapeHTML(message)}</p>
            </div>
        `;
    }

    chatBox.appendChild(messageDiv);

    chatBox.scrollTop = chatBox.scrollHeight;
}


function escapeHTML(text) {

    const div = document.createElement("div");

    div.textContent = text;

    return div.innerHTML;
}


async function sendMessage() {

    const message = messageInput.value.trim();

    if (!message) {
        return;
    }

    // Show user message
    addMessage(message, "user");

    // Clear input
    messageInput.value = "";

    // Show typing
    typing.classList.remove("hidden");

    sendButton.disabled = true;

    try {

        const response = await fetch("/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                message: message
            })
        });


        const data = await response.json();


        if (response.ok) {

            addMessage(data.reply, "bot");

        } else {

            addMessage(
                "Sorry, I could not process your request.",
                "bot"
            );
        }

    } catch (error) {

        console.error(error);

        addMessage(
            "Unable to connect to the server.",
            "bot"
        );

    } finally {

        typing.classList.add("hidden");

        sendButton.disabled = false;

        messageInput.focus();
    }
}


sendButton.addEventListener(
    "click",
    sendMessage
);


messageInput.addEventListener(
    "keydown",
    function(event) {

        if (event.key === "Enter") {

            sendMessage();
        }
    }
);