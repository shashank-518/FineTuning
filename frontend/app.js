const messagesEl = document.getElementById("messages");
const form = document.getElementById("chat-form");
const promptEl = document.getElementById("prompt");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("status");

const starters = [
  "What is a legal badminton serve?",
  "Explain front-and-back doubles formation",
  "How do you choose racket string tension?",
];

function addMessage(text, role) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function setStatus(text, isError = false) {
  if (!text) {
    statusEl.hidden = true;
    statusEl.textContent = "";
    statusEl.classList.remove("error");
    return;
  }
  statusEl.hidden = false;
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

addMessage(
  "Try a starter: " + starters.map((s) => `“${s}”`).join(" · "),
  "hint"
);

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = promptEl.value.trim();
  if (!message) return;

  addMessage(message, "user");
  promptEl.value = "";
  sendBtn.disabled = true;
  setStatus("Thinking on CPU — this can take 20–60 seconds…");

  const thinking = addMessage("…", "bot");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || `Request failed (${res.status})`);
    }
    thinking.textContent = data.reply || "(empty reply)";
    setStatus("");
  } catch (err) {
    thinking.remove();
    setStatus(err.message || "Something went wrong", true);
  } finally {
    sendBtn.disabled = false;
    promptEl.focus();
  }
});

promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});
