import "./style.css";
import { questionFor } from "./demo";
import { ChatResponse, parseTools, type DishRating, type Google, type Nutrition, type Jev, type Question, type State, type ToolResult } from "./schemas";

/*
 * Backend: restaurent-agent (FastAPI), a guided flow: cuisine -> restaurant -> date -> party size -> availability.
 * Jev is the agent: it reads each answer, the backend runs the matching tool, and Jev replies.
 * Gemini is a separate execution on the same input (own history, own tool calls), shown next to Jev's.
 *   POST /chat   { message, history: [{role, content}], gemini_history: [{role, content}], state? }
 *                -> { google: { source, available, reply, tool_calls: [{name, args, output}] },
 *                     jev: { source, available, understood: {question_id, choice, confidence, method},
 *                            reply, tool_calls: [{name, args, output}],
 *                            next_question: {id, text, choices: [{key, label, message}]} },
 *                     state }   <- sent back unchanged with the next request
 *   GET  /health -> { status: "ok" }
 * The user can click a choice (sends its `message`) or type freely.
 */

function $<T extends HTMLElement>(id: string): T {
  const e = document.getElementById(id);
  if (!e) throw new Error(`#${id} missing`);
  return e as T;
}
const chat = $("chat");
const form = $<HTMLFormElement>("form");
const input = $<HTMLInputElement>("input");
const apiUrl = $<HTMLInputElement>("apiUrl");
const health = $("health");
const sendBtn = form.querySelector("button") as HTMLButtonElement;

apiUrl.value = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

type Turn = { role: "user" | "assistant"; content: string };
const history: Turn[] = []; // Jev's conversation
const geminiHistory: Turn[] = []; // Gemini's own conversation
let state: State | null = null; // echoed back to the backend on every request

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls = "", text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
const jsonBlock = (v: unknown) => el("pre", "", JSON.stringify(v, null, 2));
const scrollDown = () => { chat.scrollTop = chat.scrollHeight; };
const base = () => apiUrl.value.replace(/\/+$/, "");

// ---------- Backend ----------
async function ask(message: string): Promise<ChatResponse> {
  const res = await fetch(`${base()}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, gemini_history: geminiHistory, state }),
  });
  if (!res.ok) {
    const detail = await res.json().then((j: { detail?: unknown }) => (typeof j.detail === "string" ? j.detail : null), () => null);
    throw new Error(detail ?? `Backend returned ${res.status}`);
  }
  return ChatResponse.parse(await res.json());
}

async function checkHealth(): Promise<void> {
  try {
    const r = await fetch(`${base()}/health`);
    const ok = r.ok && ((await r.json()) as { status?: string }).status === "ok";
    health.className = ok ? "dot ok" : "dot bad";
    health.title = ok ? "Backend online" : "Backend unhealthy";
  } catch {
    health.className = "dot bad";
    health.title = "Backend unreachable";
  }
}

// ---------- Rendering ----------
/** FDA grade A-E: A/B green, D/E red, C neutral. */
const gradeBadge = (grade: string, score: number) =>
  el("span", `badge ${"AB".includes(grade) ? "ok" : "DE".includes(grade) ? "bad" : ""}`, `Grade ${grade} · ${score}/100`);

const nutritionList = (n: Nutrition) => {
  const dl = el("dl");
  const rows: [string, string][] = [
    ["calories", `${n.calories} kcal`], ["sat fat", `${n.satFat} g`], ["sodium", `${n.sodium} mg`],
    ["added sugar", `${n.sugar} g`], ["fiber", `${n.fiber} g`], ["protein", `${n.protein} g`],
  ];
  for (const [k, v] of rows) dl.append(el("dt", "", k), el("dd", "", v));
  return dl;
};

function dishCard(d: DishRating): HTMLElement {
  const card = el("div", "card");
  card.append(el("h3", "", d.name), gradeBadge(d.grade, d.score));
  if (d.flags.length) card.append(el("div", "meta", d.flags.join(" · ")));
  if (d.source === "gemini-estimate") card.append(el("div", "meta", "Nutrition estimated by AI"));
  card.append(nutritionList(d.nutrition));
  return card;
}

function renderTool(t: ToolResult): HTMLElement {
  const box = el("div", "tool");

  if (!t.ok) {
    box.append(el("div", "tool-name", `🔧 ${t.name}`));
    box.append(el("div", "msg bot err", `Unexpected tool output (${t.issues})`), jsonBlock(t.raw));
    return box;
  }

  const c = t.call;
  const args = Object.entries(c.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
  box.append(el("div", "tool-name", `🔧 ${c.name}(${args})`));
  const grid = el("div", "cards");

  // Exhaustive switch: adding a tool to the schema without a renderer is a compile error.
  switch (c.name) {
    case "search_restaurants_by_cuisine": {
      if (!c.output.length) { box.append(el("div", "msg bot", "No restaurants found for that cuisine.")); break; }
      for (const r of c.output) {
        const card = el("div", "card");
        card.append(el("h3", "", r.name), el("div", "meta", `⭐ ${r.rating}  ·  ${r.price}`));
        grid.append(card);
      }
      box.append(grid);
      break;
    }
    case "get_restaurant_details": {
      if ("error" in c.output) { box.append(el("div", "msg bot err", c.output.error)); break; }
      const card = el("div", "card");
      const dl = el("dl");
      for (const [k, v] of Object.entries(c.output)) dl.append(el("dt", "", k), el("dd", "", String(v)));
      card.append(dl);
      grid.append(card);
      box.append(grid);
      break;
    }
    case "check_reservation_availability": {
      const ok = c.output.startsWith("✓");
      const card = el("div", "card");
      card.append(el("span", `badge ${ok ? "ok" : "bad"}`, ok ? "Available" : "Unavailable"), el("div", "meta", c.output));
      grid.append(card);
      box.append(grid);
      break;
    }
    case "rate_restaurant_menu": {
      if (typeof c.output === "string") { box.append(el("div", "msg bot err", c.output)); break; }
      const m = c.output;
      const head = el("div", "card");
      head.append(
        el("h3", "", m.restaurant),
        gradeBadge(m.overall.grade, m.overall.score),
        el("div", "meta", `${m.overall.dishCount} dishes · best: ${m.best ?? "-"} · lowest: ${m.lowest ?? "-"}`),
      );
      grid.append(head);
      for (const d of m.dishes) grid.append(dishCard(d));
      box.append(grid, el("div", "meta", m.disclaimer));
      break;
    }
    case "rate_dish": {
      if (typeof c.output === "string") { box.append(el("div", "msg bot err", c.output)); break; }
      grid.append(dishCard(c.output));
      box.append(grid);
      break;
    }
    case "estimate_dish_nutrition": {
      if (typeof c.output === "string") { box.append(el("div", "msg bot err", c.output)); break; }
      const card = el("div", "card");
      card.append(el("div", "meta", "Estimated per serving (AI estimate)"), nutritionList(c.output));
      grid.append(card);
      box.append(grid);
      break;
    }
    case "explain_dish_rating": {
      const card = el("div", "card");
      card.append(el("div", "meta", c.output));
      grid.append(card);
      box.append(grid);
      break;
    }
    default: {
      const _never: never = c;
      void _never;
    }
  }
  return box;
}

/** Gemini's separate run on the same input: its tool results and its reply. */
function renderGoogle(g: Google): HTMLElement {
  const box = el("div", "answer google");
  box.append(el("div", "source-tag", g.source));
  if (!g.available) {
    box.append(el("div", "badge bad", "Gemini unavailable"));
    return box;
  }
  parseTools(g.tool_calls).forEach((t) => box.append(renderTool(t)));
  if (g.reply) box.append(el("div", "answer-text", g.reply));
  return box;
}

/** Jev's answer: tool results, its reply, how it read the user's last answer, then the next question. Clicking a choice sends its `message`; typing works too. */
function renderJev(jev: Jev | null, q: Question): HTMLElement {
  const box = el("div", "answer jev");
  box.append(el("div", "source-tag", jev?.source ?? "Jev"));
  if (jev) {
    parseTools(jev.tool_calls).forEach((t) => box.append(renderTool(t)));
    if (jev.reply) box.append(el("div", "answer-text", jev.reply));
    const u = jev.understood;
    if (!jev.available) {
      box.append(el("div", "badge bad", "Jev unavailable, matched by text"));
    } else {
      const pct = u.confidence === null ? "" : ` · ${Math.round(u.confidence * 100)}%`;
      box.append(el("div", "badge", `understood ${u.question_id}: ${u.choice ?? "no match"}${pct}`));
    }
  }
  const opts = el("div", "options");
  opts.append(el("div", "options-prompt", q.text));
  const row = el("div", "chips-row");
  for (const c of q.choices) {
    const b = el("button", c.key === "restart" ? "own" : "", c.label);
    b.type = "button";
    b.title = c.description ?? "";
    b.addEventListener("click", () => void send(c.message));
    row.append(b);
  }
  opts.append(row);
  box.append(opts);
  return box;
}

// ---------- Events ----------
async function send(text: string): Promise<void> {
  text = text.trim();
  if (!text) return;
  document.querySelectorAll(".options").forEach((n) => n.remove()); // old choices are stale now (the Jev label and understanding stay)
  chat.append(el("div", "msg user", text));
  input.value = "";
  const typing = el("div", "msg bot typing", "Thinking…");
  chat.append(typing);
  scrollDown();
  sendBtn.disabled = true;
  try {
    const { google, jev, state: next } = await ask(text);
    state = next;
    history.push({ role: "user", content: text }, { role: "assistant", content: jev.reply });
    if (google) {
      geminiHistory.push({ role: "user", content: text }, { role: "assistant", content: google.reply });
      chat.insertBefore(renderGoogle(google), typing);
    }
    chat.insertBefore(renderJev(jev, jev.next_question), typing);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    chat.append(el("div", "msg bot err", `${msg} (is the backend running at ${base()}?)`));
  } finally {
    typing.remove();
    sendBtn.disabled = false;
    input.focus();
    scrollDown();
  }
}

form.addEventListener("submit", (e) => { e.preventDefault(); void send(input.value); });
apiUrl.addEventListener("change", () => void checkHealth());

chat.append(el("div", "msg bot", "Hi! I can find restaurants by cuisine, show details, and check reservation availability."));
chat.append(renderJev(null, questionFor({ asking: "cuisine" }))); // the backend's first question (flow.py START)
void checkHealth();
