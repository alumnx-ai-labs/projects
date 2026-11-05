/** Offline stand-in for POST /chat. Mirrors flow.py: cuisine -> restaurant -> date -> party -> availability. */
import type { Question, State } from "./schemas";

const RESTAURANTS: Record<string, Array<[string, number, string]>> = {
  indian: [["Spice Palace", 4.5, "₹₹"], ["Curry House", 4.3, "₹₹"], ["Taj Flavors", 4.7, "₹₹₹"]],
  italian: [["Pizza Bella", 4.4, "₹₹"], ["Pasta Dreams", 4.6, "₹₹₹"], ["Roma Kitchen", 4.2, "₹₹"]],
  chinese: [["Dragon House", 4.5, "₹₹"], ["Ming's Kitchen", 4.3, "₹₹"], ["Golden Wok", 4.8, "₹₹₹"]],
  mexican: [["Taco Fiesta", 4.4, "₹₹"], ["Salsa Nights", 4.6, "₹₹₹"], ["Burrito House", 4.2, "₹₹"]],
};

const DETAILS: Record<string, Record<string, string | number>> = {
  "Spice Palace": { cuisine: "Indian", rating: 4.5, address: "123 Food Street, City Center", phone: "9876543210", hours: "11 AM - 11 PM", speciality: "Butter Chicken, Biryani" },
  "Pizza Bella": { cuisine: "Italian", rating: 4.4, address: "456 Dine Avenue, Downtown", phone: "9123456789", hours: "12 PM - 10:30 PM", speciality: "Margherita Pizza, Tiramisu" },
  "Dragon House": { cuisine: "Chinese", rating: 4.5, address: "789 Noodle Lane, Market Area", phone: "9111222333", hours: "11:30 AM - 11 PM", speciality: "Hakka Noodles, Fried Rice" },
  "Taco Fiesta": { cuisine: "Mexican", rating: 4.4, address: "321 Spice Row, Hip District", phone: "9222333444", hours: "1 PM - 12 AM", speciality: "Fish Tacos, Guacamole" },
  "Curry House": { cuisine: "Indian", rating: 4.3, address: "555 Flavor Street, Old Town", phone: "9333444555", hours: "11 AM - 10 PM", speciality: "Paneer Tikka, Dal Makhani" },
  "Taj Flavors": { cuisine: "Indian", rating: 4.7, address: "999 Taste Lane, Premium Zone", phone: "9444555666", hours: "12 PM - 11 PM", speciality: "Biryani, Tandoori Chicken" },
  "Golden Wok": { cuisine: "Chinese", rating: 4.8, address: "777 Wok Street, Central Market", phone: "9555666777", hours: "11 AM - 11 PM", speciality: "Szechuan Noodles, Dim Sum" },
};

const SLOTS: Record<string, Record<string, boolean>> = {
  "Spice Palace": { "2024-11-15": true, "2024-11-16": false, "2024-11-17": true },
  "Pizza Bella": { "2024-11-15": false, "2024-11-16": true, "2024-11-17": true },
  "Dragon House": { "2024-11-15": true, "2024-11-16": true, "2024-11-17": false },
  "Taco Fiesta": { "2024-11-15": true, "2024-11-16": true, "2024-11-17": true },
  "Curry House": { "2024-11-15": true, "2024-11-16": false, "2024-11-17": true },
  "Taj Flavors": { "2024-11-15": false, "2024-11-16": true, "2024-11-17": true },
};

// ---- mirrors flow.py ----
export const CUISINES = ["Indian", "Italian", "Chinese", "Mexican"];
const DATES: Array<[string, string]> = [["15 Nov 2024", "2024-11-15"], ["16 Nov 2024", "2024-11-16"], ["17 Nov 2024", "2024-11-17"]];
const PARTY_SIZES = [2, 4, 6];
const RESTART_LABEL = "None of these - start over";

type Choice = Question["choices"][number];
const choice = (key: string, label: string, description: string): Choice => ({ key, label, description, message: label });
const question = (id: string, text: string, choices: Choice[], restartable = true): Question => ({
  id, type: "choice", text,
  choices: restartable ? [...choices, choice("restart", RESTART_LABEL, "The user wants to start over")] : choices,
});

export function questionFor(state: State): Question {
  const asking = (state.asking as string | undefined) ?? "cuisine";
  if (asking === "restaurant") {
    const cuisine = state.cuisine as string;
    return question("restaurant", `Which ${cuisine} restaurant would you like?`,
      (RESTAURANTS[cuisine.toLowerCase()] ?? []).map(([n]) => choice(n, n, `The restaurant ${n}`)));
  }
  if (asking === "date") {
    return question("date", `When would you like to book ${state.restaurant as string}?`,
      DATES.map(([label, iso]) => choice(iso, label, `The date ${label}`)));
  }
  if (asking === "party") {
    return question("party", "How many people?", PARTY_SIZES.map((n) => choice(String(n), `${n} people`, `A party of ${n}`)));
  }
  return question("cuisine", "What cuisine would you like?", CUISINES.map((c) => choice(c.toLowerCase(), c, `${c} cuisine`)), false);
}

function matchLabel(q: Question, message: string): string | null {
  const t = message.trim().toLowerCase();
  return q.choices.find((c) => t === c.label.toLowerCase() || t === c.key.toLowerCase())?.key ?? null;
}

type Step = { state: State; reply: string; tool: unknown | null; notice: string | null };

function step(state: State, picked: string | null): Step {
  const asking = (state.asking as string | undefined) ?? "cuisine";
  const same: Step = {
    state, notice: null, tool: null,
    reply: "I can only help with restaurants here. Pick one of the options below, or ask about a cuisine, restaurant or booking.",
  };
  if (picked === null) return same;
  if (picked === "restart" && asking !== "cuisine") {
    return { state: { asking: "cuisine" }, reply: "No problem, let's begin again!", tool: null, notice: "Okay, let's start over." };
  }
  if (asking === "cuisine") {
    const cuisine = CUISINES.find((c) => c.toLowerCase() === picked);
    if (!cuisine) return same;
    const list = RESTAURANTS[picked]!.map(([name, rating, price]) => ({ name, rating, price }));
    return {
      state: { asking: "restaurant", cuisine }, notice: null,
      reply: `Here are some ${cuisine} restaurants you might like.`,
      tool: { name: "search_restaurants_by_cuisine", args: { cuisine_type: cuisine }, output: JSON.stringify(list) },
    };
  }
  if (asking === "restaurant") {
    return {
      state: { ...state, asking: "date", restaurant: picked }, notice: null,
      reply: `Here are the details for ${picked}.`,
      tool: { name: "get_restaurant_details", args: { restaurant_name: picked }, output: JSON.stringify(DETAILS[picked] ?? { error: `Restaurant '${picked}' not found` }) },
    };
  }
  if (asking === "date") {
    return { state: { ...state, asking: "party", date: picked }, notice: null, tool: null, reply: `Great, ${picked} it is.` };
  }
  if (asking === "party") {
    const party = Number(picked);
    const name = state.restaurant as string;
    const date = state.date as string;
    const ok = !!SLOTS[name]?.[date];
    const output = ok ? `✓ ${name} has availability for ${party} people on ${date}` : `✗ ${name} is fully booked for ${date}`;
    return {
      state: { asking: "cuisine" },
      reply: ok ? `Good news, ${name} has a table for ${party} on ${date}.` : `Sorry, ${name} is fully booked on ${date}. We'll start over.`,
      tool: { name: "check_reservation_availability", args: { restaurant_name: name, date, party_size: party }, output },
      notice: ok ? "Good news, that table is available! Want to look for another?" : "Sorry, that table isn't available. Let's start over.",
    };
  }
  return same;
}

export function demoChat(message: string, state: State | null): unknown {
  const current: State = state ?? { asking: "cuisine" };
  const asked = questionFor(current);
  const picked = matchLabel(asked, message);
  const s = step(current, picked);
  const next = questionFor(s.state);
  if (s.notice) next.text = `${s.notice} ${next.text}`;
  return {
    google: { source: "Google Gemini", reply: s.reply, tool_calls: s.tool ? [s.tool] : [] },
    jev: {
      source: "Jev",
      available: true,
      understood: { question_id: asked.id, choice: picked ?? "other", confidence: picked ? 0.95 : 0.5, method: "jev" },
      next_question: next,
    },
    state: s.state,
  };
}
