import { z } from "zod";

// ---- Tool output schemas (mirror restaurent-agent/tools.py) ----
export const RestaurantSummary = z.object({
  name: z.string(),
  rating: z.number(),
  price: z.string(),
});

export const RestaurantDetails = z.object({
  cuisine: z.string(),
  rating: z.number(),
  address: z.string(),
  phone: z.string(),
  hours: z.string(),
  speciality: z.string(),
});

export const ToolError = z.object({ error: z.string() });

// ---- FDA menu rating MCP tools (fda-menu-rating mcp-tools.js); scores are 0-100, grades A-E ----
export const Nutrition = z.object({
  calories: z.number(),
  satFat: z.number(),
  sodium: z.number(),
  sugar: z.number(),
  fiber: z.number(),
  protein: z.number(),
});
export type Nutrition = z.infer<typeof Nutrition>;

export const DishRating = z.object({
  name: z.string(),
  score: z.number(),
  grade: z.string(),
  flags: z.array(z.string()),
  nutrition: Nutrition,
  source: z.string().optional(), // "provided" | "gemini-estimate" (menu ratings only)
});
export type DishRating = z.infer<typeof DishRating>;

export const MenuRating = z.object({
  restaurant: z.string(),
  overall: z.object({ score: z.number(), grade: z.string(), dishCount: z.number() }),
  best: z.string().optional(),
  lowest: z.string().optional(),
  dishes: z.array(DishRating),
  disclaimer: z.string(),
});

const fdaArgs = z.record(z.string(), z.unknown());

/** The backend sends every tool result as a string; list/dict results are JSON text. */
const jsonText = <T extends z.ZodTypeAny>(schema: T) =>
  z.preprocess((v) => {
    if (typeof v !== "string") return v;
    try { return JSON.parse(v); } catch { return v; }
  }, schema);

// ---- Tool calls: discriminated union on `name`; the backend sends name + args + output ----
export const ToolCall = z.discriminatedUnion("name", [
  z.object({
    name: z.literal("search_restaurants_by_cuisine"),
    args: z.object({ cuisine_type: z.string() }),
    output: jsonText(z.array(RestaurantSummary)),
  }),
  z.object({
    name: z.literal("get_restaurant_details"),
    args: z.object({ restaurant_name: z.string() }),
    output: jsonText(z.union([RestaurantDetails, ToolError])),
  }),
  z.object({
    name: z.literal("check_reservation_availability"),
    args: z.object({
      restaurant_name: z.string(),
      date: z.string(),
      party_size: z.number(),
      time: z.string().optional(),
    }),
    output: z.string(),
  }),
  // FDA tools: a plain string output is the server's error message.
  z.object({
    name: z.literal("rate_restaurant_menu"),
    args: fdaArgs,
    output: jsonText(z.union([MenuRating, z.string()])),
  }),
  z.object({
    name: z.literal("rate_dish"),
    args: fdaArgs,
    output: jsonText(z.union([DishRating, z.string()])),
  }),
  z.object({
    name: z.literal("estimate_dish_nutrition"),
    args: fdaArgs,
    output: jsonText(z.union([Nutrition, z.string()])),
  }),
  z.object({
    name: z.literal("explain_dish_rating"),
    args: fdaArgs,
    output: z.string(),
  }),
]);
export type ToolCall = z.infer<typeof ToolCall>;

// ---- Jev's question format (flow.py): clicking a choice sends its `message` ----
export const Choice = z.object({
  key: z.string(),
  label: z.string(),
  description: z.string().optional(),
  message: z.string(),
});
export const Question = z.object({
  id: z.string(), // cuisine | restaurant | date | party | time
  type: z.string(),
  text: z.string(),
  choices: z.array(Choice),
});
export type Question = z.infer<typeof Question>;

// ---- `jev`: the agent. Its reading of the user's answer, its reply and tool calls, and the next question (jev_decision.py, flow.py) ----
export const Jev = z.object({
  source: z.string().default("Jev"),
  available: z.boolean(),
  understood: z.object({
    question_id: z.string(),
    choice: z.string().nullable(),
    confidence: z.number().nullable(),
    method: z.string(), // "jev" | "fallback"
  }),
  reply: z.string(),
  tool_calls: z.array(z.unknown()),
  next_question: Question,
});
export type Jev = z.infer<typeof Jev>;

// ---- `google`: Gemini's separate run on the same input, with its own tool calls (main.py run_gemini) ----
export const Google = z.object({
  source: z.string().default("Google Gemini"),
  available: z.boolean().default(true),
  reply: z.string(),
  tool_calls: z.array(z.unknown()),
});
export type Google = z.infer<typeof Google>;

/** Opaque conversation state: echoed back to the backend on the next request. */
export const State = z.record(z.string(), z.unknown());
export type State = z.infer<typeof State>;

// ---- Full /chat response (main.py ChatResponse) ----
export const ChatResponse = z.object({
  google: Google.nullable().optional(),
  jev: Jev,
  state: State,
});
export type ChatResponse = z.infer<typeof ChatResponse>;

export type ToolResult =
  | { ok: true; call: ToolCall }
  | { ok: false; name: string; raw: unknown; issues: string };

/** Validate each tool call; one that does not match its schema is still shown, as raw JSON. */
export function parseTools(calls: unknown[]): ToolResult[] {
  return calls.map((raw): ToolResult => {
    const parsed = ToolCall.safeParse(raw);
    if (parsed.success) return { ok: true, call: parsed.data };
    const name = (raw as { name?: unknown } | null)?.name;
    return {
      ok: false,
      name: typeof name === "string" ? name : "unknown_tool",
      raw,
      issues: parsed.error.issues.map((i) => `${i.path.join(".") || "(root)"}: ${i.message}`).join("; "),
    };
  });
}
