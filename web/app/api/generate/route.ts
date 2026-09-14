import { NextResponse } from "next/server";

// ZeroGPU cold starts can take ~40s on first hit; allow generous headroom.
export const maxDuration = 120;
export const dynamic = "force-dynamic";

const SPACE_URL = process.env.HF_SPACE_URL?.replace(/\/$/, "");
const SPACE_TOKEN = process.env.HF_SPACE_TOKEN;

type SpacePayload = {
  sql: string;
  resultMarkdown: string;
  meta: {
    model?: string;
    backend?: string;
    generation_ms?: number;
    executed?: boolean;
    error?: string | null;
    columns?: string[];
    rows?: (string | number | boolean | null)[][];
    row_count?: number;
  };
};

function headers() {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (SPACE_TOKEN) h.Authorization = `Bearer ${SPACE_TOKEN}`;
  return h;
}

/**
 * Gradio's REST API is two calls: POST the inputs to get an event id, then GET
 * that event as an SSE stream. We read the stream to completion rather than
 * forwarding it, because the payload is small and the client wants one JSON.
 */
async function callSpace(data: unknown[], signal: AbortSignal): Promise<SpacePayload> {
  const post = await fetch(`${SPACE_URL}/gradio_api/call/generate`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ data }),
    signal,
  });

  if (!post.ok) {
    throw new Error(`Space rejected the request (${post.status}). ${await post.text().catch(() => "")}`.trim());
  }

  const { event_id: eventId } = (await post.json()) as { event_id?: string };
  if (!eventId) throw new Error("Space did not return an event id.");

  const stream = await fetch(`${SPACE_URL}/gradio_api/call/generate/${eventId}`, {
    headers: headers(),
    signal,
  });
  if (!stream.ok || !stream.body) {
    throw new Error(`Could not read the Space result stream (${stream.status}).`);
  }

  const reader = stream.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let event = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line; keep any partial tail.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (!line.startsWith("data:")) continue;

        const raw = line.slice(5).trim();
        if (event === "error") {
          throw new Error(raw && raw !== "null" ? raw : "The model Space reported an error.");
        }
        if (event !== "complete") continue;

        const parsed = JSON.parse(raw) as [string, string, SpacePayload["meta"]];
        reader.cancel().catch(() => {});
        return { sql: parsed[0] ?? "", resultMarkdown: parsed[1] ?? "", meta: parsed[2] ?? {} };
      }
    }
  }

  throw new Error("The Space closed the connection before returning a result.");
}

/**
 * Local development without a GPU: set MOCK_BACKEND=1 to answer with a canned
 * query so the UI can be built and reviewed before the model is trained.
 * Never enabled in production unless the variable is explicitly set.
 */
function mockResponse(schema: string): SpacePayload {
  const sql =
    "SELECT region, SUM(revenue) AS total_revenue\nFROM sales\nWHERE quarter = 'Q3'\nGROUP BY region\nORDER BY total_revenue DESC";
  return {
    sql,
    resultMarkdown: "",
    meta: {
      model: "MOCK -- no model was called",
      backend: "mock",
      generation_ms: 640,
      executed: true,
      error: null,
      columns: ["region", "total_revenue"],
      rows: [
        ["North", 70000],
        ["South", 41000],
        ["West", 14200],
      ],
      row_count: 3,
    },
  };
}

export async function POST(request: Request) {
  let body: { schema?: string; question?: string; maxTokens?: number; temperature?: number };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Request body must be JSON." }, { status: 400 });
  }

  const schema = (body.schema ?? "").trim();
  const question = (body.question ?? "").trim();
  if (!schema || !question) {
    return NextResponse.json({ error: "Both a schema and a question are required." }, { status: 400 });
  }
  if (schema.length > 20_000 || question.length > 2_000) {
    return NextResponse.json({ error: "Schema or question is too long." }, { status: 413 });
  }

  if (process.env.MOCK_BACKEND === "1") {
    await new Promise((r) => setTimeout(r, 500));
    return NextResponse.json(mockResponse(schema));
  }

  if (!SPACE_URL) {
    return NextResponse.json(
      {
        error:
          "No model backend configured. Set HF_SPACE_URL to your Hugging Face Space URL and redeploy.",
      },
      { status: 503 },
    );
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 115_000);

  try {
    const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n));
    const payload = await callSpace(
      [
        schema,
        question,
        clamp(Math.round(body.maxTokens ?? 320), 64, 512),
        clamp(body.temperature ?? 0, 0, 1),
      ],
      controller.signal,
    );
    return NextResponse.json(payload);
  } catch (err) {
    const aborted = controller.signal.aborted;
    return NextResponse.json(
      {
        error: aborted
          ? "The model backend took longer than 115 seconds. A cold Space can take a minute to wake up — try again."
          : err instanceof Error
            ? err.message
            : "Unexpected error talking to the model backend.",
      },
      { status: aborted ? 504 : 502 },
    );
  } finally {
    clearTimeout(timeout);
  }
}
