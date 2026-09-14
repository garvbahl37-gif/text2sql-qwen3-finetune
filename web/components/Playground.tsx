"use client";

import { useEffect, useRef, useState } from "react";
import { PRESETS } from "@/lib/schemas";
import type { GenerateResponse } from "@/lib/types";

type Status = "idle" | "running" | "done" | "error";

/** Reveals the generated SQL character by character — the one motion moment. */
function useTypewriter(text: string, active: boolean) {
  const [shown, setShown] = useState(text);

  useEffect(() => {
    if (!active) {
      setShown(text);
      return;
    }
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced || !text) {
      setShown(text);
      return;
    }
    setShown("");
    let i = 0;
    const step = Math.max(1, Math.ceil(text.length / 90));
    const timer = setInterval(() => {
      i += step;
      setShown(text.slice(0, i));
      if (i >= text.length) clearInterval(timer);
    }, 16);
    return () => clearInterval(timer);
  }, [text, active]);

  return shown;
}

export default function Playground() {
  const [presetId, setPresetId] = useState(PRESETS[0].id);
  const [schema, setSchema] = useState(PRESETS[0].schema);
  const [question, setQuestion] = useState(PRESETS[0].questions[0]);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [error, setError] = useState("");
  const [slow, setSlow] = useState(false);
  const slowTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const preset = PRESETS.find((p) => p.id === presetId) ?? PRESETS[0];
  const typed = useTypewriter(result?.sql ?? "", status === "done");

  function choosePreset(id: string) {
    const next = PRESETS.find((p) => p.id === id);
    if (!next) return;
    setPresetId(id);
    setSchema(next.schema);
    setQuestion(next.questions[0]);
    setResult(null);
    setStatus("idle");
    setError("");
  }

  async function run() {
    setStatus("running");
    setError("");
    setResult(null);
    setSlow(false);
    slowTimer.current = setTimeout(() => setSlow(true), 8000);

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ schema, question }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body?.error ?? `Request failed (${res.status}).`);
      setResult(body as GenerateResponse);
      setStatus("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setStatus("error");
    } finally {
      if (slowTimer.current) clearTimeout(slowTimer.current);
      setSlow(false);
    }
  }

  const meta = result?.meta;
  const columns = meta?.columns ?? [];
  const rows = meta?.rows ?? [];
  const canRun = schema.trim().length > 0 && question.trim().length > 0 && status !== "running";

  return (
    <div className="console">
      <section className="panel">
        <header className="panel-head">
          <span className="filename">schema.sql</span>
          <span>{preset.note}</span>
        </header>
        <div className="panel-body grow">
          <textarea
            className="field"
            value={schema}
            spellCheck={false}
            onChange={(e) => setSchema(e.target.value)}
            aria-label="Database schema"
          />
          <div className="presets">
            {PRESETS.map((p) => (
              <button
                key={p.id}
                type="button"
                className="chip"
                aria-pressed={p.id === presetId}
                onClick={() => choosePreset(p.id)}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="panel">
        <header className="panel-head">
          <span className="filename">query.sql</span>
          <span>{meta?.backend ?? "not run yet"}</span>
        </header>

        <div className="panel-body">
          <input
            className="field"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about this data"
            aria-label="Question"
            onKeyDown={(e) => {
              if (e.key === "Enter" && canRun) run();
            }}
          />
          <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 10 }}>
            {preset.questions.map((q) => (
              <button key={q} type="button" className="suggest" onClick={() => setQuestion(q)}>
                {q}
              </button>
            ))}
          </div>
          <div className="row">
            <button type="button" className="run" onClick={run} disabled={!canRun}>
              {status === "running" ? "Generating" : "Generate SQL"}
            </button>
            {status === "done" && meta?.generation_ms != null && (
              <span className="mono" style={{ fontSize: 12, color: "var(--muted)" }}>
                {meta.generation_ms} ms
              </span>
            )}
          </div>
        </div>

        <pre className={`sql-out${result?.sql ? "" : " empty"}`}>
          {status === "running" ? (
            <>
              waiting for the model
              <span className="caret" />
            </>
          ) : result?.sql ? (
            typed
          ) : status === "error" ? (
            "—"
          ) : (
            "The generated query will appear here."
          )}
        </pre>

        {status === "running" && slow && (
          <p className="notice">
            Waking the model backend. A Space that has been idle takes about a minute to come back;
            later requests return in a few seconds.
          </p>
        )}

        {status === "error" && <p className="notice bad">{error}</p>}

        {status === "done" && meta && !meta.executed && (
          <p className="notice bad">{meta.error ?? "The query did not run."}</p>
        )}

        {status === "done" && meta?.executed && columns.length > 0 && (
          <>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {columns.map((c, i) => (
                      <th key={`${c}-${i}`}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.slice(0, 25).map((r, i) => (
                    <tr key={i}>
                      {r.map((v, j) => (
                        <td key={j}>{v === null ? "null" : String(v)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="notice">
              {meta.row_count} {meta.row_count === 1 ? "row" : "rows"} returned
              {(meta.row_count ?? 0) > 25 ? ", showing the first 25" : ""}, executed against a
              throwaway in-memory SQLite database built from your schema.
            </p>
          </>
        )}

        {status === "done" && meta?.executed && columns.length === 0 && (
          <p className="notice">The query ran and returned no rows.</p>
        )}
      </section>
    </div>
  );
}
