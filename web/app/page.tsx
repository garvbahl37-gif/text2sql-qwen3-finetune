"use client";

import { useState } from "react";
import Benchmark from "@/components/Benchmark";
import Method from "@/components/Method";
import Playground from "@/components/Playground";
import report from "@/data/eval_report.json";
import type { EvalReport } from "@/lib/types";

const TABS = [
  { id: "demo", label: "Try it" },
  { id: "benchmark", label: "Benchmark" },
  { id: "method", label: "Method" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function Page() {
  const [tab, setTab] = useState<TabId>("demo");
  const evalReport = report as EvalReport;
  const measured = !evalReport.placeholder && evalReport.metrics;

  return (
    <main className="shell">
      <div className="topbar">
        <div className="wordmark">
          <b>qwen3-4b-text2sql</b> <span>/ LoRA fine-tune</span>
        </div>
        <nav className="tabs" role="tablist" aria-label="Sections">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              className="tab"
              aria-selected={tab === t.id}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </div>

      {tab === "demo" && (
        <>
          <header className="hero">
            <h1>Ask in English. Get SQL that runs.</h1>
            <div>
            <p>
              A 4-billion-parameter Qwen3 model, fine-tuned on{" "}
              {evalReport.train_examples
                ? `${evalReport.train_examples.toLocaleString()} verified queries`
                : "verified queries"}{" "}
              to turn a question and a schema into SQLite that executes.
            </p>
            <p>
              {measured && evalReport.metrics ? (
                <>
                  Execution accuracy on held-out tests went from{" "}
                  <span className="ink">
                    {(evalReport.metrics.base.execution_accuracy * 100).toFixed(1)}%
                  </span>{" "}
                  to{" "}
                  <span className="ink">
                    {(evalReport.metrics.tuned.execution_accuracy * 100).toFixed(1)}%
                  </span>
                  . Every query below is run against a real database, not just printed.
                </>
              ) : (
                <>
                  Every query below is run against a real database built from the schema, not just
                  printed.
                </>
              )}
            </p>
            </div>
          </header>
          <Playground />
        </>
      )}

      {tab === "benchmark" && <Benchmark report={evalReport} />}
      {tab === "method" && <Method report={evalReport} />}

      <footer className="foot">
        Fine-tuned with MLX on an Apple M5 Pro. Frontend on Vercel.
      </footer>
    </main>
  );
}
