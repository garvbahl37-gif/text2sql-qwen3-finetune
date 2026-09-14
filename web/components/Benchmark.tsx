import type { EvalReport, Metrics } from "@/lib/types";

const METRIC_COPY: Record<keyof Metrics, { title: string; desc: string }> = {
  execution_accuracy: {
    title: "Execution accuracy",
    desc: "The generated query runs against the real database and returns the same rows as the reference query. This is the number that matters — a correct query written differently still counts.",
  },
  valid_sql_rate: {
    title: "Valid SQL rate",
    desc: "The query parses and executes without an error, whether or not the answer is right.",
  },
  exact_match: {
    title: "Exact string match",
    desc: "The query is character-identical to the reference after whitespace normalisation. Reported for completeness; it undercounts correct answers.",
  },
};

function pct(n: number) {
  return `${(n * 100).toFixed(1)}%`;
}

function Bars({ label, base, tuned }: { label: keyof Metrics; base: number; tuned: number }) {
  const delta = tuned - base;
  return (
    <div className="metric">
      <h3>{METRIC_COPY[label].title}</h3>
      <p className="desc">{METRIC_COPY[label].desc}</p>
      <div className="bar-row">
        <span className="bar-label">base</span>
        <div className="bar-track">
          <div className="bar-fill base" style={{ width: `${Math.max(base * 100, 0.5)}%` }} />
        </div>
        <span className="bar-value">{pct(base)}</span>
      </div>
      <div className="bar-row">
        <span className="bar-label">fine-tuned</span>
        <div className="bar-track">
          <div className="bar-fill tuned" style={{ width: `${Math.max(tuned * 100, 0.5)}%` }} />
        </div>
        <span className={`bar-value${delta > 0 ? " up" : ""}`}>{pct(tuned)}</span>
      </div>
      <p className="desc" style={{ margin: "10px 0 0" }}>
        {delta >= 0 ? "+" : ""}
        {(delta * 100).toFixed(1)} points
      </p>
    </div>
  );
}

export default function Benchmark({ report }: { report: EvalReport }) {
  if (report.placeholder || !report.metrics) {
    return (
      <div className="section">
        <h2>Benchmark</h2>
        <p className="lede">
          Base Qwen3-4B against the fine-tuned adapter on held-out examples the model never saw
          during training.
        </p>
        <div className="banner">
          <strong>Not measured yet.</strong> Train and evaluate, then copy{" "}
          <code className="mono">training/outputs/eval_report.json</code> to{" "}
          <code className="mono">web/data/eval_report.json</code>. Nothing on this page is
          simulated, so it stays empty until there are real numbers to show.
        </div>
      </div>
    );
  }

  const { base, tuned } = report.metrics;
  const complexity = Object.entries(report.by_complexity ?? {});

  return (
    <div className="section">
      <h2>Benchmark</h2>
      <p className="lede">
        {report.n_examples} held-out examples from <code className="mono">{report.dataset}</code>,
        greedy decoding, scored by executing every query against the database its schema describes.
        Examples whose reference query fails to run are excluded so neither model is penalised for a
        broken gold answer.
      </p>

      {(Object.keys(METRIC_COPY) as (keyof Metrics)[]).map((key) => (
        <Bars key={key} label={key} base={base[key]} tuned={tuned[key]} />
      ))}

      {report.counts && (
        <p className="desc" style={{ marginTop: 18, color: "var(--muted)", fontSize: "var(--step--1)" }}>
          Fine-tuned right where base was wrong: {report.counts.tuned_win}. Both right:{" "}
          {report.counts.both_correct}. Fine-tuned wrong where base was right:{" "}
          {report.counts.tuned_regression}. Both wrong: {report.counts.both_wrong}.
        </p>
      )}

      {complexity.length > 0 && (
        <>
          <h2 style={{ marginTop: 44 }}>By query complexity</h2>
          <p className="lede">Execution accuracy split by the difficulty label in the dataset.</p>
          <div className="table-wrap" style={{ borderTop: "1px solid var(--rule)" }}>
            <table>
              <thead>
                <tr>
                  <th>complexity</th>
                  <th>n</th>
                  <th>base</th>
                  <th>fine-tuned</th>
                  <th>change</th>
                </tr>
              </thead>
              <tbody>
                {complexity.map(([name, v]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{v.n}</td>
                    <td>{pct(v.base)}</td>
                    <td>{pct(v.tuned)}</td>
                    <td style={{ color: v.tuned > v.base ? "var(--cobalt)" : "var(--muted)" }}>
                      {v.tuned - v.base >= 0 ? "+" : ""}
                      {((v.tuned - v.base) * 100).toFixed(1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {report.samples.length > 0 && (
        <>
          <h2 style={{ marginTop: 44 }}>Side by side</h2>
          <p className="lede">
            Real outputs from the scored run, including cases the fine-tuned model still gets wrong.
          </p>
          <div className="samples">
            {report.samples.map((s, i) => (
              <article className="sample" key={i}>
                <p className="sample-q">{s.question}</p>
                <div className="sample-grid">
                  <div className="sample-col">
                    <div className="who">
                      <span>base</span>
                      <span className={`mark ${s.base_exec_ok ? "pass" : "fail"}`}>
                        {s.base_exec_ok ? "correct" : "wrong"}
                      </span>
                    </div>
                    <pre>{s.base_sql || "(no query produced)"}</pre>
                  </div>
                  <div className="sample-col is-tuned">
                    <div className="who">
                      <span>fine-tuned</span>
                      <span className={`mark ${s.tuned_exec_ok ? "pass" : "fail"}`}>
                        {s.tuned_exec_ok ? "correct" : "wrong"}
                      </span>
                    </div>
                    <pre>{s.tuned_sql || "(no query produced)"}</pre>
                  </div>
                  <div className="sample-col">
                    <div className="who">
                      <span>reference</span>
                    </div>
                    <pre>{s.gold_sql}</pre>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
