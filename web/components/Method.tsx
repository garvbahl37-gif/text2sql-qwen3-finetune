import type { EvalReport } from "@/lib/types";

const SPEC: [string, React.ReactNode][] = [
  ["Base model", <code className="mono" key="m">Qwen/Qwen3-4B-Instruct-2507</code>],
  [
    "Dataset",
    <>
      <code className="mono">gretelai/synthetic_text_to_sql</code> — 8,000 training examples, every
      one verified to execute against a real SQLite database before it is used. Examples whose
      reference query fails are dropped rather than trained on.
    </>,
  ],
  [
    "Method",
    "4-bit QLoRA through Unsloth. Rank 32 adapters on all attention and MLP projections, one epoch, cosine schedule at 2e-4, effective batch size 16, 2048-token context.",
  ],
  [
    "Loss masking",
    "Gradients flow only through the assistant turn, so the model is graded on the SQL it writes rather than on re-predicting the schema it was handed. Worth several points on its own.",
  ],
  ["Hardware", "One 48GB GPU (A40 / L40S / A6000) on RunPod, roughly 45 minutes of wall clock."],
  [
    "Serving",
    "Adapter merged to 16-bit, published to the Hugging Face Hub, served from a free Hugging Face Space. This page is a Next.js app on Vercel that proxies to it, so no token ever reaches the browser.",
  ],
  [
    "Scoring",
    "Execution accuracy: build the database from the example's own schema, run both the generated and the reference query, compare result sets. Order only matters when the reference query asked for an order.",
  ],
  [
    "Fairness",
    "Both models get the identical prompt and greedy decoding. The SQL extractor is deliberately lenient about markdown fences and preamble, because the base model wraps its answers in prose and penalising formatting would inflate the result.",
  ],
];

export default function Method({ report }: { report: EvalReport }) {
  return (
    <div className="section">
      <h2>How it was built</h2>
      <p className="lede">
        The whole pipeline is in this repository: data preparation, training, evaluation, the
        inference backend and this frontend.
      </p>
      <table className="spec">
        <tbody>
          {SPEC.map(([k, v]) => (
            <tr key={k}>
              <th scope="row">{k}</th>
              <td>{v}</td>
            </tr>
          ))}
          <tr>
            <th scope="row">Reproduce it</th>
            <td>
              <code className="mono">
                export HF_TOKEN=... HF_REPO=you/qwen3-4b-text2sql && bash
                training/runpod_bootstrap.sh
              </code>
            </td>
          </tr>
        </tbody>
      </table>
      {!report.placeholder && (
        <p className="lede" style={{ marginTop: 24 }}>
          Weights: <code className="mono">{report.tuned_model}</code>
        </p>
      )}
    </div>
  );
}
