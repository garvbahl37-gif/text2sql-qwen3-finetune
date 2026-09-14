import type { EvalReport } from "@/lib/types";

const HARDWARE: Record<string, string> = {
  "mlx-apple-silicon":
    "One Apple Silicon Mac, trained locally with MLX. Measured at 3.2 examples/sec on an M5 Pro (24GB unified memory), about 21 minutes for a full epoch. No rented GPU.",
  "cuda-unsloth":
    "One 48GB GPU (A40 / L40S / A6000) on RunPod, roughly 45 minutes of wall clock.",
};

function spec(report: EvalReport): [string, React.ReactNode][] {
  const trained = report.train_examples;
  return [
  ["Base model", <code className="mono" key="m">Qwen/Qwen3-4B-Instruct-2507</code>],
  [
    "Dataset",
    <>
      <code className="mono">gretelai/synthetic_text_to_sql</code>
      {trained ? ` — ${trained.toLocaleString()} training examples, ` : " — every training example "}
      verified to execute against a real SQLite database before being used. 21% of the raw dataset
      has reference SQL that does not run; those examples are dropped rather than trained on.
    </>,
  ],
  [
    "Method",
    "LoRA against a 4-bit base. Rank 32 adapters on every attention and MLP projection in all 36 blocks, one epoch, cosine schedule at 2e-4, effective batch size 16, 640-token context — the measured maximum in this data is 581.",
  ],
  [
    "Loss masking",
    "Gradients flow only through the assistant turn, so the model is graded on the SQL it writes rather than on re-predicting the schema it was handed. Worth several points on its own.",
  ],
  ["Hardware", HARDWARE[report.backend ?? ""] ?? HARDWARE["cuda-unsloth"]],
  [
    "Serving",
    "Adapter fused into the full-precision base — not a dequantised copy, so no 4-bit error is baked in — published to the Hugging Face Hub and served from a free Hugging Face Space. This page is a Next.js app on Vercel that proxies to it, so no token ever reaches the browser.",
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
}

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
          {spec(report).map(([k, v]) => (
            <tr key={k}>
              <th scope="row">{k}</th>
              <td>{v}</td>
            </tr>
          ))}
          <tr>
            <th scope="row">Reproduce it</th>
            <td>
              <code className="mono">
                {report.backend === "cuda-unsloth"
                  ? "bash training/runpod_bootstrap.sh"
                  : "bash training/mac_pipeline.sh"}
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
