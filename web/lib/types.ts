export type Metrics = {
  execution_accuracy: number;
  valid_sql_rate: number;
  exact_match: number;
};

export type Sample = {
  question: string;
  domain?: string;
  complexity?: string;
  gold_sql: string;
  base_sql: string;
  tuned_sql: string;
  base_exec_ok: boolean;
  tuned_exec_ok: boolean;
};

export type EvalReport = {
  placeholder?: boolean;
  base_model: string;
  tuned_model: string;
  dataset: string;
  n_examples: number;
  decoding: string;
  backend?: string;
  train_examples?: number | null;
  metrics: { base: Metrics; tuned: Metrics } | null;
  deltas?: Partial<Metrics>;
  by_complexity?: Record<string, { n: number; base: number; tuned: number }>;
  counts?: {
    tuned_win: number;
    both_correct: number;
    tuned_regression: number;
    both_wrong: number;
  };
  samples: Sample[];
};

export type GenerateResponse = {
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
    repaired?: boolean;
    lint_issues?: string[];
    first_attempt_sql?: string | null;
    first_attempt_error?: string | null;
  };
};
