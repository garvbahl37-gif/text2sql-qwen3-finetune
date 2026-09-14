"""Free inference backend for the fine-tuned text-to-SQL Qwen3-4B.

Runs as a Hugging Face Space. Two execution modes, chosen by env var:

  USE_GGUF=0 (default)  ZeroGPU / any CUDA GPU, transformers, merged 16-bit weights
  USE_GGUF=1            free CPU tier, llama.cpp, q4_k_m GGUF (slower, unlimited)

Doubles as a standalone demo *and* as the JSON API the Vercel frontend calls
through `api_name="generate"`.
"""
from __future__ import annotations

import os
import re
import sqlite3
import time

import gradio as gr

MODEL_ID = os.getenv("MODEL_ID", "unsloth/Qwen3-4B-Instruct-2507")
GGUF_REPO = os.getenv("GGUF_REPO", f"{MODEL_ID}-gguf")
GGUF_FILE = os.getenv("GGUF_FILE", "*q4_k_m.gguf")
# A local .gguf path, which skips the Hub download entirely. Used when running
# this server on your own machine instead of on a Space.
GGUF_PATH = os.getenv("GGUF_PATH", "")
USE_GGUF = os.getenv("USE_GGUF", "0") == "1"
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "320"))

SYSTEM_PROMPT = (
    "You are a precise text-to-SQL engine. Given a SQLite schema and a question, "
    "reply with a single valid SQLite query that answers it. "
    "Output only the SQL. No explanation, no markdown, no commentary."
)
USER_TEMPLATE = "### Schema\n{schema}\n\n### Question\n{question}"

# Shown to the model when its first query fails to run. Feeding the database's
# own error back is far more effective than asking for "valid SQL" up front:
# the model cannot know it misused a window function until SQLite says so.
REPAIR_TEMPLATE = """### Schema
{schema}

### Question
{question}

### Your previous answer
{sql}

### It failed with this SQLite error
{error}

Rewrite the query so it runs. Use only tables and columns that appear in the
schema above.{hint}

Output only the corrected SQL, and make it different from the previous answer."""

# SQLite-specific guidance for the error classes this model actually hits.
# General SQL advice, not a patch for one query: window functions cannot appear
# in WHERE or HAVING, and aliases are not visible there either.
REPAIR_HINTS = (
    ("window function", "\nSQLite does not allow window functions in WHERE or HAVING. "
                        "Compute the window expression in a subquery or CTE, then filter "
                        "the outer query on its result."),
    ("no such column", "\nOne of the columns does not exist. Re-read the CREATE TABLE "
                       "statements and use only the column names written there."),
    ("no such table", "\nOne of the tables does not exist. Use only the tables in the "
                      "schema above."),
    ("ambiguous column", "\nQualify every column with its table alias."),
)


# ---------------------------------------------------------------------------
# Structural lint
#
# The repair pass above only fires on a SQLite error. These patterns produce
# SQL that runs fine and answers the wrong question, which is worse, because
# nothing downstream notices. Each rule below was written against a real
# failure, and each is narrow enough to rarely fire on a correct query.
# ---------------------------------------------------------------------------
TOP_N = re.compile(r"\b(?:top|highest|best|largest|bottom|lowest)\s+(\d+|two|three|five|ten)\b", re.I)
RANK_FN = re.compile(r"\b(?:RANK|DENSE_RANK|ROW_NUMBER)\s*\(\s*\)\s*OVER", re.I)
RANK_FILTERED = re.compile(r"\bWHERE\b[^;]*\b(?:rnk|rank|rn|row_num|position)\b|\bQUALIFY\b|\bLIMIT\b", re.I)
WINDOW_FN = re.compile(r"\bOVER\s*\(", re.I)
GROUP_BY = re.compile(r"\bGROUP\s+BY\b", re.I)
WITH_CTE = re.compile(r"^\s*WITH\b", re.I)
SHARE_WORDS = re.compile(r"\b(percentage|percent|share|proportion|fraction|ratio of)\b", re.I)
RESTRICT_WORDS = re.compile(r"\b(exclude|excluding|only|completed|cancelled|canceled|active|pending|shipped|delivered|returned)\b", re.I)
HAS_WHERE = re.compile(r"\bWHERE\b", re.I)


# Telling this model what to do rarely works after narrow fine-tuning; showing
# it the shape to copy works better. This skeleton is the standard two-stage
# pattern for per-group shares and top-N, which is what the lint rules detect.
CTE_SKELETON = """
Use this exact shape, substituting the real tables, columns and filters:

WITH agg AS (
  SELECT <group_cols>, SUM(<measure>) AS metric, COUNT(DISTINCT <id>) AS uniques
  FROM <tables with joins>
  WHERE <row filters from the question>
  GROUP BY <group_cols>
),
ranked AS (
  SELECT *,
         metric * 100.0 / SUM(metric) OVER (PARTITION BY <outer_group>) AS pct,
         RANK() OVER (PARTITION BY <outer_group> ORDER BY metric DESC) AS rnk
  FROM agg
)
SELECT * FROM ranked WHERE rnk <= <n> ORDER BY <outer_group>, rnk;
"""


def lint_sql(question: str, sql: str) -> list[str]:
    """Structural problems in SQL that runs but answers the wrong question."""
    issues = []

    if TOP_N.search(question) and RANK_FN.search(sql) and not RANK_FILTERED.search(sql):
        issues.append(
            "The question asks for a top-N within each group, and the query computes a "
            "rank but never filters on it, so every row is returned. Put the ranked "
            "query in a CTE and filter the rank in an outer SELECT."
        )

    # Share-of-group-total needs the window applied to an already-aggregated CTE.
    # Windowing in the same SELECT as the GROUP BY aggregates windows over the
    # pre-aggregation rows, which is how percentages above 100 appear.
    if (SHARE_WORDS.search(question) and WINDOW_FN.search(sql)
            and GROUP_BY.search(sql) and not WITH_CTE.search(sql)):
        issues.append(
            "A share of a group total cannot be computed with a window function in the "
            "same SELECT as the GROUP BY aggregates -- it windows over the rows before "
            "grouping and can exceed 100%. Aggregate in a CTE first, then apply the "
            "window function to that CTE in an outer SELECT."
        )

    if RESTRICT_WORDS.search(question) and not HAS_WHERE.search(sql):
        issues.append(
            "The question restricts which rows should count, but the query has no WHERE "
            "clause, so every row is included."
        )

    return issues


def repair_hint(error: str) -> str:
    low = (error or "").lower()
    for needle, hint in REPAIR_HINTS:
        if needle in low:
            return hint
    return ""

# ZeroGPU decorator when running on a Space that has it; a no-op otherwise so
# the same file runs on a plain GPU box or locally.
try:
    import spaces

    gpu_task = spaces.GPU(duration=60)
except Exception:  # not a ZeroGPU Space
    def gpu_task(fn):
        return fn


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
if USE_GGUF:
    from llama_cpp import Llama

    _n_threads = int(os.getenv("N_THREADS", "2"))
    if GGUF_PATH:
        print(f"[init] llama.cpp from local file: {GGUF_PATH} ({_n_threads} threads)")
        _llm = Llama(model_path=GGUF_PATH, n_ctx=4096, n_threads=_n_threads, verbose=False)
    else:
        print(f"[init] llama.cpp CPU mode: {GGUF_REPO} :: {GGUF_FILE} ({_n_threads} threads)")
        _llm = Llama.from_pretrained(
            repo_id=GGUF_REPO,
            filename=GGUF_FILE,
            n_ctx=4096,
            n_threads=_n_threads,
            verbose=False,
        )

    def _complete_raw(user_content: str, max_new_tokens: int, temperature: float) -> str:
        out = _llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return out["choices"][0]["message"]["content"]

else:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[init] transformers mode: {MODEL_ID}")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available() or "SPACES_ZERO_GPU" in os.environ:
        # ZeroGPU intercepts this at import time and attaches a GPU per request.
        _model = _model.to("cuda")
    _model.eval()

    @gpu_task
    def _complete_raw(user_content: str, max_new_tokens: int, temperature: float) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = _tokenizer(text, return_tensors="pt").to(_model.device)
        with torch.inference_mode():
            gen = _model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
            )
        return _tokenizer.decode(gen[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)


def _complete(schema: str, question: str, max_new_tokens: int, temperature: float) -> str:
    """The normal first-attempt prompt."""
    return _complete_raw(
        USER_TEMPLATE.format(schema=schema, question=question), max_new_tokens, temperature
    )


# --------------------------------------------------------------------------
# SQL post-processing + execution
# --------------------------------------------------------------------------
FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
SQL_KEYWORD = r"(?:WITH|SELECT|INSERT|UPDATE|DELETE)"
LINE_START = re.compile(rf"^\s*{SQL_KEYWORD}\b", re.IGNORECASE | re.MULTILINE)
ANYWHERE = re.compile(rf"\b{SQL_KEYWORD}\b", re.IGNORECASE)

# Only reads are executed here. The schema comes from untrusted input, so the
# DB is in-memory and thrown away, but refusing writes keeps intent obvious.
WRITE_STATEMENT = re.compile(r"^\s*(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE)\b", re.IGNORECASE)
# DROP/ALTER/TRUNCATE never survive extract_sql, so check the raw completion for
# them separately -- otherwise they'd be reported as "no SQL", which is a lie.
DESTRUCTIVE = re.compile(r"\b(DROP|ALTER|TRUNCATE|ATTACH|PRAGMA)\b", re.IGNORECASE)


def extract_sql(text: str) -> str:
    if m := FENCE.search(text):
        text = m.group(1)
    if m := LINE_START.search(text):
        text = text[m.start() :]
    else:
        matches = list(ANYWHERE.finditer(text))
        if not matches:
            return ""
        text = text[matches[-1].start() :]
    return text.split(";")[0].strip()


def _jsonable(v):
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    return str(v)


def execute(schema: str, sql: str) -> tuple[list[str], list[list], str | None]:
    """Run the query against a throwaway in-memory DB built from the schema."""
    if not sql:
        return [], [], "model produced no SQL"
    if WRITE_STATEMENT.match(sql):
        return [], [], "only read queries are executed in this demo"
    try:
        con = sqlite3.connect(":memory:")
        con.executescript(schema)
    except Exception as exc:
        return [], [], f"schema error: {exc}"
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [[_jsonable(v) for v in r] for r in cur.fetchmany(200)]
        return cols, rows, None
    except Exception as exc:
        return [], [], f"query error: {exc}"
    finally:
        con.close()


def generate(schema: str, question: str, max_new_tokens: int = MAX_NEW_TOKENS,
             temperature: float = 0.0, repair: bool = True):
    """Main entrypoint. Exposed to the frontend as api_name='generate'."""
    schema, question = (schema or "").strip(), (question or "").strip()
    if not schema or not question:
        return "", "", {"error": "schema and question are both required"}

    t0 = time.time()
    raw = _complete(schema, question, int(max_new_tokens), float(temperature))

    sql = extract_sql(raw)
    if not sql and DESTRUCTIVE.search(raw):
        sql, cols, rows, err = raw.strip(), [], [], "only read queries are executed in this demo"
    else:
        cols, rows, err = execute(schema, sql)

    # One repair attempt. The model writes elaborate queries for hard questions
    # and sometimes trips over a dialect rule -- SQLite rejects window functions
    # in HAVING, for instance. Handing back the error lets it correct itself.
    first_sql, first_error, repaired = sql, err, False
    repair_attempted, repair_sql, repair_error = False, None, None
    if repair and err and not err.startswith("only read"):
        repair_attempted = True
        prompt = REPAIR_TEMPLATE.format(schema=schema, question=question, sql=sql,
                                        error=err, hint=repair_hint(err))
        # Greedy decoding reproduces the same wrong query no matter what the
        # prompt says, so repairs sample instead. Two tries with rising
        # temperature: the first stays close to the original, the second is
        # free to restructure.
        for attempt, temp in enumerate((0.5, 0.9), start=1):
            raw2 = _complete_raw(prompt, int(max_new_tokens), temp)
            sql2 = extract_sql(raw2)
            repair_sql = sql2 or repair_sql
            if not sql2:
                repair_error = "the repair attempt produced no SQL"
                continue
            if sql2 == sql:
                repair_error = "the model returned the same query unchanged"
                continue
            cols2, rows2, err2 = execute(schema, sql2)
            if err2 is None:
                sql, cols, rows, err, repaired = sql2, cols2, rows2, None, True
                repair_error = None
                break
            repair_error = err2

    # A query that runs can still answer the wrong question. Lint it, and if it
    # trips a known-bad pattern, ask the model to rewrite with that specific
    # guidance -- keeping the original unless the rewrite runs and lints clean.
    lint_issues = lint_sql(question, sql) if (repair and err is None) else []
    if lint_issues:
        needs_cte = any("CTE" in i for i in lint_issues)
        prompt = REPAIR_TEMPLATE.format(
            schema=schema, question=question, sql=sql,
            error="The query runs, but: " + " ".join(lint_issues),
            hint=CTE_SKELETON if needs_cte else "")
        for temp in (0.5, 0.9):
            raw3 = _complete_raw(prompt, int(max_new_tokens), temp)
            sql3 = extract_sql(raw3)
            if not sql3 or sql3 == sql:
                continue
            cols3, rows3, err3 = execute(schema, sql3)
            if err3 is None and not lint_sql(question, sql3):
                first_sql, first_error = sql, "structural: " + " ".join(lint_issues)
                sql, cols, rows, repaired = sql3, cols3, rows3, True
                lint_issues = []
                break

    gen_ms = int((time.time() - t0) * 1000)

    if err:
        table_md = f"_{err}_"
    elif not rows:
        table_md = "_query ran successfully and returned no rows_"
    else:
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        divider = "| " + " | ".join("---" for _ in cols) + " |"
        body = "\n".join("| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows[:50])
        table_md = "\n".join([header, divider, body])

    meta = {
        "model": MODEL_ID,
        "backend": "llama.cpp-cpu" if USE_GGUF else "transformers-gpu",
        "generation_ms": gen_ms,
        "executed": err is None,
        "error": err,
        "repaired": repaired,
        "repair_attempted": repair_attempted,
        # Kept even when the repair fails -- "attempted and still wrong" and
        # "never attempted" are different bugs and must not look identical.
        "repair_sql": repair_sql,
        "repair_error": repair_error,
        "lint_issues": lint_issues,
        "first_attempt_sql": first_sql if repair_attempted else None,
        "first_attempt_error": first_error if repair_attempted else None,
        "columns": cols,
        "rows": rows[:200],
        "row_count": len(rows),
    }
    return sql, table_md, meta


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
EXAMPLE_SCHEMA = """CREATE TABLE sales (
  id INTEGER PRIMARY KEY,
  region TEXT,
  quarter TEXT,
  revenue INTEGER
);
INSERT INTO sales VALUES
  (1,'North','Q3',52000),(2,'South','Q3',41000),
  (3,'North','Q4',61000),(4,'South','Q4',38000);"""

with gr.Blocks(title="Qwen3-4B Text-to-SQL", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"## Qwen3-4B fine-tuned for text-to-SQL\n"
        f"`{MODEL_ID}` — generates a SQLite query and executes it against the schema you provide."
    )
    with gr.Row():
        with gr.Column():
            schema_in = gr.Code(value=EXAMPLE_SCHEMA, language="sql", label="Schema (CREATE + INSERT)", lines=14)
            question_in = gr.Textbox(value="Total revenue per region in Q3", label="Question")
            with gr.Row():
                tokens_in = gr.Slider(64, 512, value=MAX_NEW_TOKENS, step=32, label="Max new tokens")
                temp_in = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature (0 = greedy)")
            go = gr.Button("Generate SQL", variant="primary")
        with gr.Column():
            sql_out = gr.Code(label="Generated SQL", language="sql")
            table_out = gr.Markdown(label="Result")
            meta_out = gr.JSON(label="Metadata")

    go.click(
        generate,
        inputs=[schema_in, question_in, tokens_in, temp_in],
        outputs=[sql_out, table_out, meta_out],
        api_name="generate",
    )

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=7860)
