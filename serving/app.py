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
