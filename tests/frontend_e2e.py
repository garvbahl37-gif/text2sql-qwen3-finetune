"""End-to-end accuracy check through the deployed frontend.

Sends real questions to the /api/generate endpoint the browser uses, executes
the SQL the model returns, and compares the result set against a reference
query written by hand. This tests the whole chain -- frontend route, Space,
model, SQL execution -- not just the model.

    python tests/frontend_e2e.py --base-url http://localhost:3000
    python tests/frontend_e2e.py --base-url https://your-app.vercel.app
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from sqlutil import build_db, results_match, run_sql  # noqa: E402

RETAIL = """CREATE TABLE sales (
  id INTEGER PRIMARY KEY, region TEXT, quarter TEXT,
  product TEXT, units INTEGER, revenue INTEGER);
INSERT INTO sales VALUES
  (1,'North','Q3','Widget',120,52000),(2,'South','Q3','Widget',95,41000),
  (3,'North','Q3','Gadget',40,18000),(4,'North','Q4','Widget',140,61000),
  (5,'South','Q4','Gadget',88,38000),(6,'West','Q4','Widget',61,27500),
  (7,'West','Q3','Gadget',33,14200);"""

LIBRARY = """CREATE TABLE authors (author_id INTEGER PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE books (book_id INTEGER PRIMARY KEY, title TEXT, author_id INTEGER, year INTEGER, genre TEXT);
CREATE TABLE loans (loan_id INTEGER PRIMARY KEY, book_id INTEGER, member TEXT, loaned_on TEXT, returned INTEGER);
INSERT INTO authors VALUES (1,'Ursula K. Le Guin','USA'),(2,'Italo Calvino','Italy'),(3,'Chinua Achebe','Nigeria'),(4,'Han Kang','South Korea');
INSERT INTO books VALUES (1,'The Dispossessed',1,1974,'sci-fi'),(2,'A Wizard of Earthsea',1,1968,'fantasy'),
  (3,'Invisible Cities',2,1972,'fiction'),(4,'Things Fall Apart',3,1958,'fiction'),(5,'The Vegetarian',4,2007,'fiction');
INSERT INTO loans VALUES (1,1,'ana','2026-01-04',1),(2,1,'ben','2026-02-11',0),(3,3,'ana','2026-02-20',1),
  (4,4,'cleo','2026-03-02',0),(5,5,'ben','2026-03-09',1),(6,5,'dev','2026-03-15',0);"""

CLINIC = """CREATE TABLE patients (patient_id INTEGER PRIMARY KEY, name TEXT, birth_year INTEGER, city TEXT);
CREATE TABLE visits (visit_id INTEGER PRIMARY KEY, patient_id INTEGER, visit_date TEXT, department TEXT, cost REAL);
INSERT INTO patients VALUES (1,'Rivera',1978,'Austin'),(2,'Okonkwo',1991,'Austin'),(3,'Lindqvist',1965,'Dallas'),
  (4,'Haddad',2001,'Dallas'),(5,'Moreau',1985,'Houston');
INSERT INTO visits VALUES (1,1,'2026-01-12','cardiology',420.00),(2,1,'2026-03-02','cardiology',380.50),
  (3,2,'2026-01-28','dermatology',150.00),(4,3,'2026-02-14','cardiology',610.25),
  (5,4,'2026-02-20','orthopedics',295.75),(6,5,'2026-03-11','dermatology',185.00),
  (7,2,'2026-03-19','orthopedics',340.00);"""

# (schema, question, reference SQL, difficulty)
CASES = [
    (RETAIL, "Total revenue per region in Q3",
     "SELECT region, SUM(revenue) FROM sales WHERE quarter='Q3' GROUP BY region", "aggregation"),
    (RETAIL, "Which product sold the most units overall?",
     "SELECT product FROM sales GROUP BY product ORDER BY SUM(units) DESC LIMIT 1", "aggregation"),
    (RETAIL, "How many sales rows are there?",
     "SELECT COUNT(*) FROM sales", "basic"),
    (RETAIL, "List the distinct regions",
     "SELECT DISTINCT region FROM sales", "basic"),
    (RETAIL, "What is the total revenue across all rows?",
     "SELECT SUM(revenue) FROM sales", "basic"),
    (RETAIL, "Show every sale with revenue above 40000, highest first",
     "SELECT * FROM sales WHERE revenue>40000 ORDER BY revenue DESC", "filter"),
    (RETAIL, "Average units sold per product",
     "SELECT product, AVG(units) FROM sales GROUP BY product", "aggregation"),
    (LIBRARY, "How many books did each author write? Show the author name.",
     "SELECT a.name, COUNT(b.book_id) FROM authors a JOIN books b ON a.author_id=b.author_id GROUP BY a.name", "join"),
    (LIBRARY, "Which books are currently on loan? Show the title.",
     "SELECT DISTINCT b.title FROM books b JOIN loans l ON b.book_id=l.book_id WHERE l.returned=0", "join"),
    (LIBRARY, "Count loans per author country",
     "SELECT a.country, COUNT(*) FROM loans l JOIN books b ON l.book_id=b.book_id JOIN authors a ON b.author_id=a.author_id GROUP BY a.country", "multi-join"),
    (LIBRARY, "List book titles published before 1975",
     "SELECT title FROM books WHERE year<1975", "basic"),
    (LIBRARY, "Which authors are from the USA?",
     "SELECT name FROM authors WHERE country='USA'", "basic"),
    (CLINIC, "Average visit cost by department, highest first",
     "SELECT department, AVG(cost) AS a FROM visits GROUP BY department ORDER BY a DESC", "aggregation"),
    (CLINIC, "Which patients have more than one visit? Show their names.",
     "SELECT p.name FROM patients p JOIN visits v ON p.patient_id=v.patient_id GROUP BY p.name HAVING COUNT(*)>1", "join"),
    (CLINIC, "Total spend per city",
     "SELECT p.city, SUM(v.cost) FROM visits v JOIN patients p ON v.patient_id=p.patient_id GROUP BY p.city", "join"),
    (CLINIC, "How many visits were there in total?",
     "SELECT COUNT(*) FROM visits", "basic"),
    (CLINIC, "Departments whose average visit cost is above the overall average",
     "SELECT department FROM visits GROUP BY department HAVING AVG(cost) > (SELECT AVG(cost) FROM visits)", "subquery"),
    (CLINIC, "List patients born before 1980",
     "SELECT name FROM patients WHERE birth_year<1980", "basic"),
]


def call(base_url: str, schema: str, question: str, timeout: int) -> dict:
    body = json.dumps({"schema": schema, "question": question}).encode()
    req = urllib.request.Request(f"{base_url.rstrip('/')}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:160]}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--out", type=Path, default=Path("tests/e2e_results.json"))
    args = ap.parse_args()

    print(f"Testing {args.base_url} with {len(CASES)} cases\n")
    results, correct, executed, repaired = [], 0, 0, 0

    for i, (schema, question, ref_sql, kind) in enumerate(CASES, 1):
        t0 = time.time()
        resp = call(args.base_url, schema, question, args.timeout)
        dt = time.time() - t0

        if "error" in resp and not resp.get("sql"):
            print(f"  {i:2}. [{kind:10}] REQUEST FAILED  {resp['error'][:80]}")
            results.append({"question": question, "kind": kind, "status": "request_failed",
                            "error": resp["error"]})
            continue

        sql = resp.get("sql", "")
        meta = resp.get("meta", {})
        ran = bool(meta.get("executed"))
        executed += ran
        repaired += bool(meta.get("repaired"))

        ok = False
        detail = meta.get("error") or ""
        if ran:
            try:
                con = build_db(schema)
                got = run_sql(con, sql)
                want = run_sql(con, ref_sql)
                con.close()
                ok = results_match(want, got, ref_sql)
                if not ok:
                    detail = f"got {str(got)[:60]} want {str(want)[:60]}"
            except Exception as e:
                detail = f"compare failed: {e}"
        correct += ok

        mark = "PASS" if ok else ("RAN " if ran else "FAIL")
        rp = " (repaired)" if meta.get("repaired") else ""
        print(f"  {i:2}. [{kind:10}] {mark}{rp}  {dt:5.1f}s  {question[:52]}")
        if not ok:
            print(f"      sql: {sql[:110]}")
            if detail:
                print(f"      {detail[:120]}")
        results.append({"question": question, "kind": kind, "sql": sql, "reference": ref_sql,
                        "executed": ran, "correct": ok, "repaired": bool(meta.get("repaired")),
                        "seconds": round(dt, 1), "detail": detail})

    n = len(CASES)
    print(f"\n{'='*58}")
    print(f"  executed without error : {executed}/{n} ({executed/n:.0%})")
    print(f"  matched the reference  : {correct}/{n} ({correct/n:.0%})")
    print(f"  needed a repair pass   : {repaired}")
    by = {}
    for r in results:
        if r.get("status") == "request_failed":
            continue
        k = r["kind"]
        by.setdefault(k, [0, 0])
        by[k][0] += r["correct"]; by[k][1] += 1
    print("\n  by question type:")
    for k, (c, t) in sorted(by.items(), key=lambda x: -x[1][1]):
        print(f"    {k:12} {c}/{t}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"base_url": args.base_url, "n": n, "executed": executed,
                                    "correct": correct, "repaired": repaired,
                                    "results": results}, indent=2))
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
