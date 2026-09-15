"""Hand-written multi-level questions, used only for evaluation.

These are deliberately NOT produced by gen_multilevel.py. Different schemas,
different phrasings, different query shapes. Scoring a model trained on
generator output against generator output measures memorisation; this measures
whether the skill transferred.

    python make_holdout.py --out data/multilevel_holdout.jsonl
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a precise text-to-SQL engine. Given a SQLite schema and a question, "
    "reply with a single valid SQLite query that answers it. "
    "Output only the SQL. No explanation, no markdown, no commentary."
)

RETAIL = """CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, quarter TEXT, product TEXT, units INTEGER, revenue INTEGER);
INSERT INTO sales VALUES (1,'North','Q3','Widget',120,52000),(2,'South','Q3','Widget',95,41000),(3,'North','Q3','Gadget',40,18000),(4,'North','Q4','Widget',140,61000),(5,'South','Q4','Gadget',88,38000),(6,'West','Q4','Widget',61,27500),(7,'West','Q3','Gadget',33,14200),(8,'South','Q3','Gadget',50,21000);"""

LIBRARY = """CREATE TABLE authors (author_id INTEGER PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE books (book_id INTEGER PRIMARY KEY, title TEXT, author_id INTEGER, year INTEGER, genre TEXT);
CREATE TABLE loans (loan_id INTEGER PRIMARY KEY, book_id INTEGER, member TEXT, loaned_on TEXT, returned INTEGER);
INSERT INTO authors VALUES (1,'Le Guin','USA'),(2,'Calvino','Italy'),(3,'Achebe','Nigeria'),(4,'Han Kang','South Korea'),(5,'Morrison','USA');
INSERT INTO books VALUES (1,'Dispossessed',1,1974,'sci-fi'),(2,'Earthsea',1,1968,'fantasy'),(3,'Invisible Cities',2,1972,'fiction'),(4,'Things Fall Apart',3,1958,'fiction'),(5,'The Vegetarian',4,2007,'fiction'),(6,'Beloved',5,1987,'fiction'),(7,'Sula',5,1973,'fiction');
INSERT INTO loans VALUES (1,1,'ana','2026-01-04',1),(2,1,'ben','2026-02-11',0),(3,3,'ana','2026-02-20',1),(4,4,'cleo','2026-03-02',0),(5,5,'ben','2026-03-09',1),(6,5,'dev','2026-03-15',0),(7,6,'ana','2026-03-20',1),(8,6,'ben','2026-03-22',1),(9,7,'cleo','2026-03-25',1);"""

CLINIC = """CREATE TABLE patients (patient_id INTEGER PRIMARY KEY, name TEXT, birth_year INTEGER, city TEXT);
CREATE TABLE visits (visit_id INTEGER PRIMARY KEY, patient_id INTEGER, visit_date TEXT, department TEXT, cost REAL);
INSERT INTO patients VALUES (1,'Rivera',1978,'Austin'),(2,'Okonkwo',1991,'Austin'),(3,'Lindqvist',1965,'Dallas'),(4,'Haddad',2001,'Dallas'),(5,'Moreau',1985,'Houston'),(6,'Silva',1995,'Houston');
INSERT INTO visits VALUES (1,1,'2026-01-12','cardiology',420.0),(2,1,'2026-03-02','cardiology',380.5),(3,2,'2026-01-28','dermatology',150.0),(4,3,'2026-02-14','cardiology',610.25),(5,4,'2026-02-20','orthopedics',295.75),(6,5,'2026-03-11','dermatology',185.0),(7,2,'2026-03-19','orthopedics',340.0),(8,3,'2026-02-25','dermatology',120.0),(9,6,'2026-03-30','cardiology',275.0);"""

ECOM = """CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, country TEXT, signup_date TEXT);
CREATE TABLE products (product_id INTEGER PRIMARY KEY, product_name TEXT, category TEXT, price REAL);
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER, order_date TEXT, quantity INTEGER, status TEXT);
INSERT INTO customers VALUES (1,'Alice','India','2025-01-10'),(2,'Bob','USA','2025-02-15'),(3,'Charlie','India','2025-03-20'),(4,'Diana','UK','2025-04-05');
INSERT INTO products VALUES (1,'Laptop','Electronics',800),(2,'Phone','Electronics',500),(3,'Headphones','Accessories',100),(4,'Keyboard','Accessories',75);
INSERT INTO orders VALUES (101,1,1,'2025-06-01',2,'completed'),(102,2,2,'2025-06-03',1,'completed'),(103,1,3,'2025-06-10',3,'completed'),(104,3,1,'2025-07-02',1,'cancelled'),(105,4,4,'2025-07-15',2,'completed'),(106,3,2,'2025-08-01',2,'completed'),(107,2,3,'2025-08-12',1,'completed');"""

CASES = [
 (RETAIL, "What fraction of each region's total revenue does each product account for? Give it as a percentage.",
  "WITH a AS (SELECT region, product, SUM(revenue) r FROM sales GROUP BY region,product) SELECT region,product,r*100.0/SUM(r) OVER (PARTITION BY region) FROM a ORDER BY region"),
 (RETAIL, "In each quarter, which single product brought in the most revenue?",
  "WITH a AS (SELECT quarter, product, SUM(revenue) r FROM sales GROUP BY quarter,product), b AS (SELECT *, RANK() OVER (PARTITION BY quarter ORDER BY r DESC) k FROM a) SELECT quarter,product,r FROM b WHERE k=1 ORDER BY quarter"),
 (RETAIL, "Show products whose total units in a region exceed that region's average product units.",
  "WITH a AS (SELECT region, product, SUM(units) u FROM sales GROUP BY region,product), m AS (SELECT *, AVG(u) OVER (PARTITION BY region) av FROM a) SELECT region,product,u FROM m WHERE u>av ORDER BY region"),
 (RETAIL, "Rank products inside each quarter by total revenue.",
  "WITH a AS (SELECT quarter, product, SUM(revenue) r FROM sales GROUP BY quarter,product) SELECT quarter,product,r,RANK() OVER (PARTITION BY quarter ORDER BY r DESC) FROM a ORDER BY quarter"),
 (RETAIL, "For each region give the two products with the biggest revenue and their percentage of the region total.",
  "WITH a AS (SELECT region, product, SUM(revenue) r FROM sales GROUP BY region,product), b AS (SELECT *, r*100.0/SUM(r) OVER (PARTITION BY region) p, RANK() OVER (PARTITION BY region ORDER BY r DESC) k FROM a) SELECT region,product,r,p FROM b WHERE k<=2 ORDER BY region,k"),
 (LIBRARY, "For every country, list the genres whose book count is above that country's average genre count.",
  "WITH a AS (SELECT au.country c, b.genre g, COUNT(*) n FROM books b JOIN authors au ON b.author_id=au.author_id GROUP BY au.country,b.genre), m AS (SELECT *, AVG(n) OVER (PARTITION BY c) av FROM a) SELECT c,g,n FROM m WHERE n>av ORDER BY c"),
 (LIBRARY, "Per author country, rank genres by how many times their books were loaned.",
  "WITH a AS (SELECT au.country c, b.genre g, COUNT(*) n FROM loans l JOIN books b ON l.book_id=b.book_id JOIN authors au ON b.author_id=au.author_id GROUP BY au.country,b.genre) SELECT c,g,n,RANK() OVER (PARTITION BY c ORDER BY n DESC) FROM a ORDER BY c"),
 (LIBRARY, "Which author accounts for the largest share of loans in their own country, and what share is it?",
  "WITH a AS (SELECT au.country c, au.name nm, COUNT(*) n FROM loans l JOIN books b ON l.book_id=b.book_id JOIN authors au ON b.author_id=au.author_id GROUP BY au.country,au.name), b2 AS (SELECT *, n*100.0/SUM(n) OVER (PARTITION BY c) p, RANK() OVER (PARTITION BY c ORDER BY n DESC) k FROM a) SELECT c,nm,p FROM b2 WHERE k=1 ORDER BY c"),
 (LIBRARY, "What percentage of each country's books fall into each genre?",
  "WITH a AS (SELECT au.country c, b.genre g, COUNT(*) n FROM books b JOIN authors au ON b.author_id=au.author_id GROUP BY au.country,b.genre) SELECT c,g,n*100.0/SUM(n) OVER (PARTITION BY c) FROM a ORDER BY c"),
 (CLINIC, "For each city, show the two departments with the largest total cost and what share of the city's spend each represents.",
  "WITH a AS (SELECT p.city ci, v.department d, SUM(v.cost) s FROM visits v JOIN patients p ON v.patient_id=p.patient_id GROUP BY p.city,v.department), b AS (SELECT *, s*100.0/SUM(s) OVER (PARTITION BY ci) pc, RANK() OVER (PARTITION BY ci ORDER BY s DESC) k FROM a) SELECT ci,d,s,pc FROM b WHERE k<=2 ORDER BY ci,k"),
 (CLINIC, "Which departments cost more per city than that city's typical department?",
  "WITH a AS (SELECT p.city ci, v.department d, SUM(v.cost) s FROM visits v JOIN patients p ON v.patient_id=p.patient_id GROUP BY p.city,v.department), m AS (SELECT *, AVG(s) OVER (PARTITION BY ci) av FROM a) SELECT ci,d,s FROM m WHERE s>av ORDER BY ci"),
 (CLINIC, "In every city, which department has the single highest total cost?",
  "WITH a AS (SELECT p.city ci, v.department d, SUM(v.cost) s FROM visits v JOIN patients p ON v.patient_id=p.patient_id GROUP BY p.city,v.department), b AS (SELECT *, RANK() OVER (PARTITION BY ci ORDER BY s DESC) k FROM a) SELECT ci,d,s FROM b WHERE k=1 ORDER BY ci"),
 (CLINIC, "Give each department's share of total spend within its city as a percentage.",
  "WITH a AS (SELECT p.city ci, v.department d, SUM(v.cost) s FROM visits v JOIN patients p ON v.patient_id=p.patient_id GROUP BY p.city,v.department) SELECT ci,d,s*100.0/SUM(s) OVER (PARTITION BY ci) FROM a ORDER BY ci"),
 (CLINIC, "Rank departments within each city by number of visits.",
  "WITH a AS (SELECT p.city ci, v.department d, COUNT(*) n FROM visits v JOIN patients p ON v.patient_id=p.patient_id GROUP BY p.city,v.department) SELECT ci,d,n,RANK() OVER (PARTITION BY ci ORDER BY n DESC) FROM a ORDER BY ci"),
 (ECOM, "For each country, find the top 2 product categories by completed-order revenue. Show revenue, unique customers, percentage of country revenue, and rank. Exclude cancelled orders.",
  "WITH a AS (SELECT c.country co, p.category ca, SUM(o.quantity*p.price) r, COUNT(DISTINCT o.customer_id) u FROM orders o JOIN customers c ON o.customer_id=c.customer_id JOIN products p ON o.product_id=p.product_id WHERE o.status='completed' GROUP BY c.country,p.category), b AS (SELECT *, r*100.0/SUM(r) OVER (PARTITION BY co) pc, RANK() OVER (PARTITION BY co ORDER BY r DESC) k FROM a) SELECT co,ca,r,u,pc,k FROM b WHERE k<=2 ORDER BY co,k"),
 (ECOM, "What share of each country's completed revenue comes from each category?",
  "WITH a AS (SELECT c.country co, p.category ca, SUM(o.quantity*p.price) r FROM orders o JOIN customers c ON o.customer_id=c.customer_id JOIN products p ON o.product_id=p.product_id WHERE o.status='completed' GROUP BY c.country,p.category) SELECT co,ca,r*100.0/SUM(r) OVER (PARTITION BY co) FROM a ORDER BY co"),
 (ECOM, "Which category leads each country by completed revenue?",
  "WITH a AS (SELECT c.country co, p.category ca, SUM(o.quantity*p.price) r FROM orders o JOIN customers c ON o.customer_id=c.customer_id JOIN products p ON o.product_id=p.product_id WHERE o.status='completed' GROUP BY c.country,p.category), b AS (SELECT *, RANK() OVER (PARTITION BY co ORDER BY r DESC) k FROM a) SELECT co,ca,r FROM b WHERE k=1 ORDER BY co"),
 (ECOM, "Show categories earning above their country's average category revenue, completed orders only.",
  "WITH a AS (SELECT c.country co, p.category ca, SUM(o.quantity*p.price) r FROM orders o JOIN customers c ON o.customer_id=c.customer_id JOIN products p ON o.product_id=p.product_id WHERE o.status='completed' GROUP BY c.country,p.category), m AS (SELECT *, AVG(r) OVER (PARTITION BY co) av FROM a) SELECT co,ca,r FROM m WHERE r>av ORDER BY co"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/multilevel_holdout.jsonl"))
    args = ap.parse_args()

    out, bad = [], 0
    for i, (schema, q, ref) in enumerate(CASES, 1):
        try:
            con = sqlite3.connect(":memory:")
            con.executescript(schema)
            rows = con.execute(ref).fetchall()
            con.close()
            if not rows:
                print(f"  case {i}: reference returns no rows"); bad += 1; continue
        except Exception as e:
            print(f"  case {i}: reference BROKEN -- {e}"); bad += 1; continue
        out.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"### Schema\n{schema}\n\n### Question\n{q}"},
                {"role": "assistant", "content": ref},
            ],
            "sql_context": schema, "sql_prompt": q, "gold_sql": ref,
            "domain": "handwritten", "complexity": "multi-level aggregation",
            "source": "handwritten-holdout", "id": f"hold{i}",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"  {len(out)} hand-written holdout cases -> {args.out}  ({bad} rejected)")


if __name__ == "__main__":
    main()
