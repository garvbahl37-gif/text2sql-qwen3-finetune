export type Preset = {
  id: string;
  name: string;
  note: string;
  schema: string;
  questions: string[];
};

export const PRESETS: Preset[] = [
  {
    id: "retail",
    name: "retail",
    note: "one table, grouping and filtering",
    schema: `CREATE TABLE sales (
  id       INTEGER PRIMARY KEY,
  region   TEXT,
  quarter  TEXT,
  product  TEXT,
  units    INTEGER,
  revenue  INTEGER
);
INSERT INTO sales VALUES
  (1,'North','Q3','Widget',120,52000),
  (2,'South','Q3','Widget', 95,41000),
  (3,'North','Q3','Gadget', 40,18000),
  (4,'North','Q4','Widget',140,61000),
  (5,'South','Q4','Gadget', 88,38000),
  (6,'West', 'Q4','Widget', 61,27500),
  (7,'West', 'Q3','Gadget', 33,14200);`,
    questions: [
      "Total revenue per region in Q3",
      "Which product sold the most units overall?",
      "Regions where Q4 revenue beat Q3 revenue",
    ],
  },
  {
    id: "library",
    name: "library",
    note: "three tables, joins across them",
    schema: `CREATE TABLE authors (
  author_id  INTEGER PRIMARY KEY,
  name       TEXT,
  country    TEXT
);
CREATE TABLE books (
  book_id    INTEGER PRIMARY KEY,
  title      TEXT,
  author_id  INTEGER,
  year       INTEGER,
  genre      TEXT
);
CREATE TABLE loans (
  loan_id    INTEGER PRIMARY KEY,
  book_id    INTEGER,
  member     TEXT,
  loaned_on  TEXT,
  returned   INTEGER
);
INSERT INTO authors VALUES
  (1,'Ursula K. Le Guin','USA'),(2,'Italo Calvino','Italy'),
  (3,'Chinua Achebe','Nigeria'),(4,'Han Kang','South Korea');
INSERT INTO books VALUES
  (1,'The Dispossessed',1,1974,'sci-fi'),
  (2,'A Wizard of Earthsea',1,1968,'fantasy'),
  (3,'Invisible Cities',2,1972,'fiction'),
  (4,'Things Fall Apart',3,1958,'fiction'),
  (5,'The Vegetarian',4,2007,'fiction');
INSERT INTO loans VALUES
  (1,1,'ana','2026-01-04',1),(2,1,'ben','2026-02-11',0),
  (3,3,'ana','2026-02-20',1),(4,4,'cleo','2026-03-02',0),
  (5,5,'ben','2026-03-09',1),(6,5,'dev','2026-03-15',0);`,
    questions: [
      "Which books are currently on loan and who has them?",
      "Count loans per author country",
      "Titles published before 1975 that have never been loaned",
    ],
  },
  {
    id: "clinic",
    name: "clinic",
    note: "aggregates, dates and a subquery",
    schema: `CREATE TABLE patients (
  patient_id  INTEGER PRIMARY KEY,
  name        TEXT,
  birth_year  INTEGER,
  city        TEXT
);
CREATE TABLE visits (
  visit_id    INTEGER PRIMARY KEY,
  patient_id  INTEGER,
  visit_date  TEXT,
  department  TEXT,
  cost        REAL
);
INSERT INTO patients VALUES
  (1,'Rivera',1978,'Austin'),(2,'Okonkwo',1991,'Austin'),
  (3,'Lindqvist',1965,'Dallas'),(4,'Haddad',2001,'Dallas'),
  (5,'Moreau',1985,'Houston');
INSERT INTO visits VALUES
  (1,1,'2026-01-12','cardiology',420.00),
  (2,1,'2026-03-02','cardiology',380.50),
  (3,2,'2026-01-28','dermatology',150.00),
  (4,3,'2026-02-14','cardiology',610.25),
  (5,4,'2026-02-20','orthopedics',295.75),
  (6,5,'2026-03-11','dermatology',185.00),
  (7,2,'2026-03-19','orthopedics',340.00);`,
    questions: [
      "Average visit cost by department, highest first",
      "Patients with more than one visit",
      "Total spend per city for visits after February 2026",
    ],
  },
];
