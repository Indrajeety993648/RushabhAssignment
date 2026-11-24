"""SQLite persistence: schema, company directory loader, cache, results.

One file, stdlib sqlite3. Shared across both ranking tracks (spec item 7).
"""
import hashlib
import json
import os
import sqlite3
import time

def _load_env():
    """Load .env (KEY=VALUE lines) into os.environ without clobbering real env."""
    path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
DB_PATH = os.environ.get("PEERFINDER_DB") or os.path.join(os.path.dirname(__file__), "peerfinder.db")
CONFIG_VERSION = "cfg-v1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY,
    sno INTEGER,
    name TEXT NOT NULL,
    canonical_id TEXT UNIQUE,      -- verified identity key (website/wikidata/name)
    website TEXT,
    wikidata_id TEXT,
    country TEXT,
    revenue_musd REAL,
    headcount INTEGER,
    source_meta TEXT,              -- json: where each field came from
    raw_json TEXT,                 -- raw provider response(s)
    missing_json TEXT,             -- explicitly recorded missing fields
    created_at REAL
);
CREATE TABLE IF NOT EXISTS profiles (
    company_id INTEGER PRIMARY KEY REFERENCES companies(id),
    sector TEXT, industry TEXT, offering TEXT,
    taxonomy_version TEXT,
    offerings_json TEXT,           -- weighted core/adjacent offerings
    features_json TEXT,            -- capabilities, customer_type, business_model, geography, scale
    evidence_json TEXT,
    confidence REAL,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS talent_profiles (
    company_id INTEGER PRIMARY KEY REFERENCES companies(id),
    career_site TEXT,
    roles_json TEXT, seniority_json TEXT,
    skills_json TEXT,              -- distinctive skill weights (idf-discounted)
    requirements_json TEXT,        -- observed vs inferred
    geo_json TEXT, hiring_volume_json TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS cache (
    provider TEXT, query_hash TEXT, params_hash TEXT,
    query TEXT, params_json TEXT, response_json TEXT, created_at REAL,
    PRIMARY KEY (provider, query_hash, params_hash)
);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seed_company_id INTEGER,       -- discovery was triggered by this input
    name TEXT, canonical_id TEXT,
    bucket TEXT, market TEXT,
    evidence_json TEXT, verified INTEGER, source TEXT,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_cand_bucket ON candidates(bucket);
CREATE INDEX IF NOT EXISTS idx_cand_market ON candidates(market);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    input_company_id INTEGER,
    status TEXT, progress REAL, stage TEXT, blocker TEXT,
    config_version TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT, input_company_id INTEGER,
    track TEXT,                    -- 'business' | 'talent'
    rank INTEGER, candidate_name TEXT, candidate_id INTEGER,
    score REAL, confidence TEXT, explanation TEXT,
    features_json TEXT, config_version TEXT,
    incomplete INTEGER, blocker TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS gold_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    a TEXT, b TEXT, track TEXT, label REAL, created_at REAL
);
"""


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def h(s):
    return hashlib.sha256(json.dumps(s, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------- company directory ----------

def canonical(name, website=None):
    """Stable identity key. Prefer verified website, else normalized name."""
    if website:
        return "web:" + website.lower().replace("https://", "").replace("http://", "").strip("/").split("/")[0]
    return "name:" + "".join(ch for ch in name.lower() if ch.isalnum())


def load_directory(con, xlsx_path):
    """Seed companies table from the supplied workbook. Idempotent (dedup by canonical_id)."""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active or wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip().lower() if c else "" for c in rows[0]]
    name_col = next((i for i, hname in enumerate(header) if "name" in hname), 1)
    sno_col = next((i for i, hname in enumerate(header) if "s.no" in hname or hname == "sno"), 0)
    n = 0
    for row in rows[1:]:
        if not row or not row[name_col]:
            continue
        name = str(row[name_col]).strip()
        sno = row[sno_col] if sno_col < len(row) else None
        cid = canonical(name)
        con.execute(
            """INSERT INTO companies(sno,name,canonical_id,source_meta,missing_json,created_at)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(canonical_id) DO UPDATE SET name=excluded.name, sno=excluded.sno""",
            (sno, name, cid, json.dumps({"name": "workbook"}),
             json.dumps(["website", "revenue", "headcount", "country"]), time.time()),
        )
        n += 1
    con.commit()
    return n


def get_company(con, name_or_id):
    if isinstance(name_or_id, int) or str(name_or_id).isdigit():
        r = con.execute("SELECT * FROM companies WHERE id=?", (int(name_or_id),)).fetchone()
        if r:
            return r
    return con.execute(
        "SELECT * FROM companies WHERE lower(name)=lower(?) OR canonical_id=?",
        (str(name_or_id), canonical(str(name_or_id)))).fetchone()


def search_companies(con, q, limit=20):
    rows = con.execute(
        "SELECT id,name FROM companies WHERE name LIKE ? ORDER BY length(name) LIMIT ?",
        (f"%{q}%", limit)).fetchall()
    return [{"id": r["id"], "name": r["name"]} for r in rows]


# ---------- cache ----------

def cache_get(con, provider, query, params):
    r = con.execute(
        "SELECT response_json FROM cache WHERE provider=? AND query_hash=? AND params_hash=?",
        (provider, h(query), h(params))).fetchone()
    return json.loads(r["response_json"]) if r else None


def cache_put(con, provider, query, params, response):
    con.execute(
        """INSERT OR REPLACE INTO cache(provider,query_hash,params_hash,query,params_json,response_json,created_at)
           VALUES(?,?,?,?,?,?,?)""",
        (provider, h(query), h(params), query, json.dumps(params),
         json.dumps(response), time.time()))
    con.commit()


if __name__ == "__main__":
    import sys
    con = connect()
    xlsx = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "500 Companies list(1).xlsx")
    n = load_directory(con, xlsx)
    print(f"loaded {n} companies into {DB_PATH}")
    assert con.execute("SELECT COUNT(*) FROM companies").fetchone()[0] >= 500, "directory incomplete"
    print("self-check ok: >=500 companies, canonical ids unique")
