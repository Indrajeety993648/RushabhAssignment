# Company & Talent Peer Finder

Pick one company from the supplied 500-company workbook; get two ranked lists of
25 — **direct business competitors** and **talent peers** — each row with rank,
score, and an evidence-based explanation.

## Run

```bash
python3 db.py                 # load the 500-company directory into peerfinder.db
python3 test_app.py           # acceptance checks (25 unique/list, reproducible, cache reuse)
PORT=8000 python3 server.py   # open http://localhost:8000
```

Stdlib only for the app runtime (`sqlite3`, `http.server`) + `openpyxl` to read
the workbook. No framework, no Streamlit.

## How enrichment works

Keys are read from `.env` (auto-loaded; git-ignored). The enrichment/discovery
layer is a pluggable provider chain (`providers.py`): **free curated seed →
LLM → web search**, all cached in SQLite by `(provider, query, params)`.

Cost discipline (keys are metered): a search first ranks against the **free**
seed pool + cache and spends **nothing**; paid providers fire only if that pool
can't fill 25 (spec item 4's "expand until 25 qualify"). A seed-cluster company
(e.g. Salesforce) returns 25+25 in ~0.02s with zero network calls. The LLM is
marked dead after one hard failure so it isn't retried.

- **LLM** — set `AI_GATEWAY_API_KEY` (OpenAI-compatible AI Gateway). The model
  id is verified against the live `/v1/models` list at runtime (picked
  `anthropic/claude-sonnet-4.5` here), never hard-coded. Enriches profiles and
  discovers candidates beyond the 500, including hiring skills for the talent
  track. **Note:** the supplied key returns `403 customer_verification_required`
  — Vercel locks even free credits until a credit card is on file, so the LLM
  path is currently inactive. Add a card to unlock it; no code change needed.
- **Web search** — set `FIRECRAWL_API_KEY` (or `SEARCH_API_KEY`) to enable the
  Firecrawl-backed fallback (`providers.web_search` → `POST /v2/search`, cached).
  It kicks in when the LLM is unavailable or returns fewer than 25 candidates:
  `_websearch_discover` proposes candidate names from competitor-list results
  and `_websearch_enrich` structures scraped page text into the controlled
  offering vocabulary (deterministic keyword match, no LLM, lower confidence).
  Proposed names are re-enriched and dropped if they yield no verifiable
  offering — self-verifying. Invalid key → degrades to the offline seed, no crash.
- **Web search (Firecrawl)** — verified working with the supplied key. Fills
  the gap while the LLM is billing-locked: for a non-seed company it enriches
  the input and discovers candidates from competitor-list results (~12 metered
  calls per such search). Keyword-based enrichment is noisier than the LLM and
  does **not** extract hiring skills, so the talent track stays empty for
  web-only companies (reported as a blocker). Business ranking still fills 25.
- **Offline seed (no key)** — a hand-authored ~36-company enterprise-software cluster
  (`seed.py`) keeps the app fully runnable and deterministic. Dense inputs
  (Salesforce, Microsoft) fill 25/25; sparse ones (e.g. Intuit) return only the
  candidates with genuine offering evidence and **report a blocker** — the
  spec-sanctioned "keep the run incomplete" path, not a crash.

## Ranking

- **Business** (`rank.rank_business`): weighted offering overlap + customer-type
  + market/geo overlap + BM25 + local TF-IDF cosine, combined by a deterministic
  reranker (score desc, name tiebreak → reproducible). Qualifies only with a
  shared (substitutable) offering for a compatible customer type.
- **Talent** (`rank.rank_talent`): idf-discounted skill overlap + role-mix +
  seniority-mix similarity + cosine. A directed mobility-flow graph + PageRank
  tier are used *only* where authorized aggregate flow data exists; absent that
  (the default here) it uses hiring-profile similarity and reports **low
  confidence**. Business and talent are scored independently.

TF-IDF cosine is the local vector signal (no model download). A
`sentence-transformers` cross-encoder is the documented upgrade path if it beats
human-graded pairs — `gold_pairs` table holds those judgments, kept separate
from generated `results`.

## Files

| file | role |
|------|------|
| `db.py` | SQLite schema, 500-directory loader, cache, stable IDs/dedup |
| `seed.py` | versioned 3-level taxonomy + offline pilot knowledge base |
| `providers.py` | LLM / web-search / offline enrichment + discovery, cached |
| `rank.py` | BM25, overlap, cosine, deterministic rerankers (both tracks) |
| `pipeline.py` | run orchestration, progress, profile/result persistence |
| `server.py` + `index.html` | stdlib HTTP API + single-field/one-button UI |
| `test_app.py` | spec item-7 acceptance checks |

## Known ceilings (deliberate)

- Offline coverage is a **pilot cluster**, not all 500 — full coverage needs the
  LLM/web providers (set a key). The blocker message says so.
- Career-site posting ingestion (Greenhouse/Lever/Ashby/Workday scraping) and
  authorized mobility-flow data are modeled in the schema and talent profile but
  not live-scraped here; talent ranking runs on hiring-profile similarity and
  flags lower confidence accordingly.
