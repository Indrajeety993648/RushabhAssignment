"""End-to-end run: enrich -> discover -> rank both tracks -> persist (spec item 7).

Shared company IDs, caches, and results across both ranking tracks. Progress is
tracked in the runs table so the UI can poll it.
"""
import json
import time
import uuid

import db
import rank


def _set(con, run_id, **kw):
    cols = ", ".join(f"{k}=?" for k in kw)
    con.execute(f"UPDATE runs SET {cols} WHERE id=?", (*kw.values(), run_id))
    con.commit()


def _persist_profile(con, p):
    co = db.get_company(con, p["name"])
    if not co:  # discovered company outside the 500 -> add to directory
        cid = db.canonical(p["name"], p.get("website"))
        con.execute("""INSERT OR IGNORE INTO companies(name,canonical_id,website,
                       revenue_musd,headcount,source_meta,created_at)
                       VALUES(?,?,?,?,?,?,?)""",
                    (p["name"], cid, p.get("website"),
                     (p.get("scale") or {}).get("revenue_musd"),
                     (p.get("scale") or {}).get("headcount"),
                     json.dumps({"_all": p.get("provider")}), time.time()))
        con.commit()
        co = db.get_company(con, p["name"])
    cid_ = co["id"]
    con.execute("""INSERT OR REPLACE INTO profiles(company_id,sector,industry,offering,
                   taxonomy_version,offerings_json,features_json,evidence_json,confidence,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (cid_, p.get("sector"), p.get("industry"), p.get("offering"),
                 __import__("seed").TAXONOMY_VERSION, json.dumps(p.get("offerings")),
                 json.dumps({"capabilities": p.get("capabilities"),
                             "customer_type": p.get("customer_type"),
                             "business_model": p.get("business_model"),
                             "geography": p.get("geography"), "scale": p.get("scale")}),
                 json.dumps(p.get("evidence")), p.get("confidence"), time.time()))
    con.execute("""INSERT OR REPLACE INTO talent_profiles(company_id,career_site,roles_json,
                   seniority_json,skills_json,requirements_json,geo_json,hiring_volume_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (cid_, p.get("career_site"), json.dumps(p.get("roles")),
                 json.dumps(p.get("seniority")), json.dumps(p.get("skills")),
                 json.dumps({"observed": [], "inferred": p.get("skills", [])}),
                 json.dumps(p.get("geography")), json.dumps({}), time.time()))
    con.commit()
    return cid_


def _store_results(con, run_id, input_id, track, rows, blocker):
    con.execute("DELETE FROM results WHERE run_id=? AND track=?", (run_id, track))
    for r in rows:
        co = db.get_company(con, r["name"])
        con.execute("""INSERT INTO results(run_id,input_company_id,track,rank,candidate_name,
                       candidate_id,score,confidence,explanation,features_json,config_version,
                       incomplete,blocker,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (run_id, input_id, track, r["rank"], r["name"], co["id"] if co else None,
                     r["score"], r.get("confidence", "medium"), r["explanation"],
                     json.dumps(r["features"]), db.CONFIG_VERSION,
                     1 if blocker else 0, blocker, time.time()))
    con.commit()


def run(con, input_name, run_id=None):
    run_id = run_id or uuid.uuid4().hex[:12]
    co = db.get_company(con, input_name)
    con.execute("""INSERT OR REPLACE INTO runs(id,input_company_id,status,progress,stage,
                   config_version,created_at) VALUES(?,?,?,?,?,?,?)""",
                (run_id, co["id"] if co else None, "running", 0.05, "enriching input",
                 db.CONFIG_VERSION, time.time()))
    con.commit()

    ip, profiles = rank.build_pool(con, input_name, allow_web=False)
    if ip.get("unavailable"):
        _set(con, run_id, status="blocked", progress=1.0, stage="blocked",
             blocker=f'Could not build a profile for "{input_name}" from the free seed '
                     f'or web search. Check provider keys.')
        return run_id

    def _rank():
        b, bbl = rank.finalize(rank.rank_business(ip, profiles), rank.CONFIG["target"])
        t, tbl = rank.finalize(rank.rank_talent(ip, profiles), rank.CONFIG["target"])
        return b, bbl, t, tbl

    _set(con, run_id, progress=0.5, stage="ranking (free seed pool)")
    biz_top, biz_block, tal_top, tal_block = _rank()

    # Expand with paid providers only if the free pool fell short (spec item 4).
    import providers as _pv
    if (biz_block or tal_block) and _pv.websearch_available():
        _set(con, run_id, progress=0.65, stage="expanding via web search")
        ip, profiles = rank.build_pool(con, input_name, allow_web=True)
        biz_top, biz_block, tal_top, tal_block = _rank()

    # Talent track needs hiring-skill signals; web enrichment doesn't extract them.
    if not tal_top and not ip.get("skills"):
        tal_block = ("Talent peers need hiring-skill signals (role/skill mix). The curated "
                     "seed has them; web-search enrichment does not, and the LLM provider is "
                     "unavailable. Enable a working LLM key to rank talent peers for companies "
                     "outside the seed cluster.")

    _set(con, run_id, progress=0.85, stage="persisting")
    _persist_profile(con, ip)
    for p in profiles:
        _persist_profile(con, p)
    _store_results(con, run_id, co["id"] if co else None, "business", biz_top, biz_block)
    _store_results(con, run_id, co["id"] if co else None, "talent", tal_top, tal_block)

    blocker = "; ".join(b for b in (biz_block, tal_block) if b) or None
    _set(con, run_id, status="done", progress=1.0, stage="done", blocker=blocker)
    return run_id


def results(con, run_id):
    r = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not r:
        return None
    out = {"run_id": run_id, "status": r["status"], "progress": r["progress"],
           "stage": r["stage"], "blocker": r["blocker"], "business": [], "talent": []}
    for track in ("business", "talent"):
        rows = con.execute(
            "SELECT rank,candidate_name,score,confidence,explanation,features_json,blocker "
            "FROM results WHERE run_id=? AND track=? ORDER BY rank", (run_id, track)).fetchall()
        out[track] = [{"rank": x["rank"], "name": x["candidate_name"], "score": x["score"],
                       "confidence": x["confidence"], "explanation": x["explanation"],
                       "features": json.loads(x["features_json"])} for x in rows]
    return out
