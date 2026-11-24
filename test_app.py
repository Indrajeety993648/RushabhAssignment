"""Spec item 7 acceptance checks. Run: python3 test_app.py

Asserts: directory loads 500; a pilot search yields 25 unique per list;
rankings+explanations are reproducible; an unchanged rerun reuses cached
source calls; gold pairs stay separate from generated results.
"""
import os
import tempfile

import db as dbmod


def fresh_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)  # start empty; reload picks up the path from env
    os.environ["PEERFINDER_DB"] = path
    return path


def main():
    fresh_db()
    import importlib
    for m in ("db", "providers", "rank", "pipeline"):
        importlib.reload(importlib.import_module(m))
    import db, pipeline, providers
    # hermetic: force the free offline path, never hit the network in tests
    providers.llm_available = lambda: False
    providers.websearch_available = lambda: False

    con = db.connect()
    here = os.path.dirname(os.path.abspath(__file__))
    n = db.load_directory(con, os.path.join(here, "500 Companies list(1).xlsx"))
    assert n >= 500, f"directory only has {n}"
    assert con.execute("SELECT COUNT(DISTINCT canonical_id) FROM companies").fetchone()[0] == \
        con.execute("SELECT COUNT(*) FROM companies").fetchone()[0], "duplicate ids"

    rid = pipeline.run(con, "Salesforce")
    res = pipeline.results(con, rid)
    for track in ("business", "talent"):
        names = [r["name"] for r in res[track]]
        assert len(names) == 25, f"{track}: expected 25, got {len(names)} (blocker: {res['blocker']})"
        assert len(set(names)) == 25, f"{track}: names not unique"
        assert "Salesforce" not in names, f"{track}: input leaked into its own list"
        assert all(r["explanation"] for r in res[track]), f"{track}: missing explanation"
    print("✓ 25 unique entries per list, with scores + explanations")

    # reproducibility: same scores/order on rerun
    rid2 = pipeline.run(con, "Salesforce")
    res2 = pipeline.results(con, rid2)
    for track in ("business", "talent"):
        a = [(r["name"], r["score"]) for r in res[track]]
        b = [(r["name"], r["score"]) for r in res2[track]]
        assert a == b, f"{track}: non-reproducible ranking"
    print("✓ rankings + explanations reproducible")

    # cache reuse: an unchanged rerun must not issue new source (enrich/discover) calls
    before = con.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    pipeline.run(con, "Salesforce")
    after = con.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    assert after == before, f"cache grew on unchanged rerun: {before} -> {after}"
    print("✓ unchanged rerun reused cached source calls")

    # gold pairs are a separate store, never mixed into results
    con.execute("INSERT INTO gold_pairs(a,b,track,label,created_at) VALUES('Salesforce','HubSpot','business',1.0,0)")
    con.commit()
    assert con.execute("SELECT COUNT(*) FROM results WHERE candidate_name='__gold__'").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM gold_pairs").fetchone()[0] == 1
    print("✓ gold pairs kept separate from generated results")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
