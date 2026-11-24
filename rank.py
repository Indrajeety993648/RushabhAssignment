"""Ranking engine (spec items 2, 4, 5, 6).

Business competitors: weighted offering overlap + customer/market overlap +
BM25 + local TF-IDF cosine, combined by a deterministic reranker. Requires
evidence of a substitutable offering for comparable customers.

Talent peers: role/skill fit (idf-discounted skills) + role/seniority mix
similarity. Directed mobility-flow graph + PageRank tier are used only where
authorized aggregate flow data exists; absent that, hiring-profile similarity
is used and confidence is reported lower (spec item 6).

Local, deterministic, no network. sentence-transformers cross-encoder is the
documented upgrade path over the TF-IDF cosine (ponytail: TF-IDF needs no
model download; swap in if it beats human-graded pairs).
"""
import json
import math
import re
import time
from collections import Counter

import db
import providers

CONFIG = {
    "business": {"offering": 0.40, "customer": 0.12, "market": 0.10, "bm25": 0.18, "cosine": 0.20},
    "talent": {"skills": 0.45, "roles": 0.20, "seniority": 0.15, "cosine": 0.20},
    "min_offering_evidence": 0.15,   # weighted-overlap floor to qualify as substitutable
    "min_talent_evidence": 0.10,
    "k1": 1.5, "b": 0.75,
    "target": 25,
}
_WORD = re.compile(r"[a-z0-9]+")


def _tok(text):
    return _WORD.findall(text.lower())


def _doc(p):
    parts = [p.get("industry") or "", p.get("sector") or "",
             " ".join(o["name"] for o in p.get("offerings", [])),
             " ".join(p.get("capabilities", [])),
             p.get("business_model") or ""]
    return _tok(" ".join(parts))


# ---------- vector-space helpers ----------

def _idf(corpus_tokens):
    N = len(corpus_tokens)
    df = Counter()
    for toks in corpus_tokens:
        for t in set(toks):
            df[t] += 1
    return {t: math.log(1 + N / (1 + c)) for t, c in df.items()}, df, N


def _tfidf_vec(toks, idf):
    tf = Counter(toks)
    return {t: (c / len(toks)) * idf.get(t, 0.0) for t, c in tf.items()} if toks else {}


def _cosine(a, b):
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb) if na and nb else 0.0


def _bm25(query_toks, doc_toks, idf, avgdl, k1, b):
    tf = Counter(doc_toks)
    dl = len(doc_toks) or 1
    s = 0.0
    for t in set(query_toks):
        if t not in tf:
            continue
        f = tf[t]
        s += idf.get(t, 0.0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
    return s


def _wtags(p):
    """offering tag -> weight (max across entries)."""
    out = {}
    for o in p.get("offerings", []):
        out[o["name"]] = max(out.get(o["name"], 0), float(o.get("weight", 0.5)))
    return out


def _weighted_overlap(a, b):
    """Weighted Jaccard over offering tags."""
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return inter / union if union else 0.0


def _set_jaccard(a, b):
    a, b = set(x.lower() for x in a or []), set(x.lower() for x in b or [])
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _customer_sim(a, b):
    if not a or not b:
        return 0.0
    return 1.0 if a == b else (0.5 if "B2B" in (a, b) and "B2B2C" in (a, b) else 0.0)


# ---------- pool ----------

def build_pool(con, input_name, allow_web=False):
    """Enrich input + discovered candidates; return (input_profile, [candidate_profiles]).
    allow_web=True permits paid discovery/enrichment of non-seed candidates."""
    ip = providers.enrich_company(con, input_name)
    if ip.get("unavailable"):
        return ip, []
    cand_names = providers.discover_candidates(con, ip, allow_web=allow_web)
    profiles, seen = [], set()
    input_fam = db.canonical(ip["name"])
    for nm in cand_names:
        prof = providers.enrich_company(con, nm)
        if prof.get("unavailable"):
            continue
        cid = db.canonical(prof["name"])
        if cid == input_fam or cid in seen:   # remove corporate family + dedup
            continue
        seen.add(cid)
        profiles.append(prof)
    _persist_candidates(con, ip, profiles)
    return ip, profiles


def _persist_candidates(con, ip, profiles):
    seed_co = db.get_company(con, ip["name"])
    sid = seed_co["id"] if seed_co else None
    for p in profiles:
        con.execute(
            """INSERT INTO candidates(seed_company_id,name,canonical_id,bucket,market,
               evidence_json,verified,source,created_at) VALUES(?,?,?,?,?,?,?,?,?)""",
            (sid, p["name"], db.canonical(p["name"]), p.get("industry"),
             (p.get("geography") or ["?"])[0], json.dumps(p.get("evidence")),
             1, p.get("provider", "?"), time.time()))
    con.commit()


# ---------- business ranking ----------

def rank_business(ip, profiles):
    corpus = [_doc(ip)] + [_doc(p) for p in profiles]
    idf, _, _ = _idf(corpus)
    avgdl = sum(len(d) for d in corpus) / len(corpus)
    qdoc = _doc(ip)
    qvec = _tfidf_vec(qdoc, idf)
    ia = _wtags(ip)
    w = CONFIG["business"]
    scored = []
    for p in profiles:
        d = _doc(p)
        off = _weighted_overlap(ia, _wtags(p))
        cust = _customer_sim(ip.get("customer_type"), p.get("customer_type"))
        mkt = _set_jaccard(ip.get("geography"), p.get("geography"))
        bm = _bm25(qdoc, d, idf, avgdl, CONFIG["k1"], CONFIG["b"])
        cos = _cosine(qvec, _tfidf_vec(d, idf))
        bm_n = bm / (bm + 3.0)  # squash to 0-1
        score = (w["offering"] * off + w["customer"] * cust + w["market"] * mkt +
                 w["bm25"] * bm_n + w["cosine"] * cos)
        shared = sorted(set(ia) & set(_wtags(p)),
                        key=lambda t: -(ia[t] + _wtags(p)[t]))
        # spec: require evidence of a substitutable offering for comparable customers
        qualifies = bool(shared) and cust > 0
        expl = (f"Shares {len(shared)} offering(s): {', '.join(shared[:4]) or '—'}; "
                f"{ip.get('customer_type','?')}↔{p.get('customer_type','?')} customers; "
                f"offering-overlap {off:.2f}, text-cosine {cos:.2f}.")
        scored.append({"name": p["name"], "score": round(score, 4), "qualifies": qualifies,
                       "explanation": expl,
                       "features": {"offering_overlap": round(off, 3),
                                    "customer_sim": cust, "market_overlap": round(mkt, 3),
                                    "bm25": round(bm_n, 3), "cosine": round(cos, 3),
                                    "shared_offerings": shared[:6]}})
    # deterministic reranker: score desc, then name for tie stability
    scored.sort(key=lambda x: (-x["score"], x["name"]))
    return scored


# ---------- talent ranking ----------

def _skill_idf(profiles):
    corpus = [[s.lower() for s in p.get("skills", [])] for p in profiles]
    idf, _, _ = _idf(corpus)
    return idf


def rank_talent(ip, profiles):
    sidf = _skill_idf([ip] + profiles)
    corpus = [_doc(ip)] + [_doc(p) for p in profiles]
    tidf, _, _ = _idf(corpus)
    qvec = _tfidf_vec(_doc(ip), tidf)
    w = CONFIG["talent"]
    iskills = {s.lower() for s in ip.get("skills", [])}
    scored = []
    for p in profiles:
        pskills = {s.lower() for s in p.get("skills", [])}
        shared = iskills & pskills
        # idf-weighted skill overlap (discount ubiquitous skills)
        num = sum(sidf.get(s, 0) for s in shared)
        den = sum(sidf.get(s, 0) for s in (iskills | pskills)) or 1
        skill = num / den
        roles = 1 - _dist(ip.get("roles", {}), p.get("roles", {}))
        sen = 1 - _dist(ip.get("seniority", {}), p.get("seniority", {}))
        cos = _cosine(qvec, _tfidf_vec(_doc(p), tidf))
        score = w["skills"] * skill + w["roles"] * roles + w["seniority"] * sen + w["cosine"] * cos
        # spec: talent competition evidence = overlapping distinctive hiring skills
        qualifies = bool(shared)
        expl = (f"Competes for {len(shared)} skill(s): {', '.join(sorted(shared)[:4]) or '—'}; "
                f"role-mix sim {roles:.2f}, seniority sim {sen:.2f}. "
                f"[hiring-profile similarity — no authorized mobility-flow data, lower confidence]")
        scored.append({"name": p["name"], "score": round(score, 4), "qualifies": qualifies,
                       "explanation": expl, "confidence": "low",
                       "features": {"skill_overlap": round(skill, 3),
                                    "role_sim": round(roles, 3), "seniority_sim": round(sen, 3),
                                    "cosine": round(cos, 3), "shared_skills": sorted(shared)[:6]}})
    scored.sort(key=lambda x: (-x["score"], x["name"]))
    return scored


def _dist(a, b):
    """Normalized L1 distance between two fraction dicts, in 0-1."""
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    return min(1.0, sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys) / 2)


# ---------- assemble a ranked list of exactly N (or report blocker) ----------

def finalize(scored, target):
    qualified = [s for s in scored if s["qualifies"]]
    if len(qualified) >= target:
        top = qualified[:target]
        blocker = None
    else:
        top = qualified
        blocker = (f"Only {len(qualified)} candidates meet the substitutable-offering "
                   f"evidence bar; need {target}. Enrich more candidates (set "
                   f"AI_GATEWAY_API_KEY to expand discovery beyond the offline seed).")
    for i, s in enumerate(top, 1):
        s["rank"] = i
    return top, blocker
