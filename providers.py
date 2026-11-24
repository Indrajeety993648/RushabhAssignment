"""Interchangeable enrichment/discovery providers (spec items 1, 3).

Order: LLM (AI Gateway, OpenAI-compatible) -> web search -> offline seed.
Web search is the fallback when the LLM is unavailable or returns too few
verified candidates. Offline seed keeps the app runnable with zero credentials.

All raw responses are cached in SQLite by (provider, query, params).
Model id is verified against the live /v1/models list, never trusted from
memory (per project AGENTS.md).
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

import db
import seed

GATEWAY = "https://ai-gateway.vercel.sh/v1"
FIRECRAWL = "https://api.firecrawl.dev/v2/search"
_MODEL_CACHE = {}
_LLM_DEAD = None   # set to the reason string after a hard failure (e.g. 403 billing)


# ---------- LLM provider ----------

def _api_key():
    return os.environ.get("AI_GATEWAY_API_KEY") or os.environ.get("OPENAI_API_KEY")


def llm_available():
    """Key present and not known-dead this process (avoids re-hitting a 403)."""
    return bool(_api_key()) and _LLM_DEAD is None


def _pick_model():
    """Curl the live model list; prefer a current Claude Sonnet, else first."""
    if "id" in _MODEL_CACHE:
        return _MODEL_CACHE["id"]
    req = urllib.request.Request(f"{GATEWAY}/models",
                                 headers={"Authorization": f"Bearer {_api_key()}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    ids = [m["id"] for m in data.get("data", data if isinstance(data, list) else [])]
    pref = [m for m in ids if "claude" in m and "sonnet" in m] or \
           [m for m in ids if "claude" in m] or ids
    if not pref:
        raise RuntimeError("no models available from gateway")
    _MODEL_CACHE["id"] = pref[0]
    return pref[0]


def _llm_json(prompt, params):
    model = _pick_model()
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a company-intelligence analyst. "
             "Reply with ONLY valid JSON matching the requested schema. Use public "
             "evidence; set unknown fields to null; never invent identifiers."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,  # deterministic for reproducible rankings (spec item 7)
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(f"{GATEWAY}/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {_api_key()}",
                                          "Content-Type": "application/json"})
    global _LLM_DEAD
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        # 4xx (bad key, billing, quota) won't fix on retry -> mark dead this process
        if 400 <= e.code < 500:
            _LLM_DEAD = f"HTTP {e.code}: {e.read().decode()[:160]}"
        raise
    return json.loads(resp["choices"][0]["message"]["content"]), model


# ---------- web-search provider (Firecrawl v2) ----------

def _search_key():
    return os.environ.get("FIRECRAWL_API_KEY") or os.environ.get("SEARCH_API_KEY")


def websearch_available():
    return bool(_search_key())


def web_search(con, query, limit=5, scrape=False):
    """Firecrawl /v2/search. Returns list of {title, description, url, markdown}.
    Cached by (query, params) so reruns reuse the raw response (spec item 3)."""
    params = {"limit": limit, "scrape": scrape}
    cached = db.cache_get(con, "websearch", query, params)
    if cached is not None:
        return cached
    body = {"query": query, "limit": limit}
    if scrape:
        body["scrapeOptions"] = {"formats": [{"type": "markdown"}]}
    req = urllib.request.Request(FIRECRAWL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {_search_key()}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.load(r)
    web = (resp.get("data") or {}).get("web", []) if isinstance(resp.get("data"), dict) else []
    hits = [{"title": h.get("title", ""), "description": h.get("description", ""),
             "url": h.get("url", ""), "markdown": h.get("markdown", "")} for h in web]
    db.cache_put(con, "websearch", query, params, hits)
    return hits


# Offering-tag synonyms for structuring free text into the controlled vocabulary.
_OFFER_SYN = {
    "crm": ["crm", "customer relationship management"],
    "erp": ["erp", "enterprise resource planning"],
    "hcm": ["hcm", "human capital", "hr software", "workforce management"],
    "customer-support": ["help desk", "helpdesk", "customer support", "ticketing", "service desk"],
    "project-management": ["project management", "work management", "task management"],
    "collaboration": ["collaboration", "team chat", "messaging", "video meetings", "whiteboard"],
    "productivity": ["productivity", "notes", "documents", "office suite"],
    "content-management": ["content management", "cms", "digital asset"],
    "e-signature": ["e-signature", "esignature", "electronic signature", "contract"],
    "cloud-iaas": ["iaas", "cloud infrastructure", "compute instances", "virtual machines"],
    "cloud-paas": ["paas", "platform as a service", "app platform", "serverless", "edge functions"],
    "database": ["database", "sql", "nosql", "document store"],
    "data-warehouse": ["data warehouse", "lakehouse", "data lake"],
    "data-platform": ["data platform", "data pipeline", "data engineering"],
    "analytics": ["analytics", "business intelligence", "dashboards", "reporting"],
    "observability": ["observability", "monitoring", "apm", "log management", "tracing"],
    "devops": ["devops", "ci/cd", "continuous integration", "pipelines"],
    "source-control": ["source control", "version control", "git", "repositories"],
    "file-storage": ["file storage", "file sync", "cloud storage", "file sharing"],
    "identity": ["identity", "sso", "single sign-on", "mfa", "access management"],
    "endpoint-security": ["endpoint security", "endpoint protection", "edr", "antivirus"],
    "network-security": ["firewall", "network security", "secure web gateway", "sase", "cdn"],
    "cloud-security": ["cloud security", "zero trust", "cspm", "cnapp"],
    "payments": ["payments", "payment processing", "checkout", "billing"],
    "accounting": ["accounting", "bookkeeping", "invoicing", "tax software"],
    "financial-software": ["financial software", "fintech", "treasury", "spend management"],
    "design-tools": ["design tool", "interface design", "ui design", "prototyping"],
    "creative-suite": ["creative", "photo editing", "graphic design", "video editing"],
}
_CAP = re.compile(r"\b([A-Z][A-Za-z0-9&.]+(?:\s+[A-Z][A-Za-z0-9&.]+){0,2})\b")
_NOISE = {"The", "A", "An", "Top", "Best", "Vs", "And", "Or", "For", "Of", "In", "List",
          "Alternatives", "Competitors", "Company", "Companies", "Software", "Compare",
          "Review", "Reviews", "Pricing", "Free", "Home", "Blog", "Products", "Solutions"}


def _structure_from_text(name, text):
    """Deterministically derive offering tags + features from scraped text.
    ponytail: keyword match against the controlled vocab — no LLM needed to turn
    web evidence into a weighted profile. Lower confidence than LLM/seed."""
    low = text.lower()
    offs, counts = [], {}
    for tag, syns in _OFFER_SYN.items():
        c = sum(low.count(s) for s in syns)
        if c:
            counts[tag] = c
    if counts:
        # drop weakly-attested tags (single stray mention) to cut noise
        top = max(counts.values())
        counts = {t: c for t, c in counts.items() if c >= 2 and c >= 0.25 * top}
    if not counts:
        return None  # no verifiable offering evidence -> drop
    top = max(counts.values())
    for tag, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        offs.append({"name": tag, "weight": round(0.4 + 0.6 * c / top, 2),
                     "role": "core" if c >= 0.5 * top else "adjacent"})
    ct = ("B2B2C" if ("b2b" in low or "enterprise" in low) and ("consumer" in low or "b2c" in low)
          else "B2B" if ("b2b" in low or "enterprise" in low or "business" in low)
          else "B2C" if ("consumer" in low or "b2c" in low) else None)
    bm = ("subscription" if "subscription" in low or "saas" in low
          else "transaction" if "transaction" in low or "per-transaction" in low
          else "license" if "license" in low else None)
    geo = ["Global"] if ("global" in low or "worldwide" in low) else []
    return {
        "name": name, "aka": [], "sector": "Technology",
        "industry": offs[0]["name"] if offs else None, "offering": offs[0]["name"] if offs else None,
        "offerings": offs[:8], "capabilities": [o["name"] for o in offs[:5]],
        "customer_type": ct, "business_model": bm, "geography": geo, "scale": {},
        "skills": [], "roles": {}, "seniority": {}, "career_site": None,
        "confidence": 0.4, "evidence": "websearch: " + name,
    }


def _websearch_enrich(con, name):
    hits = web_search(con, f"{name} company — products, offerings, business model", limit=2, scrape=True)
    text = " ".join(f"{h['title']} {h['description']} {h['markdown']}" for h in hits)
    prof = _structure_from_text(name, text) if text.strip() else None
    if prof:
        prof["provider"] = "websearch"
    return prof


def _websearch_discover(con, name):
    """Propose candidate company names from competitor-list search results.
    Noisy by design; each name is re-enriched downstream and dropped if it has
    no verifiable offering evidence (self-verifying)."""
    names, qname = [], name.lower()
    for q in (f"{name} competitors and alternatives", f"companies like {name}"):
        for h in web_search(con, q, limit=6):
            for m in _CAP.findall(f"{h['title']} {h['description']}"):
                cand = m.strip()
                words = cand.split()
                if (len(cand) < 2 or cand.lower() == qname
                        or words[0].lower() == qname.split()[0]  # "Salesforce Competitors"
                        or any(w in _NOISE for w in words)):      # "Pipedrive Best"
                    continue
                names.append(cand)
    seen, out = set(), []
    for nm in names:
        k = nm.lower()
        if k not in seen:
            seen.add(k)
            out.append(nm)
    return out[:10]   # cap paid candidate enrichment on metered keys


# ---------- offline seed provider ----------

def _offline_profile(name, aka):
    rec = seed.SEED.get(name.lower())
    if not rec:
        for a in aka or []:
            rec = seed.SEED.get(a.lower())
            if rec:
                break
    return rec


# ---------- public API ----------

def enrich_company(con, name, aka=None, force=False):
    """Return a normalized profile dict, with provider + confidence. Cached."""
    params = {"taxonomy": seed.TAXONOMY_VERSION, "op": "enrich"}
    if not force:
        cached = db.cache_get(con, "enrich", name, params)
        if cached:
            return cached

    result = None
    # Prefer the free, curated seed before any paid provider (better + $0).
    rec = _offline_profile(name, aka)
    if rec:
        result = dict(rec)
        result["provider"] = "offline"
    if result is None and llm_available():
        try:
            prompt = _enrich_prompt(name)
            raw, model = _llm_json(prompt, params)
            db.cache_put(con, "llm", name, {**params, "model": model}, raw)
            result = _normalize_llm_profile(name, raw)
            result["provider"] = "llm"
        except Exception:  # fall through to next provider
            result = None
    if result is None and websearch_available():
        try:
            result = _websearch_enrich(con, name)
        except Exception:
            result = None
    if result is None:
        result = {"name": name, "provider": "none", "confidence": 0.0,
                  "unavailable": True}

    db.cache_put(con, "enrich", name, params, result)
    return result


def discover_candidates(con, profile, limit=120, allow_web=False):
    """Companies beyond the input to consider. The free seed cluster is always
    included; paid providers (LLM/web) fire only when `allow_web` is set — i.e.
    the caller found the free pool couldn't fill 25 (spec item 4 expansion).
    This keeps routine searches free on metered keys."""
    name = profile["name"]
    params = {"op": "discover", "sector": profile.get("sector"), "web": allow_web}
    cached = db.cache_get(con, "discover", name, params)
    if cached:
        return cached

    names = []
    if allow_web and llm_available():
        try:
            raw, model = _llm_json(_discover_prompt(profile), {**params, "model": "pick"})
            db.cache_put(con, "llm", f"discover:{name}", {**params, "model": model}, raw)
            names = [c.get("name") for c in raw.get("candidates", []) if c.get("name")]
        except Exception:
            names = []
    # web-search fallback when the LLM is unavailable or returned too few
    if allow_web and len(names) < 25 and websearch_available():
        try:
            names += _websearch_discover(con, name)
        except Exception:
            pass
    # always union with offline seed cluster so pool is non-empty
    names += [c["name"] for c in seed.SEED.values()]
    # dedup preserving order, drop the input itself
    seen, out = set(), []
    for nm in names:
        k = db.canonical(nm)
        if k in seen or nm.lower() == name.lower():
            continue
        seen.add(k)
        out.append(nm)
    out = out[:limit]
    db.cache_put(con, "discover", name, params, out)
    return out


def _enrich_prompt(name):
    tags = ", ".join(seed.all_offering_tags())
    return (f'Profile the company "{name}". Return JSON with keys: '
            'sector, industry, offering (primary), '
            'offerings (array of {{name, weight 0-1, role "core"|"adjacent"}} using tags from: '
            f'[{tags}]), capabilities (array), customer_type ("B2B"|"B2C"|"B2B2C"), '
            'business_model, geography (array), scale {{revenue_musd, headcount}}, '
            'skills (array of distinctive hiring skills), roles (object fractions), '
            'seniority (object fractions), career_site, confidence 0-1, evidence.')


def _discover_prompt(profile):
    return (f'List up to 40 real companies that are either direct competitors or '
            f'talent competitors of "{profile["name"]}" (sector {profile.get("sector")}, '
            f'offerings {[o["name"] for o in profile.get("offerings", [])][:6]}). '
            'Return JSON {"candidates":[{"name":...}]}. Real companies only; no duplicates.')


def _normalize_llm_profile(name, raw):
    offs = raw.get("offerings") or []
    norm_offs = [{"name": o.get("name"), "weight": float(o.get("weight", 0.5)),
                  "role": o.get("role", "core")} for o in offs if o.get("name")]
    return {
        "name": name, "aka": raw.get("aka", []),
        "sector": raw.get("sector"), "industry": raw.get("industry"),
        "offering": raw.get("offering"),
        "offerings": norm_offs,
        "capabilities": raw.get("capabilities", []),
        "customer_type": raw.get("customer_type"),
        "business_model": raw.get("business_model"),
        "geography": raw.get("geography", []),
        "scale": raw.get("scale", {}),
        "skills": raw.get("skills", []),
        "roles": raw.get("roles", {}),
        "seniority": raw.get("seniority", {}),
        "career_site": raw.get("career_site"),
        "confidence": float(raw.get("confidence", 0.6)),
        "evidence": raw.get("evidence", "llm"),
    }


if __name__ == "__main__":
    # offline self-check for the web-text parser + candidate extractor (no network)
    p = _structure_from_text("Acme", "Acme is a global B2B SaaS CRM platform. The CRM "
                             "pairs a help desk and ticketing with analytics dashboards and "
                             "reporting. Help desk and CRM plans are subscription-priced.")
    tags = {o["name"] for o in p["offerings"]}
    assert {"crm", "customer-support", "analytics"} <= tags, tags
    assert p["customer_type"] == "B2B" and p["business_model"] == "subscription"
    assert p["geography"] == ["Global"] and p["confidence"] == 0.4
    assert _structure_from_text("X", "nothing relevant here") is None
    # candidate extraction drops noise + the query term
    hits = [{"title": "Top 10 Salesforce Competitors: HubSpot, Zoho CRM, Pipedrive",
             "description": "Best alternatives to Salesforce in 2026", "url": "", "markdown": ""}]
    import unittest.mock as mock
    with mock.patch(__name__ + ".web_search", return_value=hits):
        got = _websearch_discover(None, "Salesforce")
    assert "HubSpot" in got and "Salesforce" not in got and "Top" not in got, got
    print("providers self-check ok:", tags, "| discovered:", got)
