import os, time, json, base64, requests, random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional, Set

import gspread
from google.auth import default as google_auth_default  # ADC via WIF
from openai import OpenAI

# ========= Config via environment (GitHub Secrets) =========
CLIENT_ID      = os.environ["REDDIT_CLIENT_ID"]
CLIENT_SECRET  = os.environ["REDDIT_CLIENT_SECRET"]
USERNAME       = os.environ["REDDIT_USERNAME"]          # Reddit handle, not email
PASSWORD       = os.environ["REDDIT_PASSWORD"]          # Reddit password
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]           # picked up automatically by OpenAI()

# Google Sheets (OPEN BY ID ONLY; no create)
PRODUCT_NAME         = os.getenv("PRODUCT_NAME", "your product")   # framing for the LLM prompts
SHEET_ID             = os.getenv("SHEET_ID", "").strip()            # REQUIRED
SHEET_TAB            = os.getenv("SHEET_TAB", "Requests")           # must exist
SHEET_CLUSTERS_TAB   = os.getenv("SHEET_CLUSTERS_TAB", "Clusters")  # must exist
SHEET_COMPANIES_TAB  = os.getenv("SHEET_COMPANIES_TAB", "Companies")  # must exist

# Tunables (keep in sync with your YAML)
LOOKBACK_DAYS    = int(os.getenv("LOOKBACK_DAYS", "90"))
PAGES_PER_QUERY  = int(os.getenv("PAGES_PER_QUERY", "5"))
LIMIT_PER_PAGE   = int(os.getenv("LIMIT_PER_PAGE", "100"))
EXPANSION_TOPN   = int(os.getenv("EXPANSION_TOPN", "35"))
MAX_CLASSIFY     = int(os.getenv("MAX_CLASSIFY", "2000"))
REQUESTS_PER_MIN = int(os.getenv("REQUESTS_PER_MIN", "40"))
CONF_THRESHOLD   = float(os.getenv("CONF_THRESHOLD", "0.8"))
VERBOSE          = os.getenv("VERBOSE", "0") == "1"

EXTRACT_SNIPPETS_MAX = int(os.getenv("EXTRACT_SNIPPETS_MAX", "200"))
EXTRACT_SNIPPET_LEN = int(os.getenv("EXTRACT_SNIPPET_LEN", "500"))
EXTRACT_DIVERSITY_TEMP = float(os.getenv("EXTRACT_DIVERSITY_TEMP", "0.2"))
EXPANSION_PHRASES = [p.strip() for p in os.getenv("EXPANSION_PHRASES","feature request").split("|") if p.strip()]

TOP_N                = int(os.getenv("TOP_N", "25"))
ENABLE_COMPANY_SCORES = os.getenv("ENABLE_COMPANY_SCORES", "1")
ENABLE_SUB_DISCOVERY  = os.getenv("ENABLE_SUB_DISCOVERY", "0")
SUBS_PER_COMPANY      = int(os.getenv("SUBS_PER_COMPANY", "3"))
SUB_CRAWL_PAGES_MAX   = int(os.getenv("SUB_CRAWL_PAGES_MAX", "2"))
CLUSTER_PROMPT_LIMIT  = int(os.getenv("CLUSTER_PROMPT_LIMIT", "0"))  # 0 = no limit

USER_AGENT = f"script:reddit-feature-harvester:1.1 (by /u/{USERNAME})"
BASE_AUTH  = "https://www.reddit.com"
BASE_API   = "https://oauth.reddit.com"
SORT = "new"
TIME_RANGE = os.getenv("REDDIT_TIME_RANGE", "year")  # "year" safely contains 90 days

# Optional feature-fit scoring
ENABLE_FIT_SCORING = os.getenv("ENABLE_FIT_SCORING", "0")
if ENABLE_FIT_SCORING != "1":
    print(
        "[WARN] ENABLE_FIT_SCORING != '1'; fit_type, fit_score, and fit_rationale will remain blank.",
        flush=True,
    )
FIT_MODEL = os.getenv("FIT_MODEL", "gpt-4o-mini")
# Optional canonical company lookup
ENABLE_CANONICAL_COMPANY = os.getenv("ENABLE_CANONICAL_COMPANY", "0")

# Broader seed phrases (recall ↑)
SEED_QUERIES = [
    "feature request",
    "please add",
    "missing feature",
    "roadmap request",
    "support for",
    "integration with",
    "ability to",
    "wishlist",
    "would love",
    "i wish",
    "could you add",
    "add option to",
    "add support for",
    "add integration",
    "please support",
    "feature suggestion",
    "rfe",
    "fr:",
    "new feature",
]

PHRASE_CACHE_FILE = os.getenv("PHRASE_CACHE_FILE", "phrase_cache.json")

STATE_FILE = os.getenv("STATE_FILE", "state.json")

SUMMARY_FILE = os.getenv("SUMMARY_FILE", "docs/data/summary.json")

COMPANY_CACHE_FILE = os.getenv("COMPANY_CACHE_FILE", "companies.json")
COMPANY_EXTRACT_TOPN = int(os.getenv("COMPANY_EXTRACT_TOPN", "0"))

SUPPORT_OUTAGE_REVIEW_FILE = os.getenv(
    "SUPPORT_OUTAGE_REVIEW_FILE", "support_outage_review.json"
)


def _load_state() -> Dict[str, Dict]:
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state: Dict[str, Dict]) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

SEED_SYNONYMS = {
    "feature request": ["feature suggestion", "feature idea"],
    "please add": ["please include", "kindly add"],
    "missing feature": ["lacking feature", "no feature"],
    "roadmap request": ["feature on roadmap"],
    "support for": ["supporting", "allow for"],
}

def _load_phrase_cache() -> Set[str]:
    try:
        with open(PHRASE_CACHE_FILE, "r") as f:
            return set(json.load(f))
    except Exception:
        return set()


def _save_phrase_cache(cache: Set[str]) -> None:
    with open(PHRASE_CACHE_FILE, "w") as f:
        json.dump(sorted(cache), f, indent=2)


def _load_company_cache() -> Set[str]:
    try:
        with open(COMPANY_CACHE_FILE, "r") as f:
            return set(json.load(f))
    except Exception:
        return set()


def _save_company_cache(names: Set[str]) -> None:
    with open(COMPANY_CACHE_FILE, "w") as f:
        json.dump(sorted(names), f, indent=2)


def expand_seed_queries(base: List[str], max_new: int = 5) -> List[str]:
    """Propose extra seed phrases via embeddings or stored synonyms."""
    cache = _load_phrase_cache()
    existing = {q.lower() for q in base} | cache
    candidates: List[str] = []
    for q in base:
        for syn in SEED_SYNONYMS.get(q.lower(), []):
            if syn.lower() not in existing:
                candidates.append(syn)

    # Rank candidates by similarity to first base query using embeddings
    try:
        if candidates:
            base_emb = client.embeddings.create(
                model="text-embedding-3-small", input=base[0]
            ).data[0].embedding
            cand_embs = client.embeddings.create(
                model="text-embedding-3-small", input=candidates
            ).data

            def cosine(a: List[float], b: List[float]) -> float:
                dot = sum(x * y for x, y in zip(a, b))
                na = sum(x * x for x in a) ** 0.5
                nb = sum(x * x for x in b) ** 0.5
                return dot / (na * nb or 1.0)

            candidates = [
                c
                for c, e in sorted(
                    zip(candidates, cand_embs),
                    key=lambda pair: cosine(base_emb, pair[1].embedding),
                    reverse=True,
                )
            ]
    except Exception:
        pass

    new_phrases: List[str] = []
    for cand in candidates:
        if len(new_phrases) >= max_new:
            break
        if cand.lower() not in existing:
            new_phrases.append(cand)
            existing.add(cand.lower())

    if new_phrases:
        cache |= {p.lower() for p in new_phrases}
        _save_phrase_cache(cache)
    return new_phrases

FIT_TYPE_GUARDRAILS = {
    "ui_tweak": "Minor UI or workflow adjustments",
    "integration": "Connecting with external services or APIs",
    "data_join": "Combining or syncing data across systems",
    "core_change": "Fundamental change to the core product",
    "compliance": "Driven by regulatory or policy requirements",
    "other": "Does not fit the above types",
}

def log(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)

# ----- simple token-bucket pacing to stay under Reddit limits -----
_MIN_INTERVAL = 60.0 / max(1, REQUESTS_PER_MIN)
_last_call_ts = [0.0]

def _pace():
    now = time.time()
    wait = _MIN_INTERVAL - (now - _last_call_ts[0])
    if wait > 0:
        time.sleep(wait)
    _last_call_ts[0] = time.time()

# ----- Reddit OAuth + calls -----
def _basic_auth_header(cid: str, csec: str) -> str:
    raw = f"{cid}:{csec}".encode("ascii")
    return "Basic " + base64.b64encode(raw).decode("ascii")

def oauth_token() -> Tuple[str, int]:
    url = f"{BASE_AUTH}/api/v1/access_token"
    headers = {
        "Authorization": _basic_auth_header(CLIENT_ID, CLIENT_SECRET),
        "User-Agent": USER_AGENT,
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    data = {
        "grant_type": "password",
        "username": USERNAME,
        "password": PASSWORD,
        "scope": "read",
    }
    log("[AUTH] Requesting Reddit OAuth token…")
    r = requests.post(url, headers=headers, data=data, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Token error {r.status_code}: {r.text}")
    js = r.json()
    log("[AUTH] Token received.")
    return js["access_token"], js.get("expires_in", 3600)

def reddit_get(path: str, params: Dict, token: str, tries: int = 3) -> requests.Response:
    url = f"{BASE_API}{path}"
    headers = {
        "Authorization": f"bearer {token}",
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    last: Optional[requests.Response] = None
    for attempt in range(tries):
        _pace()  # global rate cap
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        last = resp
        rl_rem = resp.headers.get("x-ratelimit-remaining")
        rl_used = resp.headers.get("x-ratelimit-used")
        rl_reset = resp.headers.get("x-ratelimit-reset")
        log(f"[HTTP] GET {path} ? {params} → {resp.status_code} "
            f"(rl_remaining={rl_rem}, rl_used={rl_used}, rl_reset={rl_reset})")
        if resp.status_code == 200:
            return resp
        if resp.status_code in (429, 502, 503, 504):
            reset_hdr = resp.headers.get("x-ratelimit-reset")
            wait_s = int(float(reset_hdr)) if reset_hdr else min(30, (2 ** attempt) * 2)
            log(f"[HTTP] Backing off {wait_s}s (status {resp.status_code}, attempt {attempt+1}/{tries})")
            time.sleep(wait_s)
            continue
        sleep_s = (2 ** attempt) + 1
        log(f"[HTTP] Non-200 ({resp.status_code}). Sleeping {sleep_s}s then retry…")
        time.sleep(sleep_s)
    raise RuntimeError(f"GET {url} failed after retries: {last.status_code} {last.text}")

def search_sitewide(
    query: str,
    token: str,
    pages: int = PAGES_PER_QUERY,
    limit: int = LIMIT_PER_PAGE,
    after: Optional[str] = None,
) -> List[Dict]:
    out: List[Dict] = []
    cursor = after
    for page in range(pages):
        params = {
            "q": query,
            "sort": SORT,
            "t": TIME_RANGE,
            "restrict_sr": "false",
            "limit": str(limit),
        }
        if cursor:
            params["after"] = cursor
        r = reddit_get("/search", params, token)
        js = r.json()
        children = (js.get("data") or {}).get("children") or []
        if not children:
            break
        out.extend([c.get("data") for c in children if isinstance(c, dict)])
        cursor = (js.get("data") or {}).get("after")
        log(
            f"[SEARCH] Query='{query}' page={page+1}/{pages} batch={len(children)} total={len(out)} after={cursor}"
        )
        if not cursor:
            break
        time.sleep(1.2)  # small courtesy delay
    return out

def search_company_expanded(
    name: str, token: str, after_map: Optional[Dict[str, Dict]] = None
) -> List[Dict]:
    """Search for a company name combined with each expansion phrase.

    Results from different phrases are de-duped by Reddit fullname.
    """
    merged: Dict[str, Dict] = {}
    for phrase in EXPANSION_PHRASES:
        query = f"{name} {phrase}"
        after = None
        if after_map:
            after = (after_map.get(query) or {}).get("fullname")
        posts = search_sitewide(query, token, after=after)
        if posts and after_map is not None:
            after_map[query] = {
                "fullname": posts[0].get("name"),
                "created_utc": posts[0].get("created_utc"),
            }
        for post in posts:
            fn = (post or {}).get("name")
            if fn and fn not in merged:
                merged[fn] = post
    return list(merged.values())

def discover_subreddits(name: str, token: str, k: int) -> List[str]:
    """Discover subreddit names related to a company."""
    params = {"q": name, "limit": str(k)}
    r = reddit_get("/subreddits/search", params, token)
    js = r.json()
    children = (js.get("data") or {}).get("children") or []
    subs = []
    for c in children:
        sub = ((c.get("data") or {}).get("display_name") or "").strip()
        if sub:
            subs.append(sub)
        if len(subs) >= k:
            break
    return subs

def crawl_subreddit_new(sub: str, token: str, since_ts: float) -> List[Dict]:
    """Crawl /r/{sub}/new up to SUB_CRAWL_PAGES_MAX pages until posts are older than since_ts."""
    out, after = [], None
    for _ in range(SUB_CRAWL_PAGES_MAX):
        params = {"limit": str(LIMIT_PER_PAGE)}
        if after:
            params["after"] = after
        r = reddit_get(f"/r/{sub}/new", params, token)
        js = r.json()
        children = (js.get("data") or {}).get("children") or []
        if not children:
            break
        for c in children:
            data = c.get("data") or {}
            created = data.get("created_utc") or 0
            if created < since_ts:
                return out
            out.append(data)
        after = (js.get("data") or {}).get("after")
        if not after:
            break
    return out

def normalize_posts(raw_list: List[Dict]) -> List[Dict]:
    norm = []
    for d in raw_list or []:
        if not d:
            continue
        norm.append({
            "id": d.get("id"),
            "fullname": d.get("name"),
            "permalink": f"https://www.reddit.com{d.get('permalink','')}",
            "subreddit": d.get("subreddit"),
            "author": d.get("author"),
            "title": d.get("title", ""),
            "selftext": d.get("selftext", ""),
            "created_utc": d.get("created_utc") or 0,
            "ups": d.get("ups", 0),
            "num_comments": d.get("num_comments", 0),
            "flair": d.get("link_flair_text"),
        })
    return norm

def filter_lookback(items: List[Dict], days: int = LOOKBACK_DAYS) -> List[Dict]:
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
    return [p for p in items if (p.get("created_utc") or 0) >= cutoff_ts]

SUPPORT_OUTAGE_KEYWORDS = [
    "support ticket",
    "support request",
    "outage",
    "down",
    "bug",
    "issue",
    "error",
    "not working",
    "can't log in",
    "cannot login",
]


def is_support_or_outage(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in SUPPORT_OUTAGE_KEYWORDS)

# ---------- OpenAI ----------
client = OpenAI()  # OPENAI_API_KEY from env

def llm_extract_companies(posts: List[Dict], topn: Optional[int] = None) -> List[str]:
    snippets = []
    for p in posts[:EXTRACT_SNIPPETS_MAX]:
        text = f"{p.get('title','')} {p.get('selftext','')}"
        snippets.append(text[:EXTRACT_SNIPPET_LEN])
    prompt = (
        "From the following Reddit post snippets, extract a deduplicated list of company or product names "
        "that users are asking to add or change a feature for. Return ONLY a JSON array of strings.\n\n"
        + "\n---\n".join(snippets)
        + "\n\nReturn JSON only, e.g.: [\"Notion\", \"Slack\"]"
    )
    log(f"[LLM] Extracting companies from {len(snippets)} snippets…")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=EXTRACT_DIVERSITY_TEMP,
    )
    text = resp.choices[0].message.content.strip()
    try:
        names = json.loads(text)
        clean, seen = [], set()
        for n in names:
            n = (n or "").strip()
            if n and len(n) <= 40 and n.lower() not in seen:
                seen.add(n.lower())
                clean.append(n)
        limit = topn if topn is not None else COMPANY_EXTRACT_TOPN
        out = clean if limit <= 0 else clean[:limit]
        if out:
            existing = _load_company_cache()
            existing.update(out)
            _save_company_cache(existing)
        log(f"[LLM] Extracted {len(out)} company/product names.")
        return out
    except Exception:
        log("[LLM] Failed to parse company list JSON; continuing with none.")
        return []

def _stratified_posts(posts: List[Dict], known: Set[str], k: int) -> List[Dict]:
    if not posts:
        return []
    known_lower = {n.lower() for n in known}
    new_posts, seen_posts = [], []
    for p in posts:
        text = f"{p.get('title','')} {p.get('selftext','')}".lower()
        if any(n in text for n in known_lower):
            seen_posts.append(p)
        else:
            new_posts.append(p)
    sample: List[Dict] = []
    new_quota = min(len(new_posts), k // 2 if k > 1 else k)
    if new_posts:
        sample.extend(random.sample(new_posts, new_quota))
    remaining = k - len(sample)
    pool = seen_posts if seen_posts else new_posts
    if remaining > 0 and pool:
        take = min(len(pool), remaining)
        sample.extend(random.sample(pool, take))
    if len(sample) < k:
        leftover = [p for p in posts if p not in sample]
        if leftover:
            sample.extend(random.sample(leftover, min(len(leftover), k - len(sample))))
    return sample


def llm_extract_companies_diverse(posts: List[Dict], snippets_max: int, snippet_len: int, temp: float) -> List[str]:
    known = _load_company_cache()
    sample = _stratified_posts(posts, known, min(len(posts), snippets_max))
    snippets = []
    for p in sample:
        text = f"{p.get('title','')} {p.get('selftext','')}"
        snippets.append(text[:snippet_len])
    prompt = (
        "From the following Reddit post snippets, extract a diverse set of B2B SaaS company or product names "
        "that users are asking to add or change a feature for. Prefer variety across industries and avoid regulated "
        "verticals such as healthcare, finance, or insurance. Return ONLY a JSON array of strings.\n\n"
        + "\n---\n".join(snippets)
        + "\n\nReturn JSON only, e.g.: [\"Notion\", \"Slack\"]"
    )
    log(f"[LLM] Extracting diverse companies from {len(snippets)} snippets…")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    text = resp.choices[0].message.content.strip()
    try:
        names = json.loads(text)
        clean, seen = [], set()
        for n in names:
            n = (n or "").strip()
            if n and len(n) <= 40 and n.lower() not in seen:
                seen.add(n.lower())
                clean.append(n)
        limit = COMPANY_EXTRACT_TOPN
        out = clean if limit <= 0 else clean[:limit]
        if out:
            existing = known | set(out)
            _save_company_cache(existing)
        log(f"[LLM] Extracted {len(out)} unique companies/products.")
        return out
    except Exception:
        log("[LLM] Failed to parse company list JSON; continuing with none.")
        return []

def llm_classify(post: Dict) -> Dict:
    prompt = f"""
Decide if this Reddit post contains a genuine SaaS FEATURE REQUEST.
Return STRICT JSON: {{"is_feature_request": true/false, "company_guess": string|null, "feature_summary": string, "confidence": number (0-1)}}

Title: {post.get('title','')}
Body: {post.get('selftext','')}
Subreddit: {post.get('subreddit')}
Flair: {post.get('flair')}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    txt = resp.choices[0].message.content.strip()
    try:
        data = json.loads(txt)
        return {
            "is_feature_request": bool(data.get("is_feature_request", False)),
            "company_guess": data.get("company_guess"),
            "feature_summary": data.get("feature_summary", "")[:300],
            "confidence": float(data.get("confidence", 0.0)),
        }
    except Exception:
        return {"is_feature_request": False, "company_guess": None, "feature_summary": "", "confidence": 0.0}

def llm_fit_score(post: Dict) -> Dict:
    fit_map_str = "\n".join([f"- {k}: {v}" for k, v in FIT_TYPE_GUARDRAILS.items()])
    prompt = f"""
The following Reddit post has been identified as a SaaS feature request.
Classify how well the request fits into the product and explain why.

Fit types and meanings:
{fit_map_str}

Return STRICT JSON: {{"fit_type": "ui_tweak|integration|data_join|core_change|compliance|other", "fit_score": number 0-1, "fit_rationale": string}}

Title: {post.get('title','')}
Body: {post.get('selftext','')}
Company: {post.get('company_guess')}
Feature summary: {post.get('feature_summary','')}
"""
    resp = client.chat.completions.create(
        model=FIT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    txt = resp.choices[0].message.content.strip()
    try:
        data = json.loads(txt)
        ftype = (data.get("fit_type") or "").strip().lower()
        if ftype not in FIT_TYPE_GUARDRAILS:
            ftype = "other"
        return {
            "fit_type": ftype,
            "fit_score": float(data.get("fit_score", 0.0)),
            "fit_rationale": (data.get("fit_rationale") or "")[:500],
        }
    except Exception:
        return {"fit_type": "other", "fit_score": 0.0, "fit_rationale": ""}

def canonicalize_company(raw: str) -> Dict:
    raw = (raw or "").strip()
    if not raw:
        return {"canonical_company": "", "company_domain": ""}
    prompt = f"""Given a company or product name guess, return the canonical company name and its primary web domain if known.

Return STRICT JSON: {{"canonical_company": string, "company_domain": string}}

Name: {raw}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    txt = resp.choices[0].message.content.strip()
    try:
        data = json.loads(txt)
        return {
            "canonical_company": (data.get("canonical_company") or "")[:100],
            "company_domain": (data.get("company_domain") or "")[:100],
        }
    except Exception:
        return {"canonical_company": "", "company_domain": ""}

# ---------- Clustering ----------
def llm_cluster_features(posts: List[Dict]) -> Optional[List[Dict]]:
    """
    Cluster feature requests into themes. Returns a list of cluster dicts or
    ``None`` if the model output cannot be parsed:
    [
      {
        "name": "Kill Switch / Global Sign-Out",
        "summary": "Single control to sign out all devices/sessions",
        "member_ids": ["1n6qi4w", "...."],
        "companies": ["ProtonMail","..."]
      },
      ...
    ]
    """
    items = []
    for p in posts:
        fs = (p.get("feature_summary") or "").strip()
        if not fs:
            continue
        items.append({
            "id": p.get("id"),
            "company": (p.get("company") or "").strip(),
            "summary": fs
        })
    if not items:
        return []

    if CLUSTER_PROMPT_LIMIT > 0 and len(items) > CLUSTER_PROMPT_LIMIT:
        log(
            f"[LLM] Limiting clustering to first {CLUSTER_PROMPT_LIMIT} posts (of {len(items)}) to avoid truncation."
        )
        items = items[:CLUSTER_PROMPT_LIMIT]

    # Prepare a compact list for the prompt
    lines = [f"{it['id']}|||{it['company']}|||{it['summary']}" for it in items]
    prompt = (
        "Cluster these SaaS feature request summaries into themes where the meaning is the same or very similar.\n"
        "Each line is: ID|||Company|||Summary\n\n"
        + "\n".join(lines) +
        "\n\nReturn STRICT JSON as an array of clusters with this schema:\n"
        "[\n"
        "  {\n"
        "    \"name\": \"short cluster name\",\n"
        "    \"summary\": \"one-line canonical description\",\n"
        "    \"member_ids\": [\"<id>\", \"<id>\"],\n"
        "    \"companies\": [\"CompanyA\", \"CompanyB\"]\n"
        "  }\n"
        "]\n"
        "Return ONLY JSON—no prose."
    )

    log(f"[LLM] Clustering {len(items)} feature summaries…")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    txt = resp.choices[0].message.content.strip()
    try:
        clusters = json.loads(txt)
        # basic shape validation
        out = []
        for c in clusters if isinstance(clusters, list) else []:
            name = (c.get("name") or "").strip()
            member_ids = c.get("member_ids") or []
            summary = (c.get("summary") or "").strip()
            companies = list({(x or "").strip() for x in (c.get("companies") or []) if (x or "").strip()})
            if name and isinstance(member_ids, list) and member_ids:
                out.append({
                    "name": name[:200],
                    "summary": summary[:500],
                    "member_ids": [m for m in member_ids if m],
                    "companies": companies[:20]
                })
        log(f"[LLM] Produced {len(out)} clusters.")
        return out
    except Exception:
        log(f"[LLM] Failed to parse clusters JSON; raw response:\n{txt}")
        return None

def llm_cluster_pitch(name: str, summary: str) -> str:
    """Generate a one-sentence pitch for how {PRODUCT_NAME} could address this cluster."""
    prompt = (
        "You are brainstorming product ideas for {PRODUCT_NAME}. "
        "Given the following feature request theme, describe in one sentence "
        "how {PRODUCT_NAME} could address this type of request.\n\n"
        f"Cluster Name: {name}\n"
        f"Summary: {summary}\n\n"
        "One-sentence pitch:"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip().replace("\n", " ")

def llm_region_hint(company: str) -> str:
    """Guess whether a company is primarily US, EMEA, or Other."""
    prompt = (
        "Based on the company name provided, guess whether the company is "
        "primarily based in the US, EMEA, or Other regions. "
        "Return STRICT JSON like {\"region\": \"US|EMEA|Other\"}.\n\n"
        f"Company: {company}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        txt = resp.choices[0].message.content.strip()
        data = json.loads(txt)
        region = (data.get("region") or "Other").strip()
        if region not in {"US", "EMEA"}:
            region = "Other"
        return region
    except Exception:
        return "Other"

def compute_company_scores(request_rows: List[Dict], cluster_rows: List[Dict], top_n: int = TOP_N) -> List[Dict]:
    """Aggregate request and cluster data into per-company opportunity scores."""
    now_ts = time.time()
    lookback_sec = LOOKBACK_DAYS * 86400
    companies: Dict[str, Dict] = {}

    for r in request_rows:
        comp = r.get("company")
        if not comp:
            continue
        entry = companies.setdefault(comp, {
            "count": 0,
            "confidence_total": 0.0,
            "ups_comments": 0,
            "latest_ts": 0.0,
            "domain": r.get("company_domain") or "",
            "fit_scores": [],
            "clusters": {},
        })
        entry["count"] += 1
        entry["confidence_total"] += float(r.get("confidence") or 0.0)
        entry["ups_comments"] += int(r.get("ups") or 0) + int(r.get("num_comments") or 0)
        ts = float(r.get("created_utc") or 0.0)
        if ts > entry["latest_ts"]:
            entry["latest_ts"] = ts
        fit = r.get("fit_score")
        if isinstance(fit, (int, float)):
            entry["fit_scores"].append(float(fit))

    for c in cluster_rows:
        cname = c.get("name")
        cf_score = c.get("cluster_fit_score") or 0.0
        comp_counts = c.get("company_member_counts", {}) or {}
        for comp, cnt in comp_counts.items():
            entry = companies.setdefault(comp, {
                "count": 0,
                "confidence_total": 0.0,
                "ups_comments": 0,
                "latest_ts": 0.0,
                "domain": "",
                "fit_scores": [],
                "clusters": {},
            })
            entry["clusters"].setdefault(cname, {"size": cnt, "fit": cf_score})

    if not companies:
        return []

    max_posts = max(v["count"] for v in companies.values()) or 1
    max_clusters = max(len(v["clusters"]) for v in companies.values()) or 1
    max_eng = max(v["ups_comments"] for v in companies.values()) or 1

    out = []
    for comp, data in companies.items():
        avg_conf = data["confidence_total"] / data["count"] if data["count"] else 0.0
        volume_posts = data["count"] / max_posts
        volume_clusters = len(data["clusters"]) / max_clusters
        volume_score = 0.5 * (volume_posts + volume_clusters)
        engagement_score = data["ups_comments"] / max_eng if max_eng else 0.0
        recency_score = 0.0
        if data["latest_ts"]:
            recency_score = max(0.0, 1.0 - (now_ts - data["latest_ts"]) / lookback_sec)
        fit_score = sum(data["fit_scores"]) / len(data["fit_scores"]) if data["fit_scores"] else 0.0
        region_hint = llm_region_hint(comp)
        region_score = 1.0 if region_hint in ("US", "EMEA") else 0.0
        opp = (
            volume_score * 0.35 +
            engagement_score * 0.10 +
            recency_score * 0.10 +
            fit_score * 0.40 +
            region_score * 0.05
        ) * 100.0
        top_clusters = sorted(
            data["clusters"].items(),
            key=lambda kv: kv[1]["size"],
            reverse=True,
        )
        top_feature = top_clusters[0][0] if top_clusters else ""
        top_feature_count = top_clusters[0][1]["size"] if top_clusters else 0
        out.append({
            "company": comp,
            "domain": data.get("domain", ""),
            "count": data["count"],
            "avg_confidence": avg_conf,
            "top_clusters": [kv[0] for kv in top_clusters[:3]],
            "top_feature": top_feature,
            "top_feature_count": top_feature_count,
            "opportunity_score": opp,
            "region_hint": region_hint,
            "status": "",
        })

    out.sort(key=lambda x: x["opportunity_score"], reverse=True)
    return out[:top_n]

# ---------- Google Sheets via ADC (no creation) ----------
def gspread_client_from_env():
    creds, _ = google_auth_default(scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ])
    return gspread.authorize(creds)

def open_existing_sheet_and_tab(gc, sheet_id: str, tab: str):
    if not sheet_id:
        raise RuntimeError(
            "SHEET_ID is not set. This program will not create files to avoid Drive quota errors.\n"
            "Set the repo secret SHEET_ID to your spreadsheet ID."
        )
    log(f"[SHEETS] Opening spreadsheet by ID: {sheet_id}")
    try:
        sh = gc.open_by_key(sheet_id)
    except Exception as e:
        raise RuntimeError(
            f"Could not open spreadsheet by SHEET_ID '{sheet_id}'. "
            f"Ensure the service account has Editor access. Error: {e}"
        )
    try:
        ws = sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        raise RuntimeError(
            f"Worksheet/tab '{tab}' not found in the spreadsheet. "
            f"Please create a tab named '{tab}' manually, then re-run."
        )
    return sh, ws

def read_existing_ids(ws) -> set:
    vals = ws.col_values(1)
    return set(v.strip() for v in vals if v and v.strip() != "id")

def append_rows(ws, rows: List[List[str]]):
    if rows:
        log(f"[SHEETS] Appending {len(rows)} new rows…")
        ws.append_rows(rows, value_input_option="RAW")

def ensure_request_headers(ws):
    """Ensure the Requests tab has canonical and fit scoring columns."""
    expected = [
        "id",
        "permalink",
        "subreddit",
        "author",
        "title",
        "created_utc",
        "ups",
        "num_comments",
        "feature_summary",
        "company_guess",
        "canonical_company",
        "company_domain",
        "confidence",
        "fit_type",
        "fit_score",
        "fit_rationale",
        "harvested_at_utc",
    ]
    current = ws.row_values(1)
    if [h.strip().lower() for h in current] != expected:
        ws.update('1:1', [expected])

def clear_and_write_clusters(ws, rows: List[List[str]]):
    """
    Overwrite the existing Clusters tab (no creation).
    """
    log(f"[SHEETS] Writing {len(rows)} cluster rows (overwrite).")
    ws.clear()
    header = [
        "cluster_name",
        "feature_summary",
        "request_count",
        "example_ids",
        "companies_mentioned",
        "cluster_fit_score",
        "cluster_pitch",
        "last_seen_utc",
    ]
    ws.append_row(header, value_input_option="RAW")
    if rows:
        ws.append_rows(rows, value_input_option="RAW")

def clear_and_write_companies(ws, rows: List[List[str]]):
    """Overwrite the existing Companies tab with new scores."""
    log(f"[SHEETS] Writing {len(rows)} company rows (overwrite).")
    ws.clear()
    header = [
        "company",
        "domain",
        "90d_request_count",
        "avg_confidence",
        "top_feature",
        "top_feature_count",
        "top_clusters",
        "opportunity_score",
        "region_hint",
        "status",
    ]
    ws.append_row(header, value_input_option="RAW")
    if rows:
        ws.append_rows(rows, value_input_option="RAW")

# ---------- main ----------
def main():
    start = time.time()
    token, _ = oauth_token()

    state = _load_state()

    extra_seeds = expand_seed_queries(SEED_QUERIES, max_new=5)
    if extra_seeds:
        log(f"[STAGE] Expanded seed queries: {extra_seeds}")
        SEED_QUERIES.extend(extra_seeds)

    # Seed searches
    log(f"[STAGE] Seed search across {len(SEED_QUERIES)} queries…")
    all_raw: List[Dict] = []
    for q in SEED_QUERIES:
        after = (state.get(q) or {}).get("fullname")
        posts = search_sitewide(q, token, after=after)
        if posts:
            state[q] = {
                "fullname": posts[0].get("name"),
                "created_utc": posts[0].get("created_utc"),
            }
        all_raw.extend(posts)
    seeds = filter_lookback(normalize_posts(all_raw), LOOKBACK_DAYS)
    log(f"[STAGE] Seeds fetched (lookback {LOOKBACK_DAYS}d): {len(seeds)}")

    # Expansion
    companies = llm_extract_companies_diverse(
        seeds, EXTRACT_SNIPPETS_MAX, EXTRACT_SNIPPET_LEN, EXTRACT_DIVERSITY_TEMP
    )
    persisted = _load_company_cache()
    merged_companies = sorted(set(persisted) | set(companies))
    if merged_companies:
        _save_company_cache(set(merged_companies))
    log(f"[STAGE] Unique companies extracted: {len(companies)} (merged {len(merged_companies)})")
    expanded_raw: Dict[str, Dict] = {}
    for name in merged_companies:
        hits = search_company_expanded(name, token, state)
        log(f"[STAGE] Expansion hits for {name}: {len(hits)}")
        for p in hits:
            fn = (p or {}).get("name")
            if fn and fn not in expanded_raw:
                expanded_raw[fn] = p
    log(f"[STAGE] Total expansion hits (unique): {len(expanded_raw)}")

    expanded = filter_lookback(normalize_posts(list(expanded_raw.values())), LOOKBACK_DAYS)
    log(f"[STAGE] Expanded posts (lookback {LOOKBACK_DAYS}d): {len(expanded)}")

    # Merge seeds and expansion hits by fullname
    merged = {p["fullname"]: p for p in (seeds + expanded) if p.get("fullname")}
    log(f"[STAGE] Candidates after seed+expansion merge: {len(merged)}")

    sub_posts = []
    if ENABLE_SUB_DISCOVERY == "1":
        since_ts = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp()
        total_subreddits = 0
        sub_posts_raw: List[Dict] = []
        for name in merged_companies:
            subs = discover_subreddits(name, token, SUBS_PER_COMPANY)
            total_subreddits += len(subs)
            log(f"[PH5] {name}: {len(subs)} subreddits discovered")
            for sub in subs:
                posts = crawl_subreddit_new(sub, token, since_ts)
                log(f"[PH5] r/{sub}: {len(posts)} posts fetched")
                sub_posts_raw.extend(posts)
        log(f"[PH5] Total subreddits discovered: {total_subreddits}")
        sub_posts = filter_lookback(normalize_posts(sub_posts_raw), LOOKBACK_DAYS)
        log(f"[PH5] Total additional candidates from subreddits: {len(sub_posts)}")
        # Merge subreddit posts into existing dictionary to avoid duplicates
        before_merge = len(merged)
        for p in sub_posts:
            fn = p.get("fullname")
            if fn and fn not in merged:
                merged[fn] = p
        added = len(merged) - before_merge
        log(f"[PH5] Final deduped candidate total: {len(merged)} (+{added} from subreddits)")

    candidates = list(merged.values())
    log(f"[STAGE] Final unique candidates: {len(candidates)}")
    candidates = candidates[:MAX_CLASSIFY]
    log(f"[STAGE] Candidates capped to: {len(candidates)} (MAX_CLASSIFY={MAX_CLASSIFY})")

    # Classify (confidence filter)
    results = []
    skipped_support: List[Dict] = []
    total = len(candidates)
    for i, p in enumerate(candidates, 1):
        if i % 25 == 0 or i == total:
            log(f"[LLM] Classified {i}/{total}")
        text = f"{p.get('title','')} {p.get('selftext','')}"
        if is_support_or_outage(text):
            skipped_support.append(
                {
                    "id": p.get("id"),
                    "title": p.get("title", ""),
                    "selftext": p.get("selftext", ""),
                }
            )
            continue
        cls = llm_classify(p)
        if cls["is_feature_request"] and cls.get("confidence", 0.0) >= CONF_THRESHOLD:
            r = {**p, **cls}
            if ENABLE_FIT_SCORING == "1":
                fit = llm_fit_score(r)
                r.update(fit)
            else:
                r.update({"fit_type": "", "fit_score": "", "fit_rationale": ""})
            if ENABLE_CANONICAL_COMPANY == "1":
                canon = canonicalize_company(r.get("company_guess"))
                r.update(canon)
            else:
                r.update({"canonical_company": "", "company_domain": ""})
            results.append(r)
    log(f"[STAGE] Feature requests detected (conf≥{CONF_THRESHOLD:.2f}): {len(results)}")
    if skipped_support:
        try:
            with open(SUPPORT_OUTAGE_REVIEW_FILE, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []
        existing.extend(skipped_support)
        with open(SUPPORT_OUTAGE_REVIEW_FILE, "w") as f:
            json.dump(existing, f, indent=2)
        log(
            f"[STAGE] Support/outage posts skipped: {len(skipped_support)} (saved for review)"
        )

    # Rank
    results.sort(key=lambda x: (x.get("ups", 0) + x.get("num_comments", 0), x.get("created_utc", 0)), reverse=True)

    # Sheets append only NEW (no create)
    gc = gspread_client_from_env()
    sh, ws_requests = open_existing_sheet_and_tab(gc, SHEET_ID, SHEET_TAB)
    ensure_request_headers(ws_requests)
    _, ws_clusters = open_existing_sheet_and_tab(gc, SHEET_ID, SHEET_CLUSTERS_TAB)
    existing = read_existing_ids(ws_requests)
    now_iso = datetime.now(timezone.utc).isoformat()

    new_rows = []
    for r in results:
        if r["id"] in existing:
            continue
        fit_score = r.get("fit_score")
        fit_score_str = f"{fit_score:.2f}" if isinstance(fit_score, (int, float)) else ""
        row = [
            r["id"], r["permalink"], r.get("subreddit"), r.get("author"),
            r.get("title","")[:49000],
            str(r.get("created_utc",0)), str(r.get("ups",0)), str(r.get("num_comments",0)),
            r.get("feature_summary","")[:49000],
            r.get("company_guess") or "",
            r.get("canonical_company") or "",
            r.get("company_domain") or "",
            f"{r.get('confidence',0):.2f}",
        ]
        row.extend([
            r.get("fit_type",""),
            fit_score_str,
            r.get("fit_rationale","")[:49000],
        ])
        row.append(now_iso)
        new_rows.append(row)
    append_rows(ws_requests, new_rows)

    _save_state(state)

    # ----- Clustering (works on ALL current Requests rows, not just new) -----
    # Re-read all rows from Requests to cluster the full set
    all_values = ws_requests.get_all_values()
    # Expect header row present; find column indices
    if all_values and all_values[0]:
        header = [h.strip().lower() for h in all_values[0]]
        def idx(colname, default=-1):
            try:
                return header.index(colname)
            except ValueError:
                return default

        id_i       = idx("id")
        sum_i      = idx("feature_summary")
        guess_i    = idx("company_guess")
        canon_i    = idx("canonical_company")
        created_i  = idx("created_utc")
        fit_i      = idx("fit_score")
        ups_i      = idx("ups")
        comments_i = idx("num_comments")
        conf_i     = idx("confidence")
        domain_i   = idx("company_domain")

        posts_for_cluster = []
        request_rows = []
        for row in all_values[1:]:
            try:
                pid = row[id_i].strip() if id_i >= 0 and id_i < len(row) else ""
                fsum = row[sum_i].strip() if sum_i >= 0 and sum_i < len(row) else ""
                guess = row[guess_i].strip() if guess_i >= 0 and guess_i < len(row) else ""
                canon = row[canon_i].strip() if canon_i >= 0 and canon_i < len(row) else ""
                comp = canon or guess
                created = float(row[created_i]) if created_i >= 0 and created_i < len(row) and row[created_i] else 0.0
                fit = float(row[fit_i]) if fit_i >= 0 and fit_i < len(row) and row[fit_i] else None
                ups = int(row[ups_i]) if ups_i >= 0 and ups_i < len(row) and row[ups_i] else 0
                comments = int(row[comments_i]) if comments_i >= 0 and comments_i < len(row) and row[comments_i] else 0
                conf = float(row[conf_i]) if conf_i >= 0 and conf_i < len(row) and row[conf_i] else 0.0
                domain = row[domain_i].strip() if domain_i >= 0 and domain_i < len(row) else ""
            except Exception:
                pid, fsum, comp, created, fit, ups, comments, conf, domain = "", "", "", 0.0, None, 0, 0, 0.0, ""
            if pid and fsum:
                posts_for_cluster.append({"id": pid, "feature_summary": fsum, "company": comp, "created_utc": created, "fit_score": fit})
                request_rows.append({
                    "id": pid,
                    "company": comp,
                    "company_domain": domain,
                    "confidence": conf,
                    "ups": ups,
                    "num_comments": comments,
                    "created_utc": created,
                    "fit_score": fit,
                })

        id_to_company = {p["id"]: (p.get("company") or "") for p in posts_for_cluster}
        clusters = llm_cluster_features(posts_for_cluster)

        if clusters is not None:
            # Build cluster rows (overwrite tab)
            # Columns: cluster_name | feature_summary | request_count | example_ids | companies_mentioned | cluster_fit_score | cluster_pitch | last_seen_utc
            rows = []
            # Maps for lookup
            created_map = {p["id"]: p.get("created_utc", 0.0) for p in posts_for_cluster}
            fit_map = {p["id"]: p.get("fit_score") for p in posts_for_cluster}

            for c in clusters:
                member_ids = c.get("member_ids", [])

                company_counts: Dict[str, int] = {}
                for mid in member_ids:
                    comp = id_to_company.get(mid)
                    if comp:
                        company_counts[comp] = company_counts.get(comp, 0) + 1

                c["company_member_counts"] = company_counts
                c["companies"] = sorted(company_counts.keys())

                # last seen timestamp
                last_seen_ts = 0.0
                for mid in member_ids:
                    ts = created_map.get(mid, 0.0) or 0.0
                    if ts > last_seen_ts:
                        last_seen_ts = ts
                last_seen_iso = datetime.fromtimestamp(last_seen_ts, tz=timezone.utc).isoformat() if last_seen_ts else ""

                # average fit score
                scores = [fit_map.get(mid) for mid in member_ids if isinstance(fit_map.get(mid), (int, float))]
                cluster_fit = sum(scores) / len(scores) if scores else 0.0
                cluster_fit_str = f"{cluster_fit:.2f}" if scores else ""

                # pitch
                pitch = llm_cluster_pitch(c.get("name", ""), c.get("summary", ""))

                c["request_count"] = len(member_ids)
                c["cluster_fit_score"] = cluster_fit

                rows.append([
                    c.get("name","")[:200],
                    c.get("summary","")[:500],
                    str(len(member_ids)),
                    ", ".join(member_ids[:50]),
                    ", ".join(c.get("companies", [])[:20]),
                    cluster_fit_str,
                    pitch[:500],
                    last_seen_iso
                ])

            clear_and_write_clusters(ws_clusters, rows)

            if ENABLE_COMPANY_SCORES == "1":
                _, ws_companies = open_existing_sheet_and_tab(gc, SHEET_ID, SHEET_COMPANIES_TAB)
                company_dicts = compute_company_scores(request_rows, clusters, TOP_N)
                log(f"[STAGE] Companies scored: {len(company_dicts)}")
                comp_rows = []
                for d in company_dicts:
                    comp_rows.append([
                        d.get("company",""),
                        d.get("domain",""),
                        str(d.get("count",0)),
                        f"{d.get('avg_confidence',0.0):.2f}",
                        d.get("top_feature",""),
                        str(d.get("top_feature_count",0)),
                        ", ".join(d.get("top_clusters", [])),
                        f"{d.get('opportunity_score',0.0):.2f}",
                        d.get("region_hint",""),
                        d.get("status",""),
                    ])
                clear_and_write_companies(ws_companies, comp_rows)
                print(json.dumps({"top_companies": company_dicts}, indent=2), flush=True)
        else:
            log("[STAGE] Clustering failed; preserving existing cluster sheet.")

    elapsed = time.time() - start

    counts = {
        "search": len(candidates),
        "classify": len(results),
        "clusters": len(rows) if 'rows' in locals() else 0,
        "companies": len(comp_rows) if 'comp_rows' in locals() else 0,
        "appended": len(new_rows),
    }

    summary = {
        "harvested_at_utc": now_iso,
        "seed_queries": SEED_QUERIES,
        "found": counts["classify"],
        "appended": counts["appended"],
        "clusters_written": counts["clusters"],
        "companies_written": counts["companies"],
        "counts": counts,
        "elapsed_sec": round(elapsed, 2),
        "settings": {
            "LOOKBACK_DAYS": LOOKBACK_DAYS,
            "PAGES_PER_QUERY": PAGES_PER_QUERY,
            "LIMIT_PER_PAGE": LIMIT_PER_PAGE,
            "EXPANSION_TOPN": EXPANSION_TOPN,
            "MAX_CLASSIFY": MAX_CLASSIFY,
            "REQUESTS_PER_MIN": REQUESTS_PER_MIN,
            "CONF_THRESHOLD": CONF_THRESHOLD,
            "TIME_RANGE": TIME_RANGE,
            "ENABLE_FIT_SCORING": ENABLE_FIT_SCORING,
            "FIT_MODEL": FIT_MODEL,
            "ENABLE_COMPANY_SCORES": ENABLE_COMPANY_SCORES,
            "TOP_N": TOP_N,
            "ENABLE_SUB_DISCOVERY": ENABLE_SUB_DISCOVERY,
            "SUBS_PER_COMPANY": SUBS_PER_COMPANY,
            "SUB_CRAWL_PAGES_MAX": SUB_CRAWL_PAGES_MAX,
        }
    }

    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("FATAL ERROR:", repr(e))
        traceback.print_exc()
        raise
