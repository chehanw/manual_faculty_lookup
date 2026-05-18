"""
Faculty enrichment for manual single-person lookup.

Agents:
  1) Enricher — NIH Reporter + Scopus + web search for roles/title
  2) QC       — faithfulness verification of role claims

Usage:
    export OPENAI_API_KEY="your-openai-key"
    export SCOPUS_API_KEY="your-scopus-key"
    python3 server.py
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

import observe
from providers.live import LiveNihProvider, LiveRolesProvider, LiveHIndexProvider, merge_specialty_considerations
from providers.profile_scrapers import StanfordProfileScraper, TavilyProfileScraper, PlaywrightProfileScraper
from providers.profile_scrapers.base import ProfileScraper
from providers.qc import verify_faculty_roles

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
SCOPUS_API_KEY = os.environ.get("SCOPUS_API_KEY")

DISCIPLINE  = "Pediatric Endocrinology"
MAX_WORKERS = 8


# ─────────────────────────────────────────────
# HELPER: Responses API web search
# ─────────────────────────────────────────────

def responses_web_search(prompt: str) -> str:
    t0 = time.time()
    observe.metric("api_calls", sub="openai_responses")
    try:
        response = client.responses.create(
            model="gpt-4o",
            tools=[{"type": "web_search_preview"}],
            input=[{"role": "user", "content": prompt}],
        )
        raw = ""
        for block in response.output:
            if hasattr(block, "content"):
                for item in block.content:
                    if hasattr(item, "text"):
                        raw += item.text
        cost = observe.add_cost("openai_responses_web_search")
        observe.log("api_call", provider="openai_responses",
                    latency_ms=int((time.time() - t0) * 1000),
                    cost_usd=round(cost, 4), status="ok")
        return raw.strip()
    except Exception as e:
        observe.metric("api_errors", sub="openai")
        observe.log("api_error", level="error", provider="openai_responses",
                    error=str(e), latency_ms=int((time.time() - t0) * 1000))
        raise


def parse_json(raw: str) -> any:
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("[")
        if start == -1:
            return []
        for end in range(len(clean), start, -1):
            try:
                return json.loads(clean[start:end])
            except json.JSONDecodeError:
                continue
        return []


# ─────────────────────────────────────────────
# AGENT 1: ENRICHER
# NIH Reporter + Scopus + web search
# ─────────────────────────────────────────────

_PROFILE_SCRAPERS: list[ProfileScraper] = [
    StanfordProfileScraper(),
    PlaywrightProfileScraper(),
]


def _get_profile_scraper(profile_url: str) -> ProfileScraper | None:
    for scraper in _PROFILE_SCRAPERS:
        if scraper.handles(profile_url):
            return scraper
    return None


def agent1_enrich(
    faculty: dict,
    nih_provider=None,
    roles_provider=None,
    hindex_provider=None,
    discipline: str = DISCIPLINE,
) -> dict:
    if nih_provider is None:
        nih_provider = LiveNihProvider(discipline=discipline)
    if roles_provider is None:
        roles_provider = LiveRolesProvider(client)
    if hindex_provider is None:
        hindex_provider = LiveHIndexProvider(SCOPUS_API_KEY)

    t0 = time.time()
    name = faculty.get("full_name", "Unknown")
    school = faculty.get("school", "")
    profile_url = faculty.get("profile_url") or ""

    print(f"  [Enrich] {name} ({school})")

    observe.metric("cache_misses_nih")
    observe.metric("cache_misses_roles")

    nih_grants       = []
    editorial_roles  = []
    society_roles    = []
    training_roles   = []
    leadership_roles = []
    h_index          = None

    print(f"    -> Fetching H-index via Scopus...")
    try:
        hindex_data = hindex_provider.get_hindex(name, school)
        h_index = hindex_data.get("h_index")
        print(f"    -> H-index: {h_index}")
    except Exception as e:
        print(f"    [!] H-index lookup failed: {e}")

    # Try profile scraper first — free, no LLM
    profile_scraper_roles: list | None = None
    if profile_url:
        scraper = _get_profile_scraper(profile_url)
        if scraper:
            try:
                scraped = scraper.get_roles(profile_url)
                if scraped:
                    profile_scraper_roles = scraped
                    print(f"    -> [ProfileScraper] {len(scraped)} roles from profile page")
            except Exception as e:
                print(f"    [!] ProfileScraper error, falling back to LLM: {e}")

    need_title = not faculty.get("title")
    fetch_keys = ["nih"] + ([] if profile_scraper_roles is not None else ["roles"])
    with ThreadPoolExecutor(max_workers=2) as _pool:
        futures_map = {"nih": _pool.submit(nih_provider.get_grants, name, school)}
        if profile_scraper_roles is None:
            if need_title:
                futures_map["roles"] = _pool.submit(roles_provider.get_roles_and_title, name, school, profile_url)
            else:
                futures_map["roles"] = _pool.submit(roles_provider.get_roles, name, school, profile_url)
        fetch_results = []
        for key in fetch_keys:
            try:
                fetch_results.append(futures_map[key].result())
            except Exception as exc:
                fetch_results.append(exc)

    if profile_scraper_roles is not None:
        all_roles        = profile_scraper_roles
        editorial_roles  = [r for r in all_roles if r.get("category") == "editorial"]
        society_roles    = [r for r in all_roles if r.get("category") == "society"]
        training_roles   = [r for r in all_roles if r.get("category") == "training"]
        leadership_roles = [r for r in all_roles if r.get("category") == "leadership"]

    extracted_title: str | None = None
    for key, result in zip(fetch_keys, fetch_results):
        if key == "nih":
            if isinstance(result, Exception):
                print(f"    [!] NIH lookup failed for {name}: {result}")
            else:
                nih_grants = result
                print(f"    -> NIH grants found: {len(nih_grants)}")
        elif key == "roles":
            if isinstance(result, Exception):
                print(f"    [!] Roles search failed for {name}: {result}")
            else:
                if need_title and isinstance(result, tuple):
                    extracted_title, all_roles = result
                    if extracted_title:
                        print(f"    -> [LLM] Title found: {extracted_title}")
                else:
                    all_roles = result
                editorial_roles  = [r for r in all_roles if r.get("category") == "editorial"]
                society_roles    = [r for r in all_roles if r.get("category") == "society"]
                training_roles   = [r for r in all_roles if r.get("category") == "training"]
                leadership_roles = [r for r in all_roles if r.get("category") == "leadership"]

    nih_project_nums = ", ".join(g["project_num"] for g in nih_grants)
    nih_grant_titles = " | ".join(g["title"] for g in nih_grants)
    nih_grant_count  = len(nih_grants)
    nih_institution_unconfirmed = any(g.get("institution_unconfirmed") for g in nih_grants)

    latency_ms = int((time.time() - t0) * 1000)
    observe.metric("faculty_enriched")
    observe.metric("latencies_ms", value=latency_ms)
    if nih_grants:       observe.metric("field_coverage", sub="nih_grants")
    if editorial_roles:  observe.metric("field_coverage", sub="editorial")
    if society_roles:    observe.metric("field_coverage", sub="society")
    if training_roles:   observe.metric("field_coverage", sub="training")
    if leadership_roles: observe.metric("field_coverage", sub="leadership")
    observe.log("faculty_enriched", faculty=name, school=school,
                nih_grants=nih_grant_count,
                editorial=len(editorial_roles), society=len(society_roles),
                training=len(training_roles), leadership=len(leadership_roles),
                latency_ms=latency_ms)

    return {
        **faculty,
        "title":                     faculty.get("title") or extracted_title,
        "h_index":                   h_index,
        "nih_project_nums":          nih_project_nums,
        "nih_grant_titles":          nih_grant_titles,
        "nih_grant_count":           nih_grant_count,
        "nih_institution_unconfirmed": nih_institution_unconfirmed,
        "editorial_roles":           editorial_roles,
        "society_roles":             society_roles,
        "training_roles":            training_roles,
        "leadership_roles":          leadership_roles,
    }


# ─────────────────────────────────────────────
# AGENT 2: QC
# ─────────────────────────────────────────────

def _qc_cross_checks(results: list[dict]) -> None:
    from collections import defaultdict

    SINGLETON_KEYWORDS = {
        "chief", "chair", "president", "director", "dean",
        "program director", "fellowship director",
    }

    role_holders: dict[str, list[str]] = defaultdict(list)
    for f in results:
        name = f.get("full_name", "?")
        for detail in (f.get("qc") or {}).get("role_details", []):
            role_lower = (detail.get("role") or "").lower()
            org_lower  = (detail.get("org")  or "").lower()
            if any(kw in role_lower for kw in SINGLETON_KEYWORDS):
                key = f"{role_lower} @ {org_lower}"
                role_holders[key].append(name)

    duplicate_warnings = {k: v for k, v in role_holders.items() if len(v) > 1}

    if duplicate_warnings:
        print("\n  [QC] DUPLICATE SINGLETON ROLES DETECTED:")
        for role_key, holders in duplicate_warnings.items():
            print(f"       '{role_key}' claimed by: {', '.join(holders)}")

    for f in results:
        name = f.get("full_name", "?")
        warnings = []
        for role_key, holders in duplicate_warnings.items():
            if name in holders:
                others = [h for h in holders if h != name]
                warnings.append(f"Role '{role_key}' also claimed by: {', '.join(others)}")
        if warnings and f.get("qc"):
            f["qc"]["warnings"] = f["qc"].get("warnings", []) + warnings

    print("\n  [QC] Completeness check:")
    for f in results:
        name        = f.get("full_name", "?")
        h_index     = f.get("h_index") or 0
        grant_count = f.get("nih_grant_count") or 0
        role_count  = sum(
            len(f.get(f"{cat}_roles") or [])
            for cat in ("editorial", "society", "leadership", "training")
        )
        is_senior = h_index >= 30 or grant_count >= 5
        if is_senior and role_count == 0:
            msg = (
                f"SPARSE PROFILE — h_index={h_index}, grants={grant_count} "
                f"but 0 roles found. Likely scraping failure."
            )
            print(f"       {name}: {msg}")
            if f.get("qc"):
                f["qc"]["warnings"] = f["qc"].get("warnings", []) + [msg]
                if f["qc"].get("overall") == "green":
                    f["qc"]["overall"] = "yellow"
        else:
            print(f"       {name}: {role_count} role(s), h={h_index}, grants={grant_count}")


def agent2_qc(faculty_list: list[dict]) -> list[dict]:
    print(f"\n[QC] Running on {len(faculty_list)} faculty members...")

    results = []
    green = yellow = red = 0

    for faculty in faculty_list:
        name = faculty.get("full_name", "Unknown")
        try:
            qc = verify_faculty_roles(faculty, client)
            faculty = {**faculty, "qc": qc}
            overall = qc["overall"]
            if overall == "green":    green  += 1
            elif overall == "yellow": yellow += 1
            else:                     red    += 1
            print(f"  [QC] {name}: {overall.upper()} ({len(qc.get('role_details', []))} roles)")
        except Exception as e:
            print(f"  [QC] Failed for {name}: {e}")
            faculty = {**faculty, "qc": {"overall": "yellow", "error": str(e)}}
        results.append(faculty)

    _qc_cross_checks(results)

    green = yellow = red = 0
    for f in results:
        o = (f.get("qc") or {}).get("overall", "yellow")
        if o == "green":    green  += 1
        elif o == "yellow": yellow += 1
        else:               red    += 1

    print(f"\n  -> QC complete. Green: {green}  Yellow: {yellow}  Red: {red}")
    return results


# ─────────────────────────────────────────────
# SUMMARY GENERATION
# ─────────────────────────────────────────────

def _parse_roles(value) -> list:
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def generate_faculty_summary(record: dict) -> str:
    name   = record.get("full_name") or "This faculty member"
    title  = record.get("title") or ""
    school = record.get("school") or ""

    sentences = []

    article = "an" if title and title[0].lower() in "aeiou" else "a"
    if title and school:
        sentences.append(f"{name} is {article} {title} at {school}.")
    elif school:
        sentences.append(f"{name} is at {school}.")
    else:
        sentences.append(f"{name} is a faculty member.")

    h = record.get("h_index")
    if h:
        sentences.append(f"Their H-index is {h}.")

    grant_count = int(record.get("nih_grant_count") or 0)
    if grant_count > 0:
        raw_titles = record.get("nih_grant_titles") or ""
        titles = [t.strip() for t in raw_titles.split("|") if t.strip()]
        if titles:
            shown  = titles[:2]
            quoted = ", ".join(f'"{t}"' for t in shown)
            extra  = grant_count - len(shown)
            more   = f", and {extra} more" if extra > 0 else ""
            sentences.append(
                f"They hold {grant_count} active NIH grant{'s' if grant_count != 1 else ''}, "
                f"including {quoted}{more}."
            )
        else:
            sentences.append(f"They hold {grant_count} active NIH grant{'s' if grant_count != 1 else ''}.")

    society_roles = _parse_roles(record.get("society_roles"))
    if society_roles:
        orgs = list(dict.fromkeys(r.get("org", "") for r in society_roles if r.get("org")))
        if orgs:
            if len(orgs) == 1:
                sentences.append(f"They are a member of the {orgs[0]}.")
            else:
                joined = ", ".join(orgs[:-1]) + f", and {orgs[-1]}"
                sentences.append(f"They are a member of the {joined}.")

    editorial_roles = _parse_roles(record.get("editorial_roles"))
    if editorial_roles:
        orgs = list(dict.fromkeys(r.get("org", "") for r in editorial_roles if r.get("org")))
        if orgs:
            sentences.append(f"They serve on the editorial board of {', '.join(orgs[:2])}.")

    leadership_roles = _parse_roles(record.get("leadership_roles"))
    if leadership_roles:
        first = leadership_roles[0]
        role_str = first.get("role", "")
        org_str  = first.get("org", "")
        if role_str and org_str:
            sentences.append(f"They hold the role of {role_str} at {org_str}.")

    return " ".join(sentences)


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_pipeline_from_records(
    records,
    on_enriched=None,
    on_qc_complete=None,
    roles_provider=None,
    discipline: str = DISCIPLINE,
    considerations: str = "",
) -> list[dict]:
    """Enrich and QC a pre-built list of faculty records.

    Records must have at minimum: full_name, school.
    Returns the final QC'd list.
    """
    observe.start()
    if not records:
        print("\n[!] No records provided.")
        return []

    if considerations and roles_provider is None:
        if discipline:
            merge_specialty_considerations(discipline, considerations, client)
            roles_provider = LiveRolesProvider(client, discipline=discipline)
        else:
            roles_provider = LiveRolesProvider(client, discipline="", considerations=considerations)

    enrich_start = time.time()
    n = len(records)
    print(f"\n[Enrich] Processing {n} faculty profiles ({MAX_WORKERS} workers)...")
    enriched = [None] * n
    _print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(agent1_enrich, f, roles_provider=roles_provider, discipline=discipline): i
            for i, f in enumerate(records)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            idx = futures[future]
            try:
                result = future.result()
                result["summary"] = generate_faculty_summary(result)
            except Exception as exc:
                original = records[idx]
                print(f"  [Warning] Enrichment failed for {original.get('full_name','?')}: {exc}")
                result = {
                    **original,
                    "summary": "", "h_index": None,
                    "nih_grant_count": 0, "nih_project_nums": "", "nih_grant_titles": "",
                }
            enriched[idx] = result
            with _print_lock:
                print(f"  [Progress] {completed}/{n} — {result.get('full_name', 'Unknown')}")
            try:
                if on_enriched:
                    on_enriched(result, completed, n)
            except Exception as exc:
                print(f"  [Warning] on_enriched callback failed: {exc}")

    elapsed = time.time() - enrich_start
    print(f"\n[Timing] Enrichment: {elapsed:.1f}s for {n} faculty ({elapsed/n:.1f}s avg)")

    final = agent2_qc(enriched)

    if on_qc_complete:
        for record in final:
            try:
                on_qc_complete(record)
            except Exception as exc:
                print(f"  [Warning] on_qc_complete callback failed: {exc}")

    return final
