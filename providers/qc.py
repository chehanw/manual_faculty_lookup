"""
providers/qc.py — Role claim verifier using three independent checks.

For each role collected by Agent 2:
  1. url_alive     — does source_url return a 2xx response?
  2. name_on_page  — does the person's name appear in evidence_text or source_url?
  3. role_on_page  — do key tokens of the role title appear in evidence_text?

Status:
  high   — URL alive + name on page + role on page (no negation)
  low    — no evidence, or URL alive but name NOT on page (wrong page)
  medium — everything else (URL dead but evidence looks good, role fuzzy, etc.)

Negation check: if evidence contains "former", "previously", closed date ranges, etc.
  → caps status at medium even if all three checks pass.

NIH confidence:
  high   — grants found + institution confirmed, or no grants found (not a failure)
  medium — grants found but institution unconfirmed (needs human review)

H-index confidence:
  high   — h-index found on Scopus
  medium — not found (manual lookup may be needed)
"""

import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from rapidfuzz import fuzz

ROLE_CATEGORIES = ("editorial", "society", "leadership", "training")

_CREDENTIALS = frozenset({
    "MD", "PHD", "DO", "DPHIL", "MBBS", "MPH", "MS", "MSC", "MHS", "MBA",
    "FAAP", "FACS", "FACP", "FACE", "RN", "NP", "PA", "JR", "SR", "II", "III", "IV",
})

_STOP_WORDS = frozenset({
    "of", "the", "and", "at", "for", "in", "a", "an", "to", "with", "by",
})

_ORG_STOP_WORDS = _STOP_WORDS | frozenset({
    "university", "school", "college", "institute", "hospital", "center",
    "centre", "medical", "health", "system", "national", "american", "society",
    "association", "foundation", "department", "division",
})


# ── Name helpers ──────────────────────────────────────────────────────────────

def _clean_name_parts(full_name: str) -> list[str]:
    return [
        p.strip(".,")
        for p in full_name.split()
        if p.strip(".,").upper() not in _CREDENTIALS and p.strip(".,")
    ]


def _name_variants(full_name: str) -> list[str]:
    parts = _clean_name_parts(full_name)
    if not parts:
        return [full_name]
    variants = [" ".join(parts)]
    if len(parts) >= 2:
        variants.append(f"{parts[0]} {parts[-1]}")
        variants.append(f"{parts[0][0]}. {parts[-1]}")
        variants.append(f"{parts[0][0]}{parts[-1]}")
        variants.append(parts[-1])
    return variants


def _name_in_url(full_name: str, url: str) -> bool:
    if not url:
        return False
    url_lower = url.lower()
    parts = [p.lower() for p in _clean_name_parts(full_name)]
    if not parts:
        return False
    last = parts[-1]
    if last not in url_lower:
        return False
    if len(parts) >= 2 and parts[0][0] in url_lower:
        return True
    return len(last) > 7


def _name_in_evidence(full_name: str, evidence: str) -> bool:
    if not evidence:
        return False
    evidence_lower = evidence.lower()
    parts = _clean_name_parts(full_name)
    for variant in _name_variants(full_name):
        if len(variant) <= len(parts[-1] if parts else variant):
            continue
        v_lower = variant.lower()
        if v_lower in evidence_lower:
            return True
        if len(v_lower) >= 6 and fuzz.partial_ratio(v_lower, evidence_lower) >= 88:
            return True
    return False


# ── Check helpers ─────────────────────────────────────────────────────────────

def _token_overlap(query: str, text: str, stop_words: frozenset) -> float:
    tokens = [
        t.lower() for t in re.findall(r"[a-zA-Z]+", query)
        if t.lower() not in stop_words and len(t) > 2
    ]
    if not tokens:
        return 0.0
    text_lower = text.lower()
    matched = sum(1 for t in tokens if t in text_lower)
    return matched / len(tokens)


def _role_on_page(role: str, evidence: str) -> bool:
    return _token_overlap(role, evidence, _STOP_WORDS) >= 0.5


def _negation_detected(role: str, evidence: str) -> bool:
    if not evidence:
        return False
    ev = evidence.lower()
    negation_phrases = [
        r"\bformer\b", r"\bpreviously\b", r"\bno longer\b",
        r"\bstepped down\b", r"\bretired\b", r"\bpast\b\s+(?:chair|president|director|chief)",
    ]
    for pat in negation_phrases:
        if re.search(pat, ev):
            return True
    role_tokens = [t.lower() for t in re.findall(r"[a-zA-Z]+", role) if len(t) > 2]
    closed_range = re.search(
        r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+)?(\d{4})\s*[-–]\s*"
        r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+)?(\d{4})\b",
        ev, re.IGNORECASE,
    )
    if closed_range:
        end_year = int(closed_range.group(4))
        if end_year < 2020:
            range_pos = closed_range.start()
            for tok in role_tokens:
                tok_pos = ev.find(tok)
                if tok_pos >= 0 and abs(tok_pos - range_pos) < 250:
                    return True
    return False


def _check_url(url: str) -> "bool | None":
    """True = alive (2xx), False = dead (4xx/5xx), None = inconclusive/no URL."""
    if not url:
        return None
    try:
        resp = requests.get(
            url, timeout=5, allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code == 403:
            return None  # bot-blocked, can't determine
        return resp.status_code < 400
    except Exception:
        return None


def _status_from_checks(url_alive, name_on_page: bool, role_on_page: bool,
                         negation: bool, has_evidence: bool) -> str:
    if not has_evidence:
        return "red"
    if url_alive is True and not name_on_page:
        return "red"  # URL resolves but it's the wrong page
    if url_alive is True and name_on_page and role_on_page and not negation:
        return "green"
    return "yellow"


# ── Main verifier ─────────────────────────────────────────────────────────────

def _worst(statuses: list[str]) -> str:
    if "red" in statuses:
        return "red"
    if "yellow" in statuses:
        return "yellow"
    if "green" in statuses:
        return "green"
    return "green"


def _nih_confidence(faculty: dict) -> str:
    if faculty.get("nih_grant_count") and faculty.get("nih_institution_unconfirmed"):
        return "yellow"
    return "green"


def _hindex_confidence(faculty: dict) -> str:
    return "green" if faculty.get("h_index") is not None else "yellow"


def verify_faculty_roles(faculty: dict, client=None) -> dict:
    """
    Checks all role claims + NIH + H-index confidence for one faculty member.

    Shape:
    {
      "overall":               "green" | "yellow" | "red",
      "nih_confidence":        "green" | "yellow",
      "hindex_confidence":     "green" | "yellow",
      "editorial_confidence":  "green" | "yellow" | "red",
      "society_confidence":    "green" | "yellow" | "red",
      "leadership_confidence": "green" | "yellow" | "red",
      "training_confidence":   "green" | "yellow" | "red",
      "role_details": [ ... ]
    }
    """
    name = faculty.get("full_name", "Unknown")

    nih_conf    = _nih_confidence(faculty)
    hindex_conf = _hindex_confidence(faculty)

    all_roles: list[dict] = []
    for cat in ROLE_CATEGORIES:
        for role in (faculty.get(f"{cat}_roles") or []):
            if isinstance(role, dict):
                all_roles.append({**role, "_cat": cat})

    if not all_roles:
        empty = {f"{cat}_confidence": "green" for cat in ROLE_CATEGORIES}
        overall = _worst([nih_conf, hindex_conf])
        print(f"    [QC] {name} — no roles | NIH: {nih_conf} | H-index: {hindex_conf} | OVERALL: {overall.upper()}")
        return {"overall": overall, "nih_confidence": nih_conf, "hindex_confidence": hindex_conf,
                "role_details": [], **empty}

    print(f"    [QC] {name} — checking {len(all_roles)} role(s) (URL + evidence)")

    # Check all URLs in parallel
    urls = [r.get("source_url", "") for r in all_roles]
    url_results: dict[str, "bool | None"] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        future_to_url = {pool.submit(_check_url, u): u for u in set(urls) if u}
        for future in as_completed(future_to_url):
            url_results[future_to_url[future]] = future.result()

    role_details = []
    category_statuses: dict[str, list[str]] = {cat: [] for cat in ROLE_CATEGORIES}

    for r in all_roles:
        cat        = r.get("_cat", "editorial")
        role_title = r.get("role", "")
        org        = r.get("org", "")
        evidence   = (r.get("evidence_text") or "").strip()
        source_url = r.get("source_url", "")

        url_alive   = url_results.get(source_url) if source_url else None
        name_on_pg  = _name_in_evidence(name, evidence) or _name_in_url(name, source_url)
        role_on_pg  = _role_on_page(role_title, evidence)
        negation    = _negation_detected(role_title, evidence)

        status = _status_from_checks(url_alive, name_on_pg, role_on_pg, negation, bool(evidence))
        category_statuses[cat].append(status)

        url_label = "alive" if url_alive is True else ("dead" if url_alive is False else "unknown")
        print(
            f"           [{status.upper()}] {cat}: \"{role_title}\" @ \"{org}\" | "
            f"url={url_label} name={'✓' if name_on_pg else '✗'} "
            f"role={'✓' if role_on_pg else '✗'} neg={'✓' if negation else '—'}"
        )

        role_details.append({
            "category":         cat,
            "role":             role_title,
            "org":              org,
            "status":           status,
            "source_url":       source_url,
            "url_alive":        url_alive,
            "name_on_page":     name_on_pg,
            "role_on_page":     role_on_pg,
            "negation_detected": negation,
            "evidence_snippet": (evidence[:200] + "…") if len(evidence) > 200 else evidence,
        })

    cat_confidences = {
        f"{cat}_confidence": _worst(category_statuses[cat]) if category_statuses[cat] else "green"
        for cat in ROLE_CATEGORIES
    }
    overall = _worst([*cat_confidences.values(), nih_conf, hindex_conf])

    print(
        f"    [QC] {name} summary — "
        + " | ".join(f"{cat}: {cat_confidences[f'{cat}_confidence']}" for cat in ROLE_CATEGORIES)
        + f" | NIH: {nih_conf} | H-index: {hindex_conf} | OVERALL: {overall.upper()}"
    )

    return {
        "overall": overall,
        "nih_confidence": nih_conf,
        "hindex_confidence": hindex_conf,
        **cat_confidences,
        "role_details": role_details,
    }
