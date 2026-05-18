"""
Live providers — perform real network calls to NIH Reporter API, OpenAI Responses API, Scopus.
Use ReplayProviders for tests.
"""
import time
import threading
import re
from pathlib import Path
from rapidfuzz import fuzz
import json
import requests
from datetime import datetime, timezone

from providers.base import (
    NihProvider, RolesProvider, HIndexProvider,
    Grant, RoleEvidence, HIndexEvidence,
)

_INSTITUTION_NIH_NAMES: dict[str, list[str]] = {
    "Stanford University School of Medicine":                                          ["STANFORD", "LUCILE SALTER PACKARD"],
    "University of California San Francisco School of Medicine":                       ["UNIVERSITY OF CALIFORNIA SAN FRANCISCO", "UCSF"],
    "Harvard Medical School":                                                          ["HARVARD", "BOSTON CHILDREN"],
    "Johns Hopkins School of Medicine":                                                ["JOHNS HOPKINS"],
    "University of Pennsylvania Perelman School of Medicine":                          ["UNIVERSITY OF PENNSYLVANIA", "CHILDREN'S HOSPITAL OF PHILADELPHIA"],
    "Columbia University Vagelos College of Physicians and Surgeons":                  ["COLUMBIA UNIVERSITY"],
    "Duke University School of Medicine":                                              ["DUKE UNIVERSITY"],
    "Washington University in St. Louis School of Medicine":                           ["WASHINGTON UNIVERSITY"],
    "University of Michigan Medical School":                                           ["UNIVERSITY OF MICHIGAN"],
    "Yale School of Medicine":                                                         ["YALE UNIVERSITY"],
    "University of Colorado School of Medicine / Children's Hospital Colorado":        ["UNIVERSITY OF COLORADO", "CHILDREN'S HOSPITAL COLORADO"],
    "University of Cincinnati College of Medicine / Cincinnati Children's":            ["UNIVERSITY OF CINCINNATI", "CINCINNATI CHILDREN"],
    "Baylor College of Medicine / Texas Children's Hospital":                          ["BAYLOR COLLEGE OF MEDICINE", "TEXAS CHILDREN"],
    "University of Washington School of Medicine / Seattle Children's":                ["UNIVERSITY OF WASHINGTON", "SEATTLE CHILDREN"],
    "Children's National Hospital / George Washington University":                     ["CHILDREN'S NATIONAL", "GEORGE WASHINGTON UNIVERSITY"],
    "Northwestern University Feinberg School of Medicine / Lurie Children's":          ["NORTHWESTERN UNIVERSITY", "LURIE CHILDREN"],
    "NIH National Institute of Child Health and Human Development (NICHD)":            ["NICHD", "NATIONAL INSTITUTE OF CHILD HEALTH"],
    "Keck School of Medicine USC / Children's Hospital Los Angeles":                   ["UNIVERSITY OF SOUTHERN CALIFORNIA", "CHILDREN'S HOSPITAL LOS ANGELES"],
    "Emory University School of Medicine / Children's Healthcare of Atlanta":          ["EMORY UNIVERSITY", "CHILDREN'S HEALTHCARE OF ATLANTA"],
    "University of Florida College of Medicine":                                       ["UNIVERSITY OF FLORIDA"],
    "Ohio State University College of Medicine / Nationwide Children's":               ["OHIO STATE UNIVERSITY", "NATIONWIDE CHILDREN"],
    "UC San Diego School of Medicine / Rady Children's Hospital":                      ["UNIVERSITY OF CALIFORNIA SAN DIEGO", "RADY CHILDREN"],
    "University of Pittsburgh School of Medicine / UPMC Children's Hospital":          ["UNIVERSITY OF PITTSBURGH", "UPMC", "CHILDREN'S HOSPITAL OF PITTSBURGH"],
    "David Geffen School of Medicine at UCLA / Mattel Children's Hospital":            ["UNIVERSITY OF CALIFORNIA LOS ANGELES", "UCLA", "MATTEL CHILDREN"],
    "Vanderbilt University School of Medicine / Monroe Carell Jr. Children's":         ["VANDERBILT UNIVERSITY", "MONROE CARELL"],
    "Medical University of South Carolina":                                            ["MEDICAL UNIVERSITY OF SOUTH CAROLINA", "MUSC"],
    "UT Southwestern Medical Center / Children's Medical Center Dallas":               ["UNIVERSITY OF TEXAS SOUTHWESTERN", "UT SOUTHWESTERN", "CHILDREN'S MEDICAL CENTER DALLAS"],
    "Dell Medical School, University of Texas at Austin / Dell Children's":            ["UNIVERSITY OF TEXAS AT AUSTIN", "DELL CHILDREN"],
    "University of Utah School of Medicine / Intermountain Primary Children's":        ["UNIVERSITY OF UTAH", "INTERMOUNTAIN", "PRIMARY CHILDREN"],
    "University of Missouri-Kansas City School of Medicine / Children's Mercy":        ["UNIVERSITY OF MISSOURI KANSAS CITY", "CHILDREN'S MERCY"],
    "Nemours Children's Health":                                                       ["NEMOURS", "ALFRED I. DUPONT"],
    "NYU Grossman School of Medicine / Hassenfeld Children's Hospital":                 ["NEW YORK UNIVERSITY", "NYU LANGONE"],
    "Atrium Health / Levine Children's Hospital":                                      ["ATRIUM HEALTH", "LEVINE CHILDREN", "CAROLINAS MEDICAL"],
    "Mayo Clinic Alix School of Medicine":                                             ["MAYO CLINIC", "MAYO FOUNDATION"],
    "Cleveland Clinic Lerner College of Medicine":                                     ["CLEVELAND CLINIC", "LERNER COLLEGE"],
}

_INSTITUTION_KEYWORDS: dict[str, str] = {
    "Stanford University School of Medicine":                                          "stanford",
    "University of California San Francisco School of Medicine":                       "san francisco",
    "Harvard Medical School":                                                          "harvard",
    "Johns Hopkins School of Medicine":                                                "johns hopkins",
    "University of Pennsylvania Perelman School of Medicine":                          "pennsylvania",
    "Columbia University Vagelos College of Physicians and Surgeons":                  "columbia",
    "Duke University School of Medicine":                                              "duke",
    "Washington University in St. Louis School of Medicine":                           "washington university",
    "University of Michigan Medical School":                                           "michigan",
    "Yale School of Medicine":                                                         "yale",
    "University of Colorado School of Medicine / Children's Hospital Colorado":        "colorado",
    "University of Cincinnati College of Medicine / Cincinnati Children's":            "cincinnati",
    "Baylor College of Medicine / Texas Children's Hospital":                          "baylor",
    "University of Washington School of Medicine / Seattle Children's":                "washington",
    "Children's National Hospital / George Washington University":                     "children's national",
    "Northwestern University Feinberg School of Medicine / Lurie Children's":          "northwestern",
    "NIH National Institute of Child Health and Human Development (NICHD)":            "nichd",
    "Keck School of Medicine USC / Children's Hospital Los Angeles":                   "southern california",
    "Emory University School of Medicine / Children's Healthcare of Atlanta":          "emory",
    "University of Florida College of Medicine":                                       "florida",
    "Ohio State University College of Medicine / Nationwide Children's":               "ohio state",
    "UC San Diego School of Medicine / Rady Children's Hospital":                      "rady children",
    "University of Pittsburgh School of Medicine / UPMC Children's Hospital":          "pittsburgh",
    "David Geffen School of Medicine at UCLA / Mattel Children's Hospital":            "los angeles",
    "Vanderbilt University School of Medicine / Monroe Carell Jr. Children's":         "vanderbilt",
    "Medical University of South Carolina":                                            "south carolina",
    "UT Southwestern Medical Center / Children's Medical Center Dallas":               "southwestern",
    "Dell Medical School, University of Texas at Austin / Dell Children's":            "texas austin",
    "University of Utah School of Medicine / Intermountain Primary Children's":        "utah",
    "University of Missouri-Kansas City School of Medicine / Children's Mercy":        "children's mercy",
    "Nemours Children's Health":                                                       "nemours",
    "NYU Grossman School of Medicine / Hassenfeld Children's Hospital":                 "nyu langone",
    "Atrium Health / Levine Children's Hospital":                                      "atrium",
    "Mayo Clinic Alix School of Medicine":                                             "mayo",
    "Cleveland Clinic Lerner College of Medicine":                                     "cleveland clinic",
    "Children's National Hospital / George Washington University":                     "george washington",
}

_CREDENTIAL_SUFFIXES = frozenset(
    ["MD", "PhD", "DO", "MPH", "MBA", "FAAP", "MS", "DFAACAP", "RD", "RN"]
)

_LIVE_MODEL = "gpt-4o"

_ROLES_DIR = Path(__file__).parent.parent / "data" / "roles"


def _load_discipline_context(discipline: str) -> str:
    filename = discipline.lower().replace(" ", "_").replace("/", "_") + ".md"
    path = _ROLES_DIR / filename
    if not path.exists():
        return ""
    content = path.read_text().strip()
    return f"--- Specialty guidance for {discipline} ---\n{content}\n---\n\n"


_MERGE_PROMPT = """\
You maintain a specialty guidance file used to direct an AI agent searching for professional \
roles of academic faculty. The file lists target journals, societies, and leadership roles \
relevant to the specialty.

Current file content:
{existing}

New input from a physician in this specialty:
{user_input}

Task: Produce an updated version of the file that incorporates genuinely new information:
- Add new journals at the TOP of the "Target Journals" section
- Add new societies at the TOP of the "Target Societies" section
- Add new leadership roles at the TOP of the "Leadership Roles" section
- Do NOT duplicate anything already present (check carefully before adding)
- Preserve the existing markdown structure and headings exactly
- Do NOT remove or reorder existing content

Return ONLY the updated file content. No explanation, no markdown fences.
"""

_MERGE_PROMPT_NEW = """\
Create a specialty guidance file for {discipline} faculty role searching based on this \
physician input: {user_input}

Structure it with these sections (markdown format):
# Role Search Config: {discipline}
## Target Societies
## Target Journals
## Leadership Roles

Return ONLY the file content. No explanation, no markdown fences.
"""


def merge_specialty_considerations(discipline: str, user_input: str, openai_client) -> None:
    """Merge physician input into the specialty .md file before Agent 2 runs."""
    if not user_input or not user_input.strip():
        return
    filename = discipline.lower().replace(" ", "_").replace("/", "_") + ".md"
    path = _ROLES_DIR / filename

    if path.exists():
        existing = path.read_text().strip()
        prompt = _MERGE_PROMPT.format(existing=existing, user_input=user_input.strip())
    else:
        prompt = _MERGE_PROMPT_NEW.format(discipline=discipline, user_input=user_input.strip())

    print(f"  [Merge] Updating specialty guidance for {discipline}...")
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    updated = resp.choices[0].message.content.strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated + "\n")
    print(f"  [Merge] Saved updated guidance → {path.name}")


_ROLES_PROMPT = """\
{discipline_context}Search the web for the professional roles and service activities of {name} at {school}.

Also find their current academic rank/title (e.g. "Professor", "Associate Professor", \
"Assistant Professor", "Instructor", "Lecturer"). Look at their institutional faculty \
profile page. Return null if not clearly stated.


Include ONLY roles that involve active service or responsibility beyond their faculty appointment:
- Editorial board positions (e.g. "Associate Editor, Diabetes Care")
- Professional society committee roles (e.g. "President, Pediatric Endocrine Society")
- Institutional or divisional leadership (e.g. "Division Chief", "Program Director", "Center Director")
- Training/education program roles (e.g. "Fellowship Program Director", "Clerkship Director")

Do NOT include:
- Academic titles or ranks (Professor, Assistant Professor, Instructor, Lecturer, Attending)
- General employment or affiliation (e.g. "Faculty, Stanford")
- Degrees or credentials (MD, PhD, etc.)
- Roles explicitly described as "former", "past", "previously", "emeritus", or "ex-"
- Roles with end dates clearly in the past (e.g. "(2011 - 2013)") — only include current/active roles

Respond ONLY with a JSON object with exactly two keys:
- "title": string or null — their current academic rank as it appears on their profile
- "roles": array of role objects

Each role item must have exactly these keys:
- "category": one of "editorial", "society", "leadership", "training"
- "role": the specific role title (e.g. "Associate Editor")
- "org": the organization or journal name (e.g. "Diabetes Care")
- "source_url": URL of the webpage where you found this
- "evidence_text": a RAW VERBATIM EXCERPT (up to 800 characters) copied word-for-word from the surrounding section of the source page where this role appears. Copy the full surrounding paragraph, list block, or table section — do NOT cherry-pick only the confirming sentence. Include enough context that a separate reviewer reading only this excerpt could independently confirm or deny the role claim. Do NOT paraphrase or summarize. If you cannot find raw source text, leave this field as an empty string.
- "retrieved_at": current UTC timestamp in ISO 8601 format

If no qualifying roles are found, return {{"title": null, "roles": []}}.
No markdown fences, no explanation — only the JSON object.
"""

_TITLE_PROMPT = """\
Look up the faculty profile page for {name} at {school}.
Return ONLY a JSON object with one key:
- "title": their current academic rank exactly as listed on their institutional profile \
(e.g. "Professor", "Associate Professor", "Assistant Professor", "Clinical Professor", \
"Instructor", "Lecturer"). Return null if not clearly stated.

No markdown fences, no explanation — only the JSON object.
"""

def _clean_name_parts(full_name: str) -> list[str]:
    parts = [p.strip(".,") for p in full_name.split() if p.strip(".,") not in _CREDENTIAL_SUFFIXES]
    return [p for p in parts if p]


def _parse_json(raw: str) -> dict:
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(clean)


import functools
import os

_NIH_SEARCH_URL = "https://api.reporter.nih.gov/v2/projects/search"
_NIH_FIELDS = ["project_num", "project_serial_num", "project_title", "fiscal_year", "organization", "principal_investigators"]
_NIH_CONFIDENCE_THRESHOLD = 0.75

# Scoring weights — must sum to 1.0
# name = who we're looking for (primary)
# inst = where they work — checked against xlsx aliases (people move, so not a hard reject)
# dept = NIH dept_type field — present for university grants, null for hospital grants
_W_NAME = 0.50
_W_INST = 0.35
_W_DEPT = 0.15


def _score_dept(org_dept: str, discipline: str) -> float:
    """0.5 if no data or doesn't match; 1.0 if dept confirms discipline."""
    if not org_dept:
        return 0.5
    dept  = org_dept.lower()
    words = [w for w in discipline.lower().split() if len(w) > 3]
    if any(w in dept for w in words):
        return 1.0
    return 0.5  # present but unrelated — ignore rather than penalise


def _inst_matches(org: str, aliases) -> bool:
    """True if org name matches any alias via substring or fuzzy (≥85)."""
    org_l = org.lower()
    for alias in aliases:
        if alias in org_l or org_l in alias:
            return True
        if len(alias) >= 8 and fuzz.partial_ratio(org_l, alias) >= 85:
            return True
    return False


def _score_inst(orgs: set[str], target_aliases: tuple[str, ...], all_known: dict) -> float:
    """1.0 confirmed at target; 0.3 at a different known school; 0.1 completely unknown."""
    if not orgs:
        return 0.5
    for org in orgs:
        if _inst_matches(org, target_aliases):
            return 1.0
    for org in orgs:
        for aliases in all_known.values():
            if _inst_matches(org, aliases):
                return 0.3
    return 0.1


def _nih_org_names_for(school: str) -> list[str]:
    """
    Build org_names list for the NIH Phase 1 API query using xlsx aliases.
    Strips med-school suffixes so names match NIH's abbreviated org_name format.
    """
    aliases = _xlsx_aliases_for(school)
    if not aliases:
        parts = [p.strip() for p in school.split(" / ")]
        aliases = tuple(p.lower() for p in parts if p)

    result: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        a = alias.strip()
        if not a or a in seen:
            continue
        seen.add(a)
        result.append(a)
        # Also add version with med-school suffix stripped (NIH stores "Stanford University", not the full name)
        for suffix in _MED_SUFFIXES:
            if a.endswith(suffix):
                stripped = a[: len(a) - len(suffix)].strip()
                if stripped and stripped not in seen:
                    seen.add(stripped)
                    result.append(stripped)
                break
    return result

# Keyword map: normalized discipline string → list of title keywords (lowercase).
# At least one keyword must appear in a grant title for that grant to count as a discipline hit.
_DISCIPLINE_KEYWORDS: dict[str, list[str]] = {
    "pediatric cardiology":    ["cardiac", "cardiology", "heart", "congenital", "arrhythmia", "echocardiograph"],
    "pediatric endocrinology": ["diabetes", "endocrin", "insulin", "thyroid", "adrenal", "growth hormone", "pituitary"],
    "pediatric neurology":     ["neurol", "epilep", "seizure", "brain", "cerebral", "neonatal neurol"],
    "pediatric oncology":      ["oncol", "leukemia", "lymphoma", "tumor", "cancer", "hematol"],
    "pediatric pulmonology":   ["pulmonol", "asthma", "cystic fibrosis", "respiratory", "lung"],
    "pediatric gastroenterology": ["gastro", "inflamm bowel", "crohn", "colitis", "liver", "hepat"],
    "pediatric nephrology":    ["nephrol", "kidney", "renal", "dialysis", "glomerul"],
    "pediatric rheumatology":  ["rheumatol", "arthritis", "autoimmune", "lupus", "juvenile idiopathic"],
    "pediatric infectious disease": ["infect", "antimicrobial", "antibiotic", "sepsis", "immunodeficien"],
    "pediatric hematology":    ["hematol", "sickle cell", "hemophilia", "anemia", "coagul"],
}


def _discipline_hit_score(titles: list[str], discipline: str) -> float:
    """Fraction of grant titles matching any discipline keyword (0.0–1.0)."""
    if not discipline or not titles:
        return 0.0
    keywords = _DISCIPLINE_KEYWORDS.get(discipline.lower())
    if not keywords:
        # Generic fallback: split discipline into words and search each title
        keywords = [w for w in discipline.lower().split() if len(w) > 4]
    if not keywords:
        return 0.0
    hits = sum(1 for t in titles if any(kw in t.lower() for kw in keywords))
    return hits / len(titles)


# Suffixes stripped before searching OpenAlex (which indexes universities, not med schools)
_MED_SUFFIXES = (
    " school of medicine",
    " college of medicine",
    " medical school",
    " school of medicine and health sciences",
    " feinberg school of medicine",
    " perelman school of medicine",
    " vagelos college of physicians and surgeons",
    " keck school of medicine",
)


_OA_GENERIC = frozenset({
    "university", "the", "college", "of", "at", "a", "and",
    "school", "institute", "medical", "health",
})


def _openalex_fetch(search_term: str) -> list[dict]:
    resp = requests.get(
        "https://api.openalex.org/institutions",
        params={"search": search_term, "per_page": 25},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def _distinctive_prefix(query: str) -> str:
    """Return the run of distinctive leading words from a query.

    Stops at the first generic word.  Examples:
      "Stanford University School of Medicine" → "Stanford"
      "Johns Hopkins University"               → "Johns Hopkins"
      "University of Michigan Medical School"  → ""  (starts generic)
    """
    words = query.split()
    prefix: list[str] = []
    for w in words:
        if w.lower() in _OA_GENERIC:
            break
        prefix.append(w)
    return " ".join(prefix)


def _openalex_search_one(query: str) -> list[str]:
    """Return lowercase alias list for a single OpenAlex institution search query.

    Strategy:
    - If the query starts with a distinctive prefix ("Stanford", "Johns Hopkins"),
      search OpenAlex by that prefix to get broad recall of all related entities,
      then filter results to those whose display_name contains the prefix.
    - If the query starts with a generic word ("University of …"), use the full
      query so OpenAlex ranking returns the correct specific institution; all
      returned results are included (the specific query already limits scope).
    """
    words = query.split()
    if not words:
        return []

    prefix = _distinctive_prefix(query)
    has_prefix = bool(prefix) and len(prefix) >= 4

    items = _openalex_fetch(prefix) if has_prefix else _openalex_fetch(query)
    if not items:
        return []

    if has_prefix:
        pattern = re.compile(r"\b" + re.escape(prefix.lower()) + r"\b")
    else:
        # Fall back to filtering by the first non-generic word
        filter_word = next(
            (w.lower() for w in words if w.lower() not in _OA_GENERIC),
            words[0].lower(),
        )
        pattern = re.compile(r"\b" + re.escape(filter_word) + r"\b")

    raw: list[str] = []
    for item in items:
        dn = item.get("display_name") or ""
        if not pattern.search(dn.lower()):
            continue
        raw.append(dn)
        raw += item.get("display_name_alternatives") or []
        if item.get("acronym"):
            raw.append(item["acronym"])

    return [a.lower() for a in raw if a]


def _candidate_queries(school: str) -> list[str]:
    """
    Build a list of OpenAlex search strings to try for a school name.

    Handles:
      - "Stanford University School of Medicine"          → "Stanford University"
      - "Baylor College of Medicine / Texas Children's"  → "Baylor College of Medicine"
                                                            "Texas Children's Hospital"
      - "Harvard Medical School"                         → "Harvard University"
    """
    candidates: list[str] = []
    # Split on " / " — some schools are listed as "A / B"
    parts = [p.strip() for p in school.split(" / ")]
    for part in parts:
        candidates.append(part)
        # Also try stripping known med-school suffixes to get the parent university
        lower = part.lower()
        for suffix in _MED_SUFFIXES:
            if lower.endswith(suffix):
                stripped = part[: len(part) - len(suffix)].strip()
                if stripped:
                    candidates.append(stripped)
                break
    # Deduplicate while preserving order
    seen: set[str] = set()
    return [c for c in candidates if not (c.lower() in seen or seen.add(c.lower()))]


_XLSX_PATH = os.path.join(os.path.dirname(__file__), "..", "medical_schools_lookup.xlsx")

_PAREN_RE = re.compile(r"\s*\([^)]+\)")


def _parse_hospital_field(raw: str) -> list[str]:
    """Split a hospital field on ';', strip parentheticals, return clean names."""
    names: list[str] = []
    for part in raw.split(";"):
        part = part.strip()
        if not part or part.lower().startswith("n/a"):
            continue
        paren = re.search(r"\(([^)]+)\)", part)
        base = _PAREN_RE.sub("", part).strip()
        if base:
            names.append(base)
        if paren:
            names.append(paren.group(1).strip())
    return names


@functools.lru_cache(maxsize=1)
def _load_xlsx_aliases() -> dict[str, frozenset[str]]:
    """
    Load medical_schools_lookup.xlsx into a dict:
      lowercase school name → frozenset of lowercase alias strings
    (school name + primary teaching hospital + all major affiliated hospitals).
    """
    try:
        import pandas as pd
        df = pd.read_excel(_XLSX_PATH, sheet_name="All Schools")
    except Exception as exc:
        print(f"[Lookup] Could not load medical_schools_lookup.xlsx: {exc}")
        return {}

    result: dict[str, frozenset[str]] = {}
    for _, row in df.iterrows():
        school = str(row.get("School Name") or "").strip()
        if not school or school == "nan":
            continue
        aliases: set[str] = {school.lower()}
        for col in ("Primary Teaching Hospital", "Major Affiliated Hospitals / Health Systems"):
            val = str(row.get(col) or "").strip()
            if val and val != "nan":
                for name in _parse_hospital_field(val):
                    aliases.add(name.lower())
        result[school.lower()] = frozenset(aliases)
    print(f"[Lookup] Loaded {len(result)} schools from xlsx")
    return result


def _xlsx_aliases_for(school: str) -> tuple[str, ...] | None:
    """
    Return alias tuple for a school from the xlsx lookup.
    Handles:
      - Exact match (case-insensitive)
      - Combined names with ' / ' — tries each part separately
      - Fuzzy fallback via token_set_ratio (threshold 75)
    Returns None if no confident match found.
    """
    from rapidfuzz import process as rp, fuzz as rf
    lookup = _load_xlsx_aliases()
    if not lookup:
        return None

    def _best_match(query: str) -> frozenset[str] | None:
        q = query.lower().strip()
        if q in lookup:
            return lookup[q]
        result = rp.extractOne(q, list(lookup.keys()), scorer=rf.token_set_ratio)
        if result and result[1] >= 75:
            return lookup[result[0]]
        return None

    # Try the full name first
    found = _best_match(school)
    if found:
        return tuple(found)

    # For combined names like "Baylor / Texas Children's", try each part
    if " / " in school:
        all_aliases: set[str] = set()
        for part in school.split(" / "):
            part_match = _best_match(part.strip())
            if part_match:
                all_aliases.update(part_match)
        if all_aliases:
            return tuple(all_aliases)

    return None


@functools.lru_cache(maxsize=64)
def _get_openalex_aliases(school: str) -> tuple[str, ...]:
    """
    Return all known institution name variants for NIH grant org matching.
    Primary source: medical_schools_lookup.xlsx (school + all affiliated hospitals).
    Fallback: OpenAlex API (for institutions not in the xlsx).
    Returns a deduplicated tuple of lowercase strings (cached per school).
    """
    # Try xlsx first
    xlsx_result = _xlsx_aliases_for(school)
    if xlsx_result:
        print(f"      [Lookup] Aliases for '{school}' ({len(xlsx_result)} from xlsx): {xlsx_result}")
        return xlsx_result

    # Fall back to OpenAlex
    print(f"      [OpenAlex] '{school}' not in xlsx — querying OpenAlex…")
    queries = _candidate_queries(school)
    all_aliases: list[str] = []
    seen: set[str] = set()

    for query in queries:
        try:
            aliases = _openalex_search_one(query)
            for a in aliases:
                if a not in seen:
                    seen.add(a)
                    all_aliases.append(a)
        except Exception as e:
            print(f"      [OpenAlex] Query '{query}' failed: {e}")

    if not all_aliases:
        all_aliases = [school.lower()]

    result = tuple(all_aliases)
    print(f"      [OpenAlex] Aliases for '{school}': {result}")
    return result


def _nih_search(payload: dict) -> list[dict]:
    r = requests.post(_NIH_SEARCH_URL, json=payload, timeout=20)
    r.raise_for_status()
    return r.json().get("results", [])


def _nih_fetch_all_by_profile(profile_id: int) -> list[dict]:
    """Paginate NIH Reporter to get every grant record for a given PI profile_id."""
    all_results: list[dict] = []
    offset = 0
    limit = 500
    while True:
        payload = {
            "criteria": {"pi_profile_ids": [profile_id]},
            "fields": ["project_num", "project_serial_num", "project_title", "fiscal_year", "organization"],
            "limit": limit,
            "offset": offset,
        }
        r = requests.post(_NIH_SEARCH_URL, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
        batch = data.get("results", [])
        all_results.extend(batch)
        total = data.get("meta", {}).get("total", 0)
        offset += limit
        if offset >= total:
            break
    return all_results


class LiveNihProvider(NihProvider):
    def __init__(self, discipline: str = ""):
        self._discipline = discipline

    # ── helpers ──────────────────────────────────────────────────────────────

    def _parse_results_into_profiles(
        self, results: list[dict], first_name: str, last_name: str
    ) -> dict[int, dict]:
        """Aggregate NIH search results into per-profile-id dicts."""
        profiles: dict[int, dict] = {}
        for item in results:
            org      = item.get("organization") or {}
            org_name = (org.get("org_name") or "").lower()
            org_dept = (org.get("dept_type") or "").lower()
            for pi in (item.get("principal_investigators") or []):
                pid = pi.get("profile_id")
                if not pid:
                    continue
                pi_first = (pi.get("first_name") or "").strip().lower()
                pi_last  = (pi.get("last_name")  or "").strip().lower()
                first_sim  = fuzz.ratio(first_name.lower(), pi_first) / 100.0
                last_sim   = fuzz.ratio(last_name.lower(),  pi_last)  / 100.0
                name_score = 0.5 * first_sim + 0.5 * last_sim
                if name_score < 0.5:
                    continue
                if pid not in profiles:
                    profiles[pid] = {
                        "pi_name":    f"{pi.get('first_name','')} {pi.get('last_name','')}".strip(),
                        "name_score": name_score,
                        "orgs":       set(),
                        "depts":      set(),
                    }
                profiles[pid]["orgs"].add(org_name)
                if org_dept:
                    profiles[pid]["depts"].add(org_dept)
        return profiles

    def _grants_for_profile(
        self, profile_id: int, fallback_results: list[dict], unconfirmed: bool = False
    ) -> list[Grant]:
        """Fetch all grants for a confirmed profile_id and return them."""
        try:
            all_items = _nih_fetch_all_by_profile(profile_id)
        except Exception as e:
            print(f"      [NIH] Profile pull failed for profile_id={profile_id}: {e}")
            all_items = fallback_results
        # Deduplicate by project_serial_num (base grant ID without fiscal-year suffix),
        # keeping the most recent fiscal year entry for each underlying grant.
        best: dict[str, dict] = {}
        for item in all_items:
            serial = item.get("project_serial_num") or item.get("project_num", "Unknown")
            fy = item.get("fiscal_year") or 0
            if serial not in best or fy > (best[serial].get("fiscal_year") or 0):
                best[serial] = item
        grants: list[Grant] = []
        for item in best.values():
            g: Grant = {"project_num": item.get("project_num", "Unknown"), "title": item.get("project_title", "")}
            if unconfirmed:
                g["institution_unconfirmed"] = True
            grants.append(g)
        return grants

    # ── main entry point ─────────────────────────────────────────────────────

    def get_grants(self, name: str, school: str) -> list[Grant]:
        """
        Two-phase NIH identity matching:

        Phase 1 — targeted (high confidence):
          Search pi_names + org_names using xlsx institution aliases (hospital names +
          university short forms). If any result comes back the person has grants at
          this institution → take the best name match, pull ALL grants by profile_id.

        Phase 2 — broad fallback (scored, flagged as unconfirmed):
          No institution match found (person may have moved or hold no grants here).
          Search by name only, score candidates on name (0.50) + inst (0.35) + dept (0.15).
          dept_type is present for university grants, null for hospital grants — treated
          as neutral when absent. Returns nothing if best candidate is below threshold.
        """
        parts = _clean_name_parts(name)
        if len(parts) < 2:
            return []
        first_name, last_name = parts[0], parts[-1]

        org_names_query  = _nih_org_names_for(school)
        target_aliases   = _xlsx_aliases_for(school) or (school.lower(),)
        all_known        = _load_xlsx_aliases()

        # ── Phase 1: name + institution ───────────────────────────────────────
        print(f"      [NIH] Phase 1: searching '{first_name} {last_name}' at {school}")
        try:
            p1_results = _nih_search({
                "criteria": {
                    "pi_names":  [{"first_name": first_name, "last_name": last_name}],
                    "org_names": org_names_query,
                },
                "fields": _NIH_FIELDS,
                "limit": 500,
            })
        except Exception as e:
            print(f"      [NIH] Phase 1 search failed: {e}")
            p1_results = []

        if p1_results:
            profiles = self._parse_results_into_profiles(p1_results, first_name, last_name)
            if profiles:
                best_pid = max(profiles, key=lambda pid: profiles[pid]["name_score"])
                best     = profiles[best_pid]
                print(f"      [NIH] Phase 1 match: '{best['pi_name']}' "
                      f"(name={best['name_score']:.2f}, profile_id={best_pid})")
                grants = self._grants_for_profile(best_pid, p1_results, unconfirmed=False)
                print(f"      [NIH] {name}: {len(grants)} grant(s) — institution confirmed")
                return grants

        print(f"      [NIH] Phase 1: no grants found at {school} — trying broad search")

        # ── Phase 2: name only + scoring ──────────────────────────────────────
        try:
            p2_results = _nih_search({
                "criteria": {"pi_names": [{"first_name": first_name, "last_name": last_name}]},
                "fields": _NIH_FIELDS,
                "limit": 500,
            })
        except Exception as e:
            print(f"      [NIH] Phase 2 search failed: {e}")
            return []

        if not p2_results:
            print(f"      [NIH] No NIH grants found for {name}")
            return []

        print(f"      [NIH] Phase 2: {len(p2_results)} candidate grant(s) across all institutions")
        profiles = self._parse_results_into_profiles(p2_results, first_name, last_name)
        if not profiles:
            return []

        for prof in profiles.values():
            prof["dept_score"] = max(
                (_score_dept(d, self._discipline) for d in prof["depts"]),
                default=0.5,
            )
            prof["inst_score"] = _score_inst(prof["orgs"], target_aliases, all_known)

        def _combined(p: dict) -> float:
            return _W_NAME * p["name_score"] + _W_INST * p["inst_score"] + _W_DEPT * p["dept_score"]

        ranked        = sorted(profiles.items(), key=lambda kv: _combined(kv[1]), reverse=True)
        best_pid, best = ranked[0]
        score         = _combined(best)
        ambiguous     = len(ranked) > 1 and (score - _combined(ranked[1][1])) < 0.05

        print(f"      [NIH] Phase 2 best: '{best['pi_name']}' "
              f"(score={score:.2f} | name={best['name_score']:.2f} "
              f"inst={best['inst_score']:.2f} dept={best['dept_score']:.2f} "
              f"| profiles={len(profiles)}"
              f"{' | AMBIGUOUS' if ambiguous else ''})")

        if score < _NIH_CONFIDENCE_THRESHOLD:
            print(f"      [NIH] Low confidence (score={score:.2f}) — skipping {name}")
            return []

        grants = self._grants_for_profile(best_pid, p2_results, unconfirmed=True)
        print(f"      [NIH] {name}: {len(grants)} grant(s) — institution unconfirmed")
        return grants


class LiveRolesProvider(RolesProvider):
    def __init__(self, openai_client, discipline: str = "", considerations: str = ""):
        self._client = openai_client
        self._discipline_context = _load_discipline_context(discipline) if discipline else ""
        if considerations:
            self._discipline_context += f"\nAdditional guidance for this run:\n{considerations}\n"

    def _fetch(self, name: str, school: str) -> dict:
        """Make one web search call and return the parsed JSON (title + roles)."""
        prompt = _ROLES_PROMPT.format(name=name, school=school, discipline_context=self._discipline_context)
        response = self._client.responses.create(
            model=_LIVE_MODEL,
            tools=[{"type": "web_search_preview"}],
            input=[{"role": "user", "content": prompt}],
        )
        raw = ""
        for block in response.output:
            if hasattr(block, "content"):
                for item in block.content:
                    if hasattr(item, "text"):
                        raw += item.text
        return _parse_json(raw)

    def _process_roles(self, parsed: dict) -> list[RoleEvidence]:
        roles = parsed.get("roles", [])
        now = datetime.now(timezone.utc).isoformat()
        for role in roles:
            role.setdefault("retrieved_at", now)
            role.setdefault("model", _LIVE_MODEL)
        return roles

    def get_roles(self, name: str, school: str, profile_url: str = "") -> list[RoleEvidence]:
        return self._process_roles(self._fetch(name, school))

    def get_roles_and_title(self, name: str, school: str, profile_url: str = "") -> tuple[str | None, list[RoleEvidence]]:
        """Two web search calls: one for roles, one dedicated title lookup."""
        roles = self._process_roles(self._fetch(name, school))
        prompt = _TITLE_PROMPT.format(name=name, school=school)
        response = self._client.responses.create(
            model=_LIVE_MODEL,
            tools=[{"type": "web_search_preview"}],
            input=[{"role": "user", "content": prompt}],
        )
        raw = ""
        for block in response.output:
            if hasattr(block, "content"):
                for item in block.content:
                    if hasattr(item, "text"):
                        raw += item.text
        try:
            title = _parse_json(raw).get("title") or None
        except Exception:
            title = None
        return title, roles


# ── Scoring helpers (used by LiveHIndexProvider) ──────────────────────────────

_GENERIC_WORDS = {
    "university", "universities", "medical", "center", "school", "hospital",
    "health", "institute", "system", "care", "college", "of", "and", "the",
    "at", "for", "children", "childrens", "general", "national", "research",
    "science", "sciences", "medicine", "clinic", "foundation", "department",
    "child", "maternal", "cancer", "stem", "cell", "biology", "regenerative"
}

_CREDENTIALS = {
    "MD", "PHD", "DO", "DDS", "DMD", "DVM", "PHARMD", "DRPH",
    "DPHIL", "MBBS", "MPH", "MS", "MSC", "MSN", "MHS", "MHA", "MBA",
    "FAAP", "FACS", "FACP", "FACE", "DFAACAP", "RN", "NP", "PA",
    "JR", "SR", "II", "III", "IV",
}

def _parse_name_for_scopus(full_name: str) -> tuple[str, str, str]:
    """Return (first, middle_initial, last) stripped of credentials.

    'Anna L. Gloyn, DPhil' → ('Anna', 'L', 'Gloyn')
    'Anna Gloyn'           → ('Anna', '',  'Gloyn')
    'José M. Rodriguez'    → ('José', 'M', 'Rodriguez')
    """
    name = full_name.split(",")[0].strip()
    parts = [p.strip(".,") for p in name.split() if p.strip(".,").upper() not in _CREDENTIALS and p.strip(".,")]
    if len(parts) < 2:
        return "", "", ""
    first = parts[0]
    last  = parts[-1]
    # middle initial: any single-letter part between first and last
    middle = next((p for p in parts[1:-1] if len(p) == 1), "")
    return first, middle, last


def _scopus_affil_str(entry: dict) -> str:
    affil = entry.get("affiliation-current", {})
    if isinstance(affil, list):
        return " ".join(a.get("affiliation-name", "") for a in affil)
    if isinstance(affil, dict):
        return affil.get("affiliation-name", "")
    return ""


def _affil_matches(affil_str: str, aliases: list[str]) -> bool:
    """True if any alias fuzzy-matches the affiliation string."""
    if not affil_str:
        return False
    affil = affil_str.lower()
    affil_words = set(re.findall(r"[a-zA-Z]+", affil))
    for alias in aliases:
        d_words = [w for w in re.findall(r"[a-zA-Z]+", alias.lower())
                   if w not in _GENERIC_WORDS and len(w) > 2]
        if not d_words:
            continue
        if any(w in affil_words for w in d_words) and fuzz.partial_ratio(alias.lower(), affil) >= 85:
            return True
    return False


def _name_matches(first: str, last: str, entry: dict) -> bool:
    """True if last name matches well and first name is consistent."""
    pref    = entry.get("preferred-name", {})
    s_first = (pref.get("given-name") or "").lower().strip()
    s_last  = (pref.get("surname")    or "").lower().strip()
    if fuzz.ratio(last.lower(), s_last) < 90:
        return False
    if not first or not s_first:
        return True
    # first initial must agree
    return s_first[0] == first[0].lower()


def _scopus_get(url: str, params: dict, max_retries: int = 4) -> requests.Response:
    """GET with exponential backoff on 429 rate-limit responses."""
    import os
    headers = {"Accept": "application/json"}
    insttoken = os.environ.get("SCOPUS_INSTTOKEN")
    if insttoken:
        headers["X-ELS-Insttoken"] = insttoken
    delay = 10.0
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        retry_after = resp.headers.get("X-RateLimit-Reset") or resp.headers.get("Retry-After")
        try:
            raw_wait = float(retry_after) if retry_after else delay
            # X-RateLimit-Reset is an epoch timestamp (10+ digits); convert to relative wait
            wait = max(0.0, raw_wait - time.time()) if raw_wait > 3600 else raw_wait
        except (ValueError, TypeError):
            wait = delay
        print(f"      [Scopus] Rate limited — waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
        time.sleep(wait)
        delay = min(delay * 2, 120.0)
    resp.raise_for_status()
    return resp


def _scopus_search(query: str, api_key: str) -> list[dict]:
    resp = _scopus_get(
        "https://api.elsevier.com/content/search/author",
        params={
            "query": query,
            "field": "dc:identifier,preferred-name,affiliation-current",
            "count": 25,
            "apiKey": api_key,
        },
    )
    entries = resp.json().get("search-results", {}).get("entry", [])
    if not entries or "error" in entries[0]:
        return []
    return entries


def _scopus_fetch_profile(author_id: str, api_key: str) -> dict:
    resp = _scopus_get(
        f"https://api.elsevier.com/content/author/author_id/{author_id}",
        params={"field": "h-index,document-count,affiliation-current", "apiKey": api_key},
    )
    return resp.json().get("author-retrieval-response", [{}])[0]


class LiveHIndexProvider(HIndexProvider):
    _lock = threading.Lock()
    _last_call = 0.0

    def __init__(self, scopus_api_key: str | None = None, discipline: str = "Pediatric Endocrinology"):
        self._api_key = scopus_api_key
        self._discipline = discipline

    def get_hindex(self, name: str, school: str) -> HIndexEvidence:
        # Hold the lock for the entire call sequence — this serializes all Scopus
        # requests so a rate-limit backoff in one worker blocks others from piling on
        # (thundering-herd prevention).
        with LiveHIndexProvider._lock:
            elapsed = time.time() - LiveHIndexProvider._last_call
            if elapsed < 3.0:
                time.sleep(3.0 - elapsed)
            LiveHIndexProvider._last_call = time.time()

            now = datetime.now(timezone.utc).isoformat()

            if not self._api_key:
                return {
                    "h_index": None, "scopus_author_id": None,
                    "source_url": "", "evidence_text": "Scopus API key not set — h-index skipped",
                    "retrieved_at": now,
                }

            try:
                return self._get_hindex_inner(name, school, now)
            except Exception as exc:
                print(f"      [Scopus] Unrecoverable error for {name}: {exc}")
                return {
                    "h_index": None, "scopus_author_id": None,
                    "source_url": "", "evidence_text": f"Scopus error: {exc}",
                    "retrieved_at": now,
                }

    def _get_hindex_inner(self, name: str, school: str, now: str) -> HIndexEvidence:
        first, middle, last = _parse_name_for_scopus(name)
        if not first or not last:
            return {
                "h_index": None, "scopus_author_id": None,
                "source_url": "", "evidence_text": f"Could not parse name: {name}",
                "retrieved_at": now,
            }

        # Step 1: Search — try with middle initial first, fall back to first only
        entries = []
        for q in ([f"AUTHLASTNAME({last}) AND AUTHFIRST({first} {middle})"] if middle else []) + \
                  [f"AUTHLASTNAME({last}) AND AUTHFIRST({first})"]:
            print(f"      [Scopus] Querying: {q}")
            entries = _scopus_search(q, self._api_key)
            if entries:
                break

        if not entries:
            return {
                "h_index": None, "scopus_author_id": None,
                "source_url": "", "evidence_text": f"No Scopus results for {name}",
                "retrieved_at": now,
            }

        # Step 2: Filter to name matches only
        name_matches = [e for e in entries if _name_matches(first, last, e)]
        if not name_matches:
            return {
                "h_index": None, "scopus_author_id": None,
                "source_url": "", "evidence_text": f"No name match found in Scopus for {name}",
                "retrieved_at": now,
            }

        xlsx_aliases = _xlsx_aliases_for(school)
        aliases = list(xlsx_aliases) if xlsx_aliases else [school.lower()]

        # Step 3: Pick best candidate
        # First preference: search result already shows a matching affiliation
        affil_match = next((e for e in name_matches if _affil_matches(_scopus_affil_str(e), aliases)), None)
        if affil_match:
            best_entry = affil_match
            confidence = "high"
            print(f"      [Scopus] Affiliation match in search results")
        elif len(name_matches) == 1:
            # One name match — fetch their full profile to check current affiliation
            # (search index is often stale; profile reflects current institution)
            best_entry = name_matches[0]
            confidence = "needs_review"
            pref = best_entry.get("preferred-name", {})
            display = f"{pref.get('given-name','')} {pref.get('surname','')}".strip()
            print(f"      [Scopus] Single name match '{display}' — will verify via profile")
        else:
            # Multiple name matches, none with matching affiliation in search results
            # Try fetching profiles for each to find the one at this institution
            best_entry = None
            confidence = "needs_review"
            for candidate in name_matches:
                cid = candidate.get("dc:identifier", "").replace("AUTHOR_ID:", "")
                if not cid:
                    continue
                try:
                    prof = _scopus_fetch_profile(cid, self._api_key)
                    affil_data = prof.get("author-profile", {}).get("affiliation-current", {})
                    affil_name = ""
                    if isinstance(affil_data, dict):
                        affil_name = (affil_data.get("affiliation", {}) or {}).get("ip-doc", {}).get("sort-name", "")
                    if _affil_matches(affil_name, aliases):
                        best_entry = candidate
                        confidence = "high"
                        print(f"      [Scopus] Found affiliation match via profile fetch")
                        break
                except Exception:
                    continue
            if best_entry is None:
                pref = name_matches[0].get("preferred-name", {})
                display = f"{pref.get('given-name','')} {pref.get('surname','')}".strip()
                print(f"      [Scopus] Multiple matches for '{last}', no affiliation confirmed — skipping. Top: '{display}'")
                return {
                    "h_index": None, "scopus_author_id": None,
                    "source_url": "", "evidence_text": "Multiple name matches, no affiliation confirmed — skipped",
                    "retrieved_at": now,
                }

        author_id = best_entry.get("dc:identifier", "").replace("AUTHOR_ID:", "")
        if not author_id:
            return {
                "h_index": None, "scopus_author_id": None,
                "source_url": "", "evidence_text": "Could not extract Scopus author ID",
                "retrieved_at": now,
            }

        # Step 4: Fetch full profile for h-index (also has current affiliation)
        print(f"      [Scopus] Fetching profile for author_id={author_id}")
        profile_data  = _scopus_fetch_profile(author_id, self._api_key)
        h_index_val   = profile_data.get("h-index")
        doc_count     = profile_data.get("coredata", {}).get("document-count", "?")
        h_index       = int(h_index_val) if h_index_val is not None else None

        # If search showed stale affiliation, use profile's current affiliation for display
        profile_affil = ""
        try:
            affil_data   = profile_data.get("author-profile", {}).get("affiliation-current", {})
            profile_affil = (affil_data.get("affiliation", {}) or {}).get("ip-doc", {}).get("sort-name", "")
        except Exception:
            pass
        display_affil = profile_affil or _scopus_affil_str(best_entry)

        # Upgrade confidence if profile affiliation now matches
        if confidence == "needs_review" and _affil_matches(profile_affil, aliases):
            confidence = "high"

        pref    = best_entry.get("preferred-name", {})
        display = f"{pref.get('given-name','')} {pref.get('surname','')}".strip()
        print(f"      [Scopus] Matched '{display}' @ '{display_affil}' (confidence={confidence}, h-index={h_index})")

        return {
            "h_index": h_index,
            "scopus_author_id": author_id,
            "source_url": f"https://www.scopus.com/authid/detail.uri?authorId={author_id}",
            "evidence_text": f"Scopus: h-index={h_index}, documents={doc_count}, affil='{display_affil}', confidence={confidence}",
            "retrieved_at": now,
        }
