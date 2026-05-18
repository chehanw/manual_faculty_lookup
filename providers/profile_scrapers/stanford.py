"""
StanfordProfileScraper — extracts roles from profiles.stanford.edu individual pages.

Confirmed HTML structure (static SSR, no JS needed):
  div.content-section
    h3                    ← section heading
    ul.section-listing
      li.section-list-item ← one role per item, format: "Role, Org (YYYY - YYYY|Present)"

Relevant sections:
  "Boards, Advisory Committees, Professional Organizations" → editorial + society
  "Administrative Appointments"                             → leadership + training
"""
import re
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

from providers.base import RoleEvidence
from providers.profile_scrapers.base import ProfileScraper

_HANDLES_DOMAIN = "profiles.stanford.edu"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FacultyPipelineBot/1.0)"
}

_MODEL_TAG = "stanford_profile_scraper"

# Sections to parse and their default category when no keyword matches
_TARGET_SECTIONS: dict[str, str] = {
    "Boards, Advisory Committees, Professional Organizations": "society",
    "Administrative Appointments": "leadership",
}

# Strips trailing date ranges: "(2011 - 2013)" or "(2016 - Present)"
_DATE_RE = re.compile(r"\s*\(\d{4}\s*-\s*(?:\d{4}|Present)\)\s*$", re.IGNORECASE)
_PRESENT_RE = re.compile(r"\(.*?Present\)", re.IGNORECASE)

# Classification keyword sets — checked in priority order
# leadership > training > editorial > society (section default)
_LEADERSHIP = [
    "chief", "dean", "division director", "center director",
    "department director", "vice chair", "associate chair",
    "associate dean", "vice president of",
]
_TRAINING = [
    "fellowship program", "training program", "program director",
    "fellowship director", "clerkship director", "residency program",
    "education director", "scholarly concentration",
]
_EDITORIAL = [
    "editor", "editorial",
]


def _classify(text: str, section_default: str) -> str:
    t = text.lower()
    if any(kw in t for kw in _LEADERSHIP):
        return "leadership"
    if any(kw in t for kw in _TRAINING):
        return "training"
    if any(kw in t for kw in _EDITORIAL):
        return "editorial"
    return section_default


def _parse_item(raw: str) -> tuple[str, str, bool]:
    """
    Parse one li text into (role, org, is_current).

    Input:  "Associate Editor, Diabetes Care (2024 - Present)"
    Output: ("Associate Editor", "Diabetes Care", True)
    """
    is_current = bool(_PRESENT_RE.search(raw))
    clean = _DATE_RE.sub("", raw).strip()
    if ", " in clean:
        idx = clean.index(", ")
        role = clean[:idx].strip()
        org = clean[idx + 2:].strip()
    else:
        role = clean
        org = ""
    return role, org, is_current


def _scrape_html(html: str, profile_url: str, include_historical: bool) -> list[RoleEvidence]:
    soup = BeautifulSoup(html, "html.parser")
    now = datetime.now(timezone.utc).isoformat()

    seen: set[str] = set()          # deduplicate across sections
    results: list[RoleEvidence] = []

    for section in soup.select("div.content-section"):
        h3 = section.select_one("h3")
        if not h3:
            continue
        heading = h3.get_text(strip=True)
        section_default = _TARGET_SECTIONS.get(heading)
        if section_default is None:
            continue

        for li in section.select("li.section-list-item"):
            raw = li.get_text(strip=True)
            if not raw:
                continue

            role, org, is_current = _parse_item(raw)

            if not include_historical and not is_current:
                continue

            # Deduplicate by normalised role+org key
            key = (role.lower(), org.lower())
            if key in seen:
                continue
            seen.add(key)

            category = _classify(raw, section_default)

            results.append(
                RoleEvidence(
                    category=category,
                    role=role,
                    org=org,
                    source_url=profile_url,
                    evidence_text=raw,
                    retrieved_at=now,
                    model=_MODEL_TAG,
                )
            )

    return results


class StanfordProfileScraper(ProfileScraper):

    def handles(self, profile_url: str) -> bool:
        return _HANDLES_DOMAIN in profile_url

    def get_roles(
        self,
        profile_url: str,
        include_historical: bool = False,
    ) -> list[RoleEvidence]:
        try:
            resp = requests.get(profile_url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
        except Exception as e:
            print(f"    [ProfileScraper] Failed to fetch {profile_url}: {e}")
            return []

        return _scrape_html(resp.text, profile_url, include_historical)
