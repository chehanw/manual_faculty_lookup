"""
PlaywrightProfileScraper — general-purpose catch-all profile scraper.

Uses Playwright to fetch any faculty profile URL (handles JS rendering and
cookie banners), extracts the main content text, then calls GPT-4o locally
(no web search) to extract structured role evidence from the bio.

This replaces the TavilyProfileScraper stub and works for any school without
a dedicated structural scraper (e.g. Keck, Loyola, etc.).
"""
import json
import os
from datetime import datetime, timezone

from openai import OpenAI

from providers.base import RoleEvidence
from providers.profile_scrapers.base import ProfileData, ProfileScraper
from providers.utils import fetch_page_html, html_to_text

_MODEL = "gpt-4o"
_MODEL_TAG = "playwright_profile_scraper"

_EXTRACTION_PROMPT = """\
You are extracting structured data from a faculty profile bio.

Profile URL: {profile_url}

Bio text:
{bio_text}

Extract two things:

1. "title": The faculty member's academic rank/title string (e.g. "Professor of Surgery", \
"Associate Professor of Medicine", "Assistant Professor"). Return null if not stated.

2. "roles": Named, specific professional roles. Do NOT include:
   - Academic ranks themselves (Professor, Assistant Professor, Instructor, Lecturer, Attending)
   - General employment or affiliation (e.g. "Faculty at X")
   - Degrees or credentials (MD, PhD, etc.)

Each role item must have:
- "category": one of "editorial", "society", "leadership", "training"
- "role": specific role title (e.g. "Chief", "Associate Editor", "Program Director")
- "org": organization or journal name
- "evidence_text": the exact quote from the bio that confirms this role

Return ONLY a JSON object with keys "title" and "roles".
If no qualifying roles are found, return {{"title": null, "roles": []}}.
No markdown, no explanation — only the JSON object.
"""


def _fetch_page_text(url: str, timeout: int = 20) -> str:
    """Fetch a profile page and return cleaned plain text."""
    html = fetch_page_html(url, timeout)
    return html_to_text(html)


def _extract_via_gpt4o(
    bio_text: str,
    profile_url: str,
    client: OpenAI,
) -> tuple[str | None, list[RoleEvidence]]:
    """Returns (title, roles) extracted from bio_text in a single GPT-4o call."""
    prompt = _EXTRACTION_PROMPT.format(
        profile_url=profile_url,
        bio_text=bio_text[:12000],  # stay well within token budget
    )
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(raw)

    title: str | None = parsed.get("title") or None

    now = datetime.now(timezone.utc).isoformat()
    roles: list[RoleEvidence] = []
    for item in parsed.get("roles", []):
        if not isinstance(item, dict):
            continue
        roles.append(RoleEvidence(
            category=item.get("category", "society"),
            role=item.get("role", ""),
            org=item.get("org", ""),
            source_url=profile_url,
            evidence_text=item.get("evidence_text", ""),
            retrieved_at=now,
            model=_MODEL_TAG,
        ))
    return title, roles


class PlaywrightProfileScraper(ProfileScraper):
    """
    General-purpose profile scraper. Accepts any profile URL.
    Uses Playwright to fetch the page, GPT-4o to extract roles from bio text.
    """

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self._client = OpenAI(api_key=api_key)

    def handles(self, profile_url: str) -> bool:
        return bool(profile_url)  # catch-all — accepts any non-empty URL

    def get_profile_data(
        self,
        profile_url: str,
        include_historical: bool = False,
    ) -> ProfileData:
        try:
            text = _fetch_page_text(profile_url)
            print(f"    [PlaywrightProfileScraper] Fetched {len(text)} chars from {profile_url}")
        except Exception as e:
            print(f"    [PlaywrightProfileScraper] Fetch failed ({e}): {profile_url}")
            return ProfileData(title=None, roles=[])

        if len(text) < 100:
            print(f"    [PlaywrightProfileScraper] Page too thin ({len(text)} chars), skipping")
            return ProfileData(title=None, roles=[])

        try:
            title, roles = _extract_via_gpt4o(text, profile_url, self._client)
            print(f"    [PlaywrightProfileScraper] Extracted title={title!r}, {len(roles)} roles")
            return ProfileData(title=title, roles=roles)
        except Exception as e:
            print(f"    [PlaywrightProfileScraper] GPT-4o extraction failed: {e}")
            return ProfileData(title=None, roles=[])

    def get_roles(
        self,
        profile_url: str,
        include_historical: bool = False,
    ) -> list[RoleEvidence]:
        return self.get_profile_data(profile_url, include_historical)["roles"]
