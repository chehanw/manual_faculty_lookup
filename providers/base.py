"""
Abstract base classes for data providers.

Implementations:
  PlaywrightScraperProvider                       — real network call (HTML scrape)
  ReplayScraperProvider                           — loads fixtures from disk
  LiveNihProvider    / LiveRolesProvider    / LiveHIndexProvider   — real network calls
  ReplayNihProvider  / ReplayRolesProvider  / ReplayHIndexProvider — loads fixtures from disk
"""
from abc import ABC, abstractmethod
from typing import TypedDict


class Grant(TypedDict, total=False):
    project_num: str        # required
    title: str              # required
    institution_unconfirmed: bool  # True when matched via fallback — current institution not verified


class RoleEvidence(TypedDict, total=False):
    # Required
    category: str       # "editorial" | "society" | "leadership" | "training"
    role: str           # e.g. "Associate Editor"
    org: str            # e.g. "Diabetes Care"
    source_url: str     # https://...
    evidence_text: str  # short snippet from source
    retrieved_at: str   # ISO UTC timestamp
    # Optional
    model: str
    run_id: str
    confidence: float
    url_verification: dict  # added at capture time


class HIndexEvidence(TypedDict, total=False):
    h_index: int | None
    scopus_author_id: str | None
    source_url: str
    evidence_text: str
    retrieved_at: str


ALLOWED_CATEGORIES = frozenset({"editorial", "society", "leadership", "training"})


class FacultyRecord(TypedDict, total=False):
    full_name: str
    title: str
    profile_url: str | None
    school: str
    subspecialty: str | None
    email: str | None


class ScraperProvider(ABC):
    @abstractmethod
    def get_faculty(self, school: str, discipline: str) -> list[FacultyRecord]:
        """Return faculty records for the given school and discipline."""


class NihProvider(ABC):
    @abstractmethod
    def get_grants(self, name: str, school: str) -> list[Grant]:
        """Return grants for the given PI name and school."""


class RolesProvider(ABC):
    @abstractmethod
    def get_roles(self, name: str, school: str, profile_url: str = "") -> list[RoleEvidence]:
        """Return evidence-backed role objects for the given faculty member."""

    def get_roles_and_title(self, name: str, school: str, profile_url: str = "") -> "tuple[str | None, list[RoleEvidence]]":
        """Return (academic_title, roles). Default falls back to get_roles with no title."""
        return None, self.get_roles(name, school, profile_url)


class HIndexProvider(ABC):
    @abstractmethod
    def get_hindex(self, name: str, school: str) -> HIndexEvidence:
        """Return h-index evidence for the given faculty member."""
