"""
Abstract base class for individual faculty profile scrapers.

Each implementation handles one institution's profile pages and extracts
structured role evidence without LLM calls.
"""
from abc import ABC, abstractmethod
from typing import TypedDict

from providers.base import RoleEvidence


class ProfileData(TypedDict):
    title: str | None
    roles: list[RoleEvidence]


class ProfileScraper(ABC):

    @abstractmethod
    def handles(self, profile_url: str) -> bool:
        """Return True if this scraper can handle the given profile URL domain."""

    @abstractmethod
    def get_roles(
        self,
        profile_url: str,
        include_historical: bool = False,
    ) -> list[RoleEvidence]:
        """
        Fetch structured roles from a single faculty profile page.

        Args:
            profile_url:        Full URL to the individual faculty profile.
            include_historical: If False (default), return only active roles
                                (those whose date range includes 'Present').
                                If True, return all roles found on the page.

        Returns:
            list[RoleEvidence] with category, role, org, source_url,
            evidence_text, retrieved_at, and model populated.
            Empty list if no roles found or URL unreachable.
        """
