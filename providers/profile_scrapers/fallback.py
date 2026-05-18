"""
TavilyProfileScraper — stub for schools without a dedicated profile scraper.

Intended implementation: use the Tavily Search API to find and retrieve
role information for faculty at institutions whose profile pages don't have
a structured scraper. Wire in a Tavily MCP or API client here.

Not implemented yet — returns empty list so agent2_enrich falls back to
the existing LiveRolesProvider (GPT-4o web search).
"""
from providers.base import RoleEvidence
from providers.profile_scrapers.base import ProfileScraper


class TavilyProfileScraper(ProfileScraper):

    def handles(self, profile_url: str) -> bool:
        # Accepts any URL that no other scraper claims
        return True

    def get_roles(
        self,
        profile_url: str,
        include_historical: bool = False,
    ) -> list[RoleEvidence]:
        # TODO: implement via Tavily Search API / MCP
        return []
