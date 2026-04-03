from __future__ import annotations

from typing import Any
import httpx


class LiteratureClient:
    def __init__(self, crossref_mailto: str, ncbi_api_key: str | None = None) -> None:
        self._crossref_mailto = crossref_mailto
        self._ncbi_api_key = ncbi_api_key

    @staticmethod
    def _get(url: str, params: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=20.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    def search_pubmed(self, term: str, retmax: int = 5) -> list[str]:
        params = {
            "db": "pubmed",
            "term": term,
            "retmax": retmax,
            "retmode": "json",
        }
        if self._ncbi_api_key:
            params["api_key"] = self._ncbi_api_key

        data = self._get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params)
        return data.get("esearchresult", {}).get("idlist", [])

    def summarize_pubmed(self, pmids: list[str]) -> list[dict[str, Any]]:
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }
        if self._ncbi_api_key:
            params["api_key"] = self._ncbi_api_key

        data = self._get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", params)
        result = data.get("result", {})

        rows: list[dict[str, Any]] = []
        for pmid in pmids:
            row = result.get(pmid)
            if not row:
                continue
            rows.append(
                {
                    "source": "PubMed",
                    "source_id": pmid,
                    "title": row.get("title", ""),
                    "journal": row.get("fulljournalname", ""),
                    "pubdate": row.get("pubdate", ""),
                    "doi": "",
                }
            )
        return rows

    def search_europe_pmc(self, term: str, page_size: int = 5) -> list[dict[str, Any]]:
        params = {
            "query": term,
            "pageSize": page_size,
            "format": "json",
        }
        data = self._get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params)
        results = data.get("resultList", {}).get("result", [])
        return [
            {
                "source": "EuropePMC",
                "source_id": row.get("id", ""),
                "title": row.get("title", ""),
                "journal": row.get("journalTitle", ""),
                "pubdate": row.get("firstPublicationDate", ""),
                "doi": row.get("doi", ""),
            }
            for row in results
        ]

    def search_crossref(self, term: str, rows: int = 5) -> list[dict[str, Any]]:
        params = {
            "query": term,
            "mailto": self._crossref_mailto,
            "rows": rows,
        }
        data = self._get("https://api.crossref.org/works", params)
        items = data.get("message", {}).get("items", [])

        output: list[dict[str, Any]] = []
        for item in items:
            title_list = item.get("title") or []
            container_list = item.get("container-title") or []
            output.append(
                {
                    "source": "Crossref",
                    "source_id": item.get("DOI", ""),
                    "title": title_list[0] if title_list else "",
                    "journal": container_list[0] if container_list else "",
                    "pubdate": item.get("created", {}).get("date-time", ""),
                    "doi": item.get("DOI", ""),
                }
            )
        return output

    def search_openalex(self, term: str, per_page: int = 5) -> list[dict[str, Any]]:
        params = {
            "search": term,
            "per-page": per_page,
            "mailto": self._crossref_mailto,
        }
        data = self._get("https://api.openalex.org/works", params)
        results = data.get("results", [])

        output: list[dict[str, Any]] = []
        for row in results:
            output.append(
                {
                    "source": "OpenAlex",
                    "source_id": row.get("id", ""),
                    "title": row.get("title", ""),
                    "journal": (row.get("primary_location") or {}).get("source", {}).get("display_name", ""),
                    "pubdate": row.get("publication_date", ""),
                    "doi": row.get("doi", ""),
                }
            )
        return output

    def fetch_wikipedia_summary(self, disease_name: str) -> str:
        # Use REST summary endpoint to avoid downloading full page HTML.
        title = disease_name.strip().replace(" ", "_")
        if not title:
            return ""

        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        try:
            data = self._get(url, params={})
        except Exception:  # noqa: BLE001
            data = {}

        extract = data.get("extract")
        if isinstance(extract, str) and extract.strip():
            return extract.strip()

        # Fallback: search closest Wikipedia page title then fetch its summary.
        try:
            search_data = self._get(
                "https://en.wikipedia.org/w/api.php",
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": f"{disease_name} dermatology",
                    "format": "json",
                    "srlimit": 1,
                },
            )
            search_rows = search_data.get("query", {}).get("search", [])
            if not search_rows:
                return ""
            candidate_title = str(search_rows[0].get("title", "")).strip().replace(" ", "_")
            if not candidate_title:
                return ""

            summary_data = self._get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{candidate_title}",
                params={},
            )
            candidate_extract = summary_data.get("extract")
            if isinstance(candidate_extract, str):
                return candidate_extract.strip()
        except Exception:  # noqa: BLE001
            return ""

        return ""
