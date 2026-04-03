from __future__ import annotations

from hyperderm.infrastructure.clients.literature import LiteratureClient


class EvidenceService:
    def __init__(self, literature_client: LiteratureClient) -> None:
        self._literature_client = literature_client

    def collect_for_disease(self, disease_name: str) -> list[dict]:
        term = f"{disease_name} dermatology diagnosis"

        pubmed_ids = self._literature_client.search_pubmed(term, retmax=3)
        pubmed_rows = self._literature_client.summarize_pubmed(pubmed_ids)
        europe_rows = self._literature_client.search_europe_pmc(term, page_size=3)
        crossref_rows = self._literature_client.search_crossref(term, rows=3)
        openalex_rows = self._literature_client.search_openalex(term, per_page=3)

        combined = [*pubmed_rows, *europe_rows, *crossref_rows, *openalex_rows]

        deduped: list[dict] = []
        seen = set()

        for idx, row in enumerate(combined):
            title = (row.get("title") or "").strip()
            if not title:
                continue
            key = (row.get("source"), row.get("source_id"), title.lower())
            if key in seen:
                continue

            seen.add(key)
            row["evidence_id"] = f"{row.get('source')}:{row.get('source_id')}:{idx}"
            deduped.append(row)

        return deduped
