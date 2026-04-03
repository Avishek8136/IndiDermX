from __future__ import annotations

from typing import Any

from hyperderm.infrastructure.graph.neo4j_store import Neo4jStore


class GraphRagService:
    def __init__(self, store: Neo4jStore) -> None:
        self._store = store

    def retrieve_context(self, disease_name: str, limit: int = 5) -> list[dict[str, Any]]:
        has_supported_by = self._store.has_relationship_type("SUPPORTED_BY")
        has_title = self._store.has_property_key("title")
        has_evidence_id = self._store.has_property_key("evidenceId")
        has_doi = self._store.has_property_key("doi")

        if has_supported_by:
            title_expr = "ev.title" if has_title else "''"
            source_expr = "ev.evidenceId" if has_evidence_id else "''"
            doi_expr = "ev.doi" if has_doi else "''"
            evidence_branch = f"collect(DISTINCT {{title: {title_expr}, source: {source_expr}, doi: {doi_expr}}})[0..3] AS evidence,"
        else:
            evidence_branch = "[] AS evidence,"

        query = f"""
    MATCH (d:Disease {{name: $disease}})
OPTIONAL MATCH (d)-[:PRESENTS_WITH]->(m:Morphology)
OPTIONAL MATCH (d)-[:COMMON_AT]->(b:BodyRegion)
OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (d)-[:MAY_CAUSE]->(e:Effect)
{'OPTIONAL MATCH (d)-[:SUPPORTED_BY]->(ev:Evidence)' if has_supported_by else ''}
OPTIONAL MATCH (d)<-[:HAS_DISEASE]-(sc:SubClass)<-[:HAS_SUB_CLASS]-(mc:MainClass)
WITH d,
     collect(DISTINCT toLower(m.name)) AS morphologies,
     collect(DISTINCT toLower(b.name)) AS body_regions,
     collect(DISTINCT toLower(s.name)) AS symptoms,
     collect(DISTINCT toLower(e.name)) AS effects,
     {evidence_branch}
     head(collect(DISTINCT mc.name)) AS main_class,
     head(collect(DISTINCT sc.name)) AS sub_class
RETURN {{
    disease: d.name,
    main_class: main_class,
    sub_class: sub_class,
    morphologies: morphologies,
    body_regions: body_regions,
    symptoms: symptoms,
    effects: effects,
    evidence: evidence
}} AS profile
LIMIT 1
"""
        with self._store._driver.session(database=self._store._database) as session:
            row = session.run(query, {"disease": disease_name}).single()
            if not row:
                return []
            profile = row["profile"]
            return [profile] if profile else []
