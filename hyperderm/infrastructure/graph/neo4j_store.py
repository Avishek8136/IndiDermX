from __future__ import annotations

from typing import Any
from neo4j import GraphDatabase


class Neo4jStore:
    def __init__(self, uri: str, username: str, password: str, database: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def ensure_schema(self) -> None:
        constraints = [
            "CREATE CONSTRAINT disease_name_unique IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT main_class_name_unique IF NOT EXISTS FOR (m:MainClass) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT sub_class_name_unique IF NOT EXISTS FOR (s:SubClass) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT evidence_id_unique IF NOT EXISTS FOR (e:Evidence) REQUIRE e.evidenceId IS UNIQUE",
            "CREATE CONSTRAINT source_name_unique IF NOT EXISTS FOR (s:Source) REQUIRE s.name IS UNIQUE",
            "CREATE INDEX morphology_name_index IF NOT EXISTS FOR (m:Morphology) ON (m.name)",
            "CREATE INDEX body_region_name_index IF NOT EXISTS FOR (b:BodyRegion) ON (b.name)",
            "CREATE CONSTRAINT visual_feature_id_unique IF NOT EXISTS FOR (v:VisualFeaturePrototype) REQUIRE v.featureId IS UNIQUE",
            "CREATE CONSTRAINT visual_feature_natural_key_unique IF NOT EXISTS FOR (v:VisualFeaturePrototype) REQUIRE v.featureNaturalKey IS UNIQUE",
            "CREATE CONSTRAINT visual_atom_name_unique IF NOT EXISTS FOR (a:VisualAtom) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT symptom_name_unique IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT effect_name_unique IF NOT EXISTS FOR (e:Effect) REQUIRE e.name IS UNIQUE",
        ]
        with self._driver.session(database=self._database) as session:
            for query in constraints:
                session.run(query)

    def clear_graph(self) -> None:
        with self._driver.session(database=self._database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def upsert_disease_hierarchy(self, item: dict[str, Any]) -> None:
        query = """
MERGE (mc:MainClass {name: $main_class})
MERGE (sc:SubClass {name: $sub_class})
MERGE (d:Disease {name: $disease})
MERGE (mc)-[:HAS_SUB_CLASS]->(sc)
MERGE (sc)-[:HAS_DISEASE]->(d)
WITH d
FOREACH (morph IN $morphologies |
  MERGE (mo:Morphology {name: morph})
  MERGE (d)-[:PRESENTS_WITH]->(mo)
)
FOREACH (region IN $body_regions |
  MERGE (br:BodyRegion {name: region})
  MERGE (d)-[:COMMON_AT]->(br)
)
"""
        with self._driver.session(database=self._database) as session:
            session.run(query, item)

    def upsert_evidence(self, item: dict[str, Any]) -> None:
        query = """
MERGE (d:Disease {name: $disease})
MERGE (s:Source {name: $source})
MERGE (e:Evidence {evidenceId: $evidence_id})
SET e.title = $title,
    e.journal = $journal,
    e.pubdate = $pubdate,
    e.doi = $doi,
    e.lastUpdatedAt = datetime()
MERGE (d)-[:SUPPORTED_BY]->(e)
MERGE (e)-[:FROM_SOURCE]->(s)
"""
        with self._driver.session(database=self._database) as session:
            session.run(query, item)

    def upsert_visual_feature(self, item: dict[str, Any]) -> None:
        query = """
MERGE (d:Disease {name: $disease})
MERGE (v:VisualFeaturePrototype {featureNaturalKey: $feature_natural_key})
SET v.featureId = $feature_id,
    v.featureNaturalKey = $feature_natural_key,
    v.muRef = $mu_ref,
    v.kappa = $kappa,
    v.descriptorSignature = $descriptor_signature,
    v.descriptorCount = $descriptor_count,
    v.conditionName = $condition_name,
    v.conditionKey = $condition_key,
    v.morphologySummary = $morphology_summary,
    v.extractedBy = $extracted_by,
    v.extractionStatus = $extraction_status,
    v.extractionErrorCode = $extraction_error_code,
    v.updatedAt = datetime()
REMOVE v.imagePathRef, v.descriptorTokens
MERGE (d)-[:ASSOCIATED_WITH_FEATURE]->(v)
"""
        with self._driver.session(database=self._database) as session:
            session.run(query, item)

    def upsert_symptoms_effects(self, item: dict[str, Any]) -> None:
        query = """
MERGE (d:Disease {name: $disease})
WITH d
FOREACH (symptom IN $symptoms |
    MERGE (s:Symptom {name: symptom})
    MERGE (d)-[:HAS_SYMPTOM]->(s)
)
FOREACH (effect IN $effects |
    MERGE (e:Effect {name: effect})
    MERGE (d)-[:MAY_CAUSE]->(e)
)
"""
        with self._driver.session(database=self._database) as session:
            session.run(query, item)

    def upsert_visual_atoms(self, item: dict[str, Any]) -> None:
        query = """
MERGE (v:VisualFeaturePrototype {featureNaturalKey: $feature_natural_key})
WITH v
FOREACH (atom IN $atoms |
    MERGE (a:VisualAtom {name: atom})
    MERGE (v)-[:HAS_VISUAL_ATOM]->(a)
)
"""
        with self._driver.session(database=self._database) as session:
            session.run(query, item)

    def fetch_top_candidates(
        self,
        descriptors: list[str],
        body_part: str,
        symptoms: list[str],
        effects: list[str],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        query = """
MATCH (d:Disease)-[:PRESENTS_WITH]->(m:Morphology)
OPTIONAL MATCH (d)-[:COMMON_AT]->(b:BodyRegion)
OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (d)-[:MAY_CAUSE]->(e:Effect)
OPTIONAL MATCH (d)-[:ASSOCIATED_WITH_FEATURE]->(:VisualFeaturePrototype)-[:HAS_VISUAL_ATOM]->(va:VisualAtom)
WITH d,
     count(CASE WHEN toLower(m.name) IN $descriptors THEN 1 END) AS descScore,
     count(CASE WHEN toLower(b.name) = $body_part THEN 1 END) AS bodyScore,
     count(CASE WHEN toLower(s.name) IN $symptoms THEN 1 END) AS symptomScore,
    count(CASE WHEN toLower(e.name) IN $effects THEN 1 END) AS effectScore,
    count(CASE WHEN toLower(va.name) IN $descriptors THEN 1 END) AS visualAtomScore
WITH d, (descScore + bodyScore + symptomScore + effectScore + visualAtomScore) AS totalScore
ORDER BY totalScore DESC
LIMIT $limit
MATCH (sc:SubClass)-[:HAS_DISEASE]->(d)
MATCH (mc:MainClass)-[:HAS_SUB_CLASS]->(sc)
OPTIONAL MATCH (d)-[:SUPPORTED_BY]->(e:Evidence)
RETURN mc.name AS main_class,
       sc.name AS sub_class,
       d.name AS disease,
       totalScore AS score,
       collect({title: e.title, source: e.evidenceId, doi: e.doi})[0..3] AS evidence
"""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                query,
                {
                    "descriptors": [descriptor.lower() for descriptor in descriptors],
                    "body_part": body_part.lower(),
                    "symptoms": [symptom.lower() for symptom in symptoms],
                    "effects": [effect.lower() for effect in effects],
                    "limit": limit,
                },
            )
            return [
                {
                    "main_class": row["main_class"],
                    "sub_class": row["sub_class"],
                    "disease": row["disease"],
                    "score": float(row["score"] or 0),
                    "evidence": row["evidence"],
                }
                for row in result
            ]
