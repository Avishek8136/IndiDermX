from __future__ import annotations

from dataclasses import dataclass

from hyperderm.core.config import settings
from hyperderm.infrastructure.backup.jsonl_store import JsonlBackupStore
from hyperderm.infrastructure.clients.literature import LiteratureClient
from hyperderm.infrastructure.graph.neo4j_store import Neo4jStore
from hyperderm.services.diagnosis_service import DiagnosisService
from hyperderm.services.evidence_service import EvidenceService
from hyperderm.services.local_rag_service import LocalRagService
from hyperderm.workflows.diagnosis_graph import build_diagnosis_workflow


@dataclass
class AppContainer:
    store: Neo4jStore
    backup: JsonlBackupStore
    workflow: object


def create_container() -> AppContainer:
    backup = JsonlBackupStore(settings.backup_dir)
    store = Neo4jStore(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    diagnosis_service = DiagnosisService(store)
    literature_client = LiteratureClient(
        crossref_mailto=settings.crossref_mailto,
        ncbi_api_key=settings.ncbi_api_key,
    )
    evidence_service = EvidenceService(literature_client)
    local_rag_service = LocalRagService(settings.backup_dir)
    workflow = build_diagnosis_workflow(
        diagnosis_service,
        evidence_service=evidence_service,
        local_rag_service=local_rag_service,
    )
    return AppContainer(store=store, backup=backup, workflow=workflow)
