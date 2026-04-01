from __future__ import annotations

from dataclasses import dataclass

from hyperderm.core.config import settings
from hyperderm.infrastructure.backup.jsonl_store import JsonlBackupStore
from hyperderm.infrastructure.graph.neo4j_store import Neo4jStore
from hyperderm.services.diagnosis_service import DiagnosisService
from hyperderm.workflows.diagnosis_graph import build_diagnosis_workflow


@dataclass
class AppContainer:
    store: Neo4jStore
    backup: JsonlBackupStore
    workflow: object


def create_container() -> AppContainer:
    store = Neo4jStore(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    diagnosis_service = DiagnosisService(store)
    workflow = build_diagnosis_workflow(diagnosis_service)
    backup = JsonlBackupStore(settings.backup_dir)
    return AppContainer(store=store, backup=backup, workflow=workflow)
