from __future__ import annotations

import json
import uuid

from hyperderm.core.config import settings
from hyperderm.infrastructure.backup.jsonl_store import JsonlBackupStore
from hyperderm.infrastructure.clients.literature import LiteratureClient
from hyperderm.infrastructure.graph.neo4j_store import Neo4jStore
from hyperderm.services.diagnosis_service import DiagnosisService
from hyperderm.services.evidence_service import EvidenceService
from hyperderm.services.local_rag_service import LocalRagService
from hyperderm.workflows.diagnosis_graph import build_diagnosis_workflow


def main() -> None:
    store = Neo4jStore(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    backup = JsonlBackupStore(settings.backup_dir)

    try:
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

        state = workflow.invoke(
            {
                "descriptors": ["erythema", "plaque", "scaling"],
                "body_part": "elbow",
                "image_path": None,
            }
        )
        output = state.get("final", {})

        print(json.dumps(output, indent=2))

        backup.append(
            backup.inference_audit,
            {
                "runId": str(uuid.uuid4()),
                "model": settings.bytez_model,
                "output": output,
            },
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
