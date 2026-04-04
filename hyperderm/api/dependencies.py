from __future__ import annotations

from dataclasses import dataclass

from hyperderm.core.config import settings
from hyperderm.infrastructure.backup.jsonl_store import JsonlBackupStore
from hyperderm.infrastructure.clients.literature import LiteratureClient
from hyperderm.infrastructure.graph.neo4j_store import Neo4jStore
from hyperderm.mcp.langgraph_mcp_tool import LangGraphMCPTool
from hyperderm.services.diagnosis_service import DiagnosisService
from hyperderm.services.conversation_memory import ConversationMemoryService
from hyperderm.services.graph_rag_service import GraphRagService
from hyperderm.services.evidence_service import EvidenceService
from hyperderm.services.local_backup_diagnosis_service import LocalBackupDiagnosisService
from hyperderm.services.local_rag_service import LocalRagService
from hyperderm.services.medgemma_chat_service import MedGemmaChatService
from hyperderm.workflows.chatbot_graph import build_chatbot_workflow
from hyperderm.workflows.diagnosis_graph import build_diagnosis_workflow


@dataclass
class AppContainer:
    store: Neo4jStore
    backup: JsonlBackupStore
    workflow: object
    chat_mcp_tool: LangGraphMCPTool
    memory: ConversationMemoryService


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
    graph_rag_service = GraphRagService(store)
    memory_service = ConversationMemoryService(settings.backup_dir)
    workflow = build_diagnosis_workflow(
        diagnosis_service,
        evidence_service=evidence_service,
        local_rag_service=local_rag_service,
    )
    local_backup_service = LocalBackupDiagnosisService(settings.backup_dir)
    medgemma_service = MedGemmaChatService()
    chatbot_graph = build_chatbot_workflow(
        diagnosis_service=diagnosis_service,
        medgemma_service=medgemma_service,
        local_backup_service=local_backup_service,
        local_rag_service=local_rag_service,
        graph_rag_service=graph_rag_service,
        memory_service=memory_service,
        strict_neo4j_only=settings.strict_neo4j_only,
    )
    chat_mcp_tool = LangGraphMCPTool(
        name="diagnosis_chatbot_tool",
        description="Multi-agent LangGraph RAG chatbot with Neo4j diagnosis tool, local backup fallback, and MedGemma response agent",
        graph=chatbot_graph,
    )
    return AppContainer(store=store, backup=backup, workflow=workflow, chat_mcp_tool=chat_mcp_tool, memory=memory_service)
