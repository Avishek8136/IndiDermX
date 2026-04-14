from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from tqdm import tqdm

from hyperderm.core.config import settings
from hyperderm.infrastructure.backup.jsonl_store import JsonlBackupStore
from hyperderm.infrastructure.clients.literature import LiteratureClient
from hyperderm.infrastructure.graph.neo4j_store import Neo4jStore
from hyperderm.services.evidence_service import EvidenceService
from hyperderm.services.feature_extractor import extract_visual_features
from hyperderm.services.privacy import assert_privacy_safe, sanitize_case_row
from hyperderm.services.symptom_effect_service import SymptomEffectService


def normalize_label(value: str | None, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    # Remove wrapping quotes and normalize whitespace/newlines.
    text = text.strip("'\"` ")
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text or default


def split_csv_value(value: str | None) -> list[str]:
    if not value:
        return []
    items: list[str] = []
    seen: set[str] = set()
    for part in value.replace(";", ",").split(","):
        token = normalize_label(part)
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        items.append(token)
    return items


def load_rows(csv_file_path: str | None) -> list[dict]:
    if not csv_file_path:
        return []
    path = Path(csv_file_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def resolve_image_path(dataset_csv_path: str | None, image_name: str | None) -> str | None:
    if not dataset_csv_path or not image_name:
        return None

    csv_dir = Path(dataset_csv_path).parent
    candidate_paths = [
        csv_dir / image_name,
        csv_dir / "DATASET_0" / image_name,
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)
    return str(candidate_paths[-1])


def chunked(values: list[dict] | list[str], size: int):
    if size <= 0:
        size = 25
    for index in range(0, len(values), size):
        yield values[index:index + size]


def main() -> None:
    backup = JsonlBackupStore(settings.backup_dir)
    backup.write_schema_version(schema_version="1.0.0", pipeline_version="0.2.0")

    store = Neo4jStore(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    literature_client = LiteratureClient(
        crossref_mailto=settings.crossref_mailto,
        ncbi_api_key=settings.ncbi_api_key,
    )
    evidence_service = EvidenceService(literature_client)
    symptom_effect_service = SymptomEffectService()

    progress_file = Path(settings.backup_dir) / "build_progress.json"
    start_index = 0
    if settings.build_resume and progress_file.exists():
        try:
            start_index = int(json.loads(progress_file.read_text(encoding="utf-8")).get("last_completed_row", 0))
        except Exception:  # noqa: BLE001
            start_index = 0

    try:
        if settings.clear_graph_on_start and start_index == 0:
            store.clear_graph()
        store.ensure_schema()
        raw_rows = load_rows(settings.dataset_csv_path)
        if start_index > 0:
            raw_rows = raw_rows[start_index:]
        diseases: set[str] = set()
        hierarchy_upserts = 0
        feature_upserts = 0
        evidence_upserts = 0
        symptom_effect_upserts = 0
        visual_cache: dict[tuple[str, tuple[str, ...]], dict] = {}
        graph_row_seen: set[tuple[str, str, str, tuple[str, ...], tuple[str, ...]]] = set()

        total_rows = len(raw_rows)
        batch_size = max(int(settings.build_batch_size), 1)
        batch_pause = max(float(settings.build_batch_pause_seconds), 0.0)
        print(f"Loaded {total_rows} rows from dataset chunk (resume offset: {start_index}).")
        print(f"Batch processing enabled: batch_size={batch_size}, batch_pause_seconds={batch_pause}")

        api_bar = tqdm(total=total_rows, desc="API calls", unit="row", position=0)
        neo4j_bar = tqdm(total=total_rows, desc="Neo4j updates", unit="row", position=1)
        try:
            row_index = start_index
            for batch_no, row_batch in enumerate(chunked(raw_rows, batch_size), start=1):
                print(f"Processing row batch {batch_no}: size={len(row_batch)}")
                for raw in row_batch:
                    row_index += 1
                    idx = row_index
                    row = sanitize_case_row(raw)
                    assert_privacy_safe(row)

                    disease = normalize_label(row.get("disease_label") or row.get("disease"), "Unknown Disease")
                    main_class = normalize_label(row.get("main_class"), "Other")
                    sub_class = normalize_label(row.get("sub_class"), "Unspecified")
                    morphologies = split_csv_value(row.get("descriptors"))
                    body_regions = split_csv_value(row.get("body_part"))
                    image_name = normalize_label(row.get("image_name") or row.get("imagename"), "")
                    image_path = resolve_image_path(settings.dataset_csv_path, image_name)

                    diseases.add(disease)

                # Step 1: external model/API call for visual feature extraction.
                # Reuse features for repeated disease+morphology signatures to speed up full builds.
                    cache_key = (disease.lower(), tuple(sorted(part.lower() for part in morphologies)))
                    visual_features = visual_cache.get(cache_key)
                    if visual_features is None:
                        try:
                            visual_features = extract_visual_features(
                                image_path=image_path,
                                descriptors=morphologies,
                                disease_name=disease,
                            )
                        except Exception as error:  # noqa: BLE001
                            raise RuntimeError(
                                f"Feature extraction failed at row={idx}, disease='{disease}', image='{image_name}'"
                            ) from error
                        visual_cache[cache_key] = visual_features
                    api_bar.update(1)

                    if visual_features.get("extraction_status") != "success":
                        backup.append(
                            backup.gap_audit,
                            {
                                "gapType": "visual_feature_extraction",
                                "row": idx,
                                "disease": disease,
                                "descriptors": morphologies,
                                "status": visual_features.get("extraction_status", "unknown"),
                                "errorCode": visual_features.get("extraction_error_code", "unknown"),
                                "errorDetail": visual_features.get("extraction_error_detail", ""),
                            },
                        )

                # Step 2: only after API success, persist unique graph rows into Neo4j/backups.
                    graph_key = (
                        disease.lower(),
                        main_class.lower(),
                        sub_class.lower(),
                        tuple(sorted(part.lower() for part in morphologies)),
                        tuple(sorted(part.lower() for part in body_regions)),
                    )
                    if graph_key not in graph_row_seen:
                        graph_row_seen.add(graph_key)

                        store.upsert_disease_hierarchy(
                            {
                                "main_class": main_class,
                                "sub_class": sub_class,
                                "disease": disease,
                                "morphologies": morphologies,
                                "body_regions": body_regions,
                            }
                        )
                        store.upsert_visual_feature(
                            {
                                "disease": disease,
                                "feature_id": visual_features["feature_id"],
                                "feature_natural_key": visual_features["feature_natural_key"],
                                "mu_ref": visual_features["mu_ref"],
                                "kappa": visual_features["kappa"],
                                "descriptor_signature": visual_features["descriptor_signature"],
                                "descriptor_count": visual_features["descriptor_count"],
                                "condition_name": visual_features["condition_name"],
                                "condition_key": visual_features["condition_key"],
                                "morphology_summary": visual_features["morphology_summary"],
                                "extracted_by": visual_features["extracted_by"],
                                "extraction_status": visual_features["extraction_status"],
                                "extraction_error_code": visual_features["extraction_error_code"],
                            }
                        )
                        store.upsert_visual_atoms(
                            {
                                "feature_natural_key": visual_features["feature_natural_key"],
                                "atoms": visual_features["descriptor_tokens"],
                            }
                        )
                        feature_upserts += 1

                        backup.append(
                        backup.kg_snapshot,
                        {
                            "nodeType": "Disease",
                            "nodeId": f"disease:{disease.lower().replace(' ', '_')}",
                            "properties": {
                                "disease": disease,
                                "mainClass": main_class,
                                "subClass": sub_class,
                                "morphologies": morphologies,
                                "bodyRegions": body_regions,
                            },
                            "schemaVersion": "1.0.0",
                        },
                    )

                        hierarchy_upserts += 1

                        backup.append(
                        backup.kg_snapshot,
                        {
                            "nodeType": "VisualFeaturePrototype",
                            "nodeId": visual_features["feature_id"],
                            "properties": {
                                "disease": disease,
                                "muRef": visual_features["mu_ref"],
                                "kappa": visual_features["kappa"],
                                "descriptorSignature": visual_features["descriptor_signature"],
                                "descriptorCount": visual_features["descriptor_count"],
                                "conditionName": visual_features["condition_name"],
                                "conditionKey": visual_features["condition_key"],
                                "morphologySummary": visual_features["morphology_summary"],
                                "extractedBy": visual_features["extracted_by"],
                                "extractionStatus": visual_features["extraction_status"],
                                "extractionErrorCode": visual_features["extraction_error_code"],
                            },
                            "schemaVersion": "1.0.0",
                        },
                    )
                        for atom in visual_features["descriptor_tokens"]:
                            backup.append(
                                backup.kg_snapshot,
                                {
                                    "nodeType": "VisualAtom",
                                    "nodeId": f"visual_atom:{atom}",
                                    "properties": {
                                        "featureNaturalKey": visual_features["feature_natural_key"],
                                        "atom": atom,
                                    },
                                    "schemaVersion": "1.0.0",
                                },
                            )

                    progress_file.write_text(json.dumps({"last_completed_row": idx}), encoding="utf-8")
                    neo4j_bar.update(1)

                if batch_pause > 0:
                    time.sleep(batch_pause)
        finally:
            api_bar.close()
            neo4j_bar.close()

        sorted_diseases = sorted(diseases)
        print(f"Unique diseases discovered: {len(sorted_diseases)}")

        for disease_batch_no, disease_batch in enumerate(chunked(sorted_diseases, batch_size), start=1):
            print(f"Processing disease batch {disease_batch_no}: size={len(disease_batch)}")
            for disease in tqdm(disease_batch, desc="Enriching evidence", unit="disease"):
                try:
                    evidence_rows = evidence_service.collect_for_disease(disease)
                except Exception as error:  # noqa: BLE001
                    evidence_rows = []
                    backup.append(
                        backup.gap_audit,
                        {
                            "gapType": "evidence_collection",
                            "disease": disease,
                            "status": "failed",
                            "errorCode": "external_api_error",
                            "errorDetail": str(error),
                        },
                    )
                for evidence in evidence_rows:
                    item = {
                        "disease": disease,
                        "source": evidence.get("source", "Unknown"),
                        "evidence_id": evidence.get("evidence_id", ""),
                        "title": evidence.get("title", ""),
                        "journal": evidence.get("journal", ""),
                        "pubdate": evidence.get("pubdate", ""),
                        "doi": evidence.get("doi", ""),
                    }
                    store.upsert_evidence(item)
                    backup.append(backup.evidence_cards, item)
                    evidence_upserts += 1

                wiki_summary = literature_client.fetch_wikipedia_summary(disease)
                if not wiki_summary:
                    backup.append(
                        backup.gap_audit,
                        {
                            "gapType": "wikipedia_summary",
                            "disease": disease,
                            "status": "empty",
                            "errorCode": "no_content",
                            "errorDetail": "Wikipedia summary returned empty string",
                        },
                    )
                context_parts = [wiki_summary]
                for evidence in evidence_rows[:5]:
                    title = str(evidence.get("title", "")).strip()
                    if title:
                        context_parts.append(title)
                context_text = "\n".join(part for part in context_parts if part)

                generated = symptom_effect_service.generate(
                    disease_name=disease,
                    context_text=context_text,
                    provider_mode="bytez_only",
                )
                generation_mode = str(generated.get("generation_mode", "unknown"))
                symptoms = [normalize_label(token).lower() for token in generated.get("symptoms", []) if normalize_label(token)]
                effects = [normalize_label(token).lower() for token in generated.get("effects", []) if normalize_label(token)]

                if not symptoms and not effects:
                    # Ensure we still have minimal atomic nodes for downstream diagnosis graph.
                    fallback_generated = symptom_effect_service.generate(
                        disease_name=disease,
                        context_text="",
                        provider_mode="bytez_only",
                    )
                    fallback_mode = str(fallback_generated.get("generation_mode", "unknown"))
                    symptoms = [
                        normalize_label(token).lower()
                        for token in fallback_generated.get("symptoms", [])
                        if normalize_label(token)
                    ]
                    effects = [
                        normalize_label(token).lower()
                        for token in fallback_generated.get("effects", [])
                        if normalize_label(token)
                    ]

                    # Prefer provider-backed generation mode for provenance when fallback call returns data.
                    if fallback_mode.startswith("hf") or fallback_mode.startswith("bytez"):
                        generation_mode = fallback_mode

                model_generated = generation_mode.startswith("hf") or generation_mode.startswith("bytez")

                if not model_generated:
                    backup.append(
                        backup.gap_audit,
                        {
                            "gapType": "symptom_effect_generation",
                            "disease": disease,
                            "status": "skipped_non_model_output",
                            "errorCode": "model_generation_required",
                            "errorDetail": f"generation_mode={generation_mode}",
                        },
                    )
                    continue

                if not generation_mode.startswith("bytez"):
                    backup.append(
                        backup.gap_audit,
                        {
                            "gapType": "symptom_effect_generation",
                            "disease": disease,
                            "status": "degraded",
                            "errorCode": "bytez_or_context_unavailable",
                            "errorDetail": f"generation_mode={generation_mode}",
                        },
                    )
                store.upsert_symptoms_effects(
                    {
                        "disease": disease,
                        "symptoms": symptoms,
                        "effects": effects,
                    }
                )
                symptom_effect_upserts += len(symptoms) + len(effects)

                source_label = (
                    "wikipedia_plus_bytez"
                    if generation_mode.startswith("bytez")
                    else "wikipedia_plus_fallback"
                )

                for symptom in symptoms:
                    backup.append(
                        backup.kg_snapshot,
                        {
                            "nodeType": "Symptom",
                            "nodeId": f"symptom:{symptom}",
                            "properties": {
                                "disease": disease,
                                "name": symptom,
                                "source": source_label,
                            },
                            "schemaVersion": "1.0.0",
                        },
                    )
                for effect in effects:
                    backup.append(
                        backup.kg_snapshot,
                        {
                            "nodeType": "Effect",
                            "nodeId": f"effect:{effect}",
                            "properties": {
                                "disease": disease,
                                "name": effect,
                                "source": source_label,
                            },
                            "schemaVersion": "1.0.0",
                        },
                    )

            if batch_pause > 0:
                time.sleep(batch_pause)

        print("Knowledge graph build complete.")
        print(f"Hierarchy rows upserted: {hierarchy_upserts}")
        print(f"Visual feature rows upserted: {feature_upserts}")
        print(f"Diseases processed for evidence: {len(sorted_diseases)}")
        print(f"Evidence rows upserted: {evidence_upserts}")
        print(f"Symptom/effect tokens upserted: {symptom_effect_upserts}")
        progress_file.write_text(json.dumps({"last_completed_row": start_index + len(raw_rows)}), encoding="utf-8")
    finally:
        store.close()


if __name__ == "__main__":
    main()
