# HYPERDERM-Graph: Top-Tier Research Plan (Hierarchy + KG-RAG + Privacy-First)

## Research Ambition
Design a publishable AI-dermatology system with clear novelty, reproducibility, and clinical relevance. The target is top-tier standards (Nature-family/Lancet Digital Health/JAMA-level expectations), while recognizing publication cannot be guaranteed.

## Why This Can Be Novel
Current high-impact dermatology AI literature repeatedly reports gaps in:
- Limited external and cross-site generalization.
- Under-representation of darker skin tones and poor fairness reporting.
- Weak uncertainty calibration and unsafe overconfidence.
- Limited causal/clinically grounded explainability.
- Poor integration of structured medical knowledge with multimodal vision-language reasoning.

HYPERDERM-Graph addresses these jointly in one framework.

## Proposed New Algorithm: HYPERDERM-Graph
Full name: `Hyperspherical Causal Hierarchical Evidence Routing for Dermatology`

### Core Idea
A strict hierarchical diagnosis engine that combines:
1. CHYVA-style hyperspherical visual representation (`mu`, `kappa`).
2. Causal gating to suppress shortcut paths (especially skin-tone-only shortcuts).
3. Neo4j graph retrieval for mechanistic and differential evidence.
4. Local fallback KB retrieval for robustness.
5. Bytez MedGemma reasoning with mandatory evidence citation.
6. Confidence-aware abstention and escalation policy.

### Method Components
1. Hyperspherical Feature Projection
- Extract image features with MedGemma-compatible visual encoder.
- Project embedding `z` to unit sphere and estimate vMF parameters:
  - pathology direction `mu`
  - concentration `kappa`
- Keep only derived features. No raw image in graph.

2. Causal Bias Suppression (CHYVA-CI)
- Build causal graph where morphology and lesion structure are valid parents of diagnosis.
- Treat sensitive shortcuts as blocked paths in inference scoring.
- Apply intervention-style penalty when prediction is overly sensitive to shortcut proxies.

3. Hierarchical Evidence Router
- Stage A: `MainClass` scoring.
- Stage B: `SubClass` routing inside selected main class.
- Stage C: disease ranking.
- Stage D: differential elimination by rule nodes and counter-evidence.

4. Graph-RAG + Local-RAG Fusion
- Retrieve evidence from Neo4j + local corpus.
- Score each evidence item on relevance, source quality, and consistency.
- Use consensus-aware evidence aggregation to prevent single-source hallucination.

5. Explainable Decision Compiler
- Generate structured explanation including:
  - accepted evidence,
  - rejected alternatives,
  - causal safety checks,
  - uncertainty and abstain triggers.

## Strict Data Policy (DPDP Aligned)
- No patient ID in graph.
- No person-level nodes.
- No raw image in graph.
- Only generalized medical knowledge + derived feature prototypes.
- Temporary processing artifacts auto-purged.
- Write-time privacy policy validator rejects forbidden fields.

## Neo4j Graph Design (Generalized Only)
### Node Labels
- `MainClass`
- `SubClass`
- `Disease`
- `Morphology`
- `BodyRegion`
- `VisualFeaturePrototype`
- `Descriptor`
- `DifferentialRule`
- `Evidence`
- `Source`
- `Guideline`
- `Treatment`
- `RedFlag`
- `FollowUp`
- `PopulationContext`
- `InferenceRun` (non-personal audit trace)

### Relationships
- `(:MainClass)-[:HAS_SUB_CLASS]->(:SubClass)`
- `(:SubClass)-[:HAS_DISEASE]->(:Disease)`
- `(:Disease)-[:PRESENTS_WITH]->(:Morphology)`
- `(:Disease)-[:COMMON_AT]->(:BodyRegion)`
- `(:Disease)-[:ASSOCIATED_WITH_FEATURE]->(:VisualFeaturePrototype)`
- `(:Disease)-[:HAS_DESCRIPTOR]->(:Descriptor)`
- `(:Disease)-[:HAS_DIFFERENTIAL_RULE]->(:DifferentialRule)`
- `(:Disease)-[:SUPPORTED_BY]->(:Evidence)`
- `(:Evidence)-[:FROM_SOURCE]->(:Source)`
- `(:Disease)-[:HAS_GUIDELINE]->(:Guideline)`
- `(:Disease)-[:HAS_TREATMENT]->(:Treatment)`
- `(:Disease)-[:HAS_RED_FLAG]->(:RedFlag)`
- `(:Disease)-[:HAS_FOLLOWUP]->(:FollowUp)`
- `(:Disease)-[:HAS_POPULATION_CONTEXT]->(:PopulationContext)`
- `(:InferenceRun)-[:USED_FEATURE]->(:VisualFeaturePrototype)`
- `(:InferenceRun)-[:CANDIDATE {score,rank}]->(:Disease)`
- `(:InferenceRun)-[:FINAL {confidence,abstained}]->(:Disease)`

## Mandatory Local Backup (JSON)
Maintain an immutable local backup so Neo4j is never a single point of failure.

### Backup Files
- `backup/kg_snapshot.jsonl`
- `backup/evidence_cards.jsonl`
- `backup/inference_audit.jsonl`
- `backup/schema_version.json`

### Backup Rules
- On every graph upsert, append equivalent JSONL record locally.
- Daily snapshot export of full ontology and relationship edges.
- Version every backup with `schemaVersion`, `pipelineVersion`, `sourceHash`, `timestamp`.
- Support replay mode: rebuild Neo4j fully from JSON backup.

### Minimal JSON Record Schemas
`kg_snapshot.jsonl` record:
```json
{
  "nodeType": "Disease",
  "nodeId": "disease:psoriasis",
  "properties": {"name": "Psoriasis", "icd": "L40"},
  "edges": [
    {"type": "HAS_SUB_CLASS", "to": "subclass:papulosquamous"}
  ],
  "provenance": {"source": "DermNetNZ", "retrievedAt": "2026-03-24T12:00:00Z"},
  "schemaVersion": "1.0.0"
}
```

`inference_audit.jsonl` record:
```json
{
  "runId": "uuid",
  "model": "unsloth/medgemma-27b-it",
  "features": {"muRef": "vec:abc", "kappa": 17.2, "morphology": ["plaque", "scale"]},
  "hierarchyPath": ["Inflammatory", "Papulosquamous", "Psoriasis"],
  "candidates": [{"disease": "Psoriasis", "score": 0.82}],
  "abstained": false,
  "evidenceIds": ["ev:123", "ev:456"],
  "timestamp": "2026-03-24T12:01:00Z"
}
```

## Bytez MedGemma Integration
Use:
```javascript
import Bytez from "bytez.js"
const sdk = new Bytez(process.env.BYTEZ_API_KEY)
const model = sdk.model("unsloth/medgemma-27b-it")
```

Inference prompt contract:
- Input: derived visual features + retrieved evidence chunks.
- Output: strict JSON with hierarchy levels, differentials, citations, and uncertainty.
- Hard rules: no diagnosis without cited evidence IDs; abstain if evidence weak/conflicting.

## Research Agent (Indian Skin Focus)
Purpose:
- Continuously mine evidence for Indian skin presentations and look-alike differentials.
- Update graph with population-aware morphology and descriptor nuances.

Inputs:
- DermNet NZ, AAD/Atlas, open guideline papers.
- VisualDx/UpToDate/DynaMed only when licensed access/API is available.

Outputs:
- Evidence cards, contradiction matrix, and KG triples with provenance.
- Fairness watchlist where diseases show elevated error disparity.

## Evaluation Protocol for Top-Tier Standards
1. Hierarchy-level accuracy
- Accuracy at MainClass, SubClass, Disease levels.

2. Calibration and Safety
- ECE, Brier score, abstention utility, red-flag sensitivity.

3. Fairness
- Stratified performance across skin-tone groups and sex.
- Worst-group accuracy and equalized odds gap.

4. Generalization
- External site split, temporal split, and device-shift split.

5. Explainability Quality
- Evidence citation precision.
- Clinician-rated rationale usefulness.
- Differential rejection correctness.

6. Privacy Compliance
- Zero forbidden-field writes.
- Backup integrity and replay success rate.

## Manuscript-Worthy Contributions
Planned contribution claims:
- New algorithm: HYPERDERM-Graph with CHYVA-CI causal bias suppression.
- First integrated hierarchical dermatology KG-RAG with privacy-minimal feature-only graph design.
- Robust fallback with deterministic JSON replayable backups.
- Population-context-aware explainability for Indian skin-focused differential diagnosis.

## Revised Build Roadmap
Phase 1: Foundations
- Finalize ontology and Cypher schema.
- Implement DPDP write validator.
- Implement local JSON backup writer and replay loader.

Phase 2: Algorithm Core
- Implement hyperspherical feature module (`mu`, `kappa`).
- Implement causal penalty and hierarchical router.
- Implement graph + fallback retrievers.

Phase 3: Model and Explainability
- Integrate Bytez MedGemma with strict schema output.
- Add differential elimination and abstain policy.
- Add evidence-citation enforcement.

Phase 4: Research and Validation
- Run subgroup fairness and external robustness studies.
- Run ablation studies for each novelty component.
- Freeze reproducible experiments and paper artifacts.

## Immediate Next Execution Tasks
1. Implement Neo4j schema files and constraints for generalized nodes/edges.
2. Implement JSON backup module and replay CLI.
3. Implement CHYVA feature extraction interface with no-image persistence policy.
4. Implement hierarchical inference contract and Bytez prompt template.
5. Implement evaluation harness for fairness, calibration, and external shift.
