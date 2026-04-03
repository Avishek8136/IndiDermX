# Project Summary: HYPERDERM-Graph

## 1. Vision
Build a top-tier research-grade AI dermatology system for strict hierarchical diagnosis that is:
- Explainable
- Privacy-first (DPDP aligned)
- Bias-aware for Indian skin presentations
- Robust via Neo4j graph RAG + local fallback knowledge base

The system targets publication-quality contribution quality in AI + dermatology.

## 2. Core Problem Being Solved
Current dermatology AI systems often face:
- Weak external generalization
- Skin-tone fairness gaps
- Limited uncertainty calibration
- Poor clinical explainability
- Weak integration of structured clinical knowledge

This project addresses these by combining causal visual features, hierarchical reasoning, and evidence-grounded graph retrieval.

## 3. Proposed Novel Algorithm
Name: **HYPERDERM-Graph**  
Full: **Hyperspherical Causal Hierarchical Evidence Routing for Dermatology**

Main algorithmic ideas:
1. CHYVA-style hyperspherical projection of visual features (`mu`, `kappa`).
2. Causal bias suppression to reduce shortcut reliance.
3. Strict hierarchical diagnosis path:
   - Main class -> Subclass -> Disease -> Differential elimination -> Final output.
4. Graph-RAG + Local-RAG fusion for robust evidence retrieval.
5. Evidence-cited reasoning with abstain/escalate behavior under uncertainty.

## 4. Privacy and Data Policy (Non-Negotiable)
- No raw image storage in Neo4j.
- No patient identifiers in graph (`Subject_ID`, names, contacts, etc.).
- No person-level profile nodes.
- Only generalized medical ontology + derived visual feature abstractions.
- Temporary processing artifacts are short-lived and purged.
- Write-time privacy validator blocks forbidden fields before graph update.

## 5. Knowledge Graph Design (Generalized)
Primary store: Neo4j Aura.

Core node types:
- `MainClass`, `SubClass`, `Disease`
- `Morphology`, `BodyRegion`, `Descriptor`
- `VisualFeaturePrototype`
- `DifferentialRule`
- `Evidence`, `Source`, `Guideline`
- `Treatment`, `RedFlag`, `FollowUp`, `PopulationContext`
- `InferenceRun` (non-personal trace)

Core relationships:
- Hierarchy: `MainClass -> SubClass -> Disease`
- Disease evidence and rules: `Disease -> Evidence/Rule/Guideline/Treatment`
- Visual grounding: `Disease -> VisualFeaturePrototype`
- Population knowledge: `Disease -> PopulationContext`

## 6. RAG and Reasoning Pipeline
1. Extract derived visual features from image.
2. Retrieve hierarchical candidates and evidence from Neo4j.
3. Retrieve backup evidence from local knowledge base.
4. Fuse and rank evidence by relevance + source quality + consistency.
5. Use Bytez MedGemma (`unsloth/medgemma-27b-it`) for structured, cited reasoning.
6. Return strict hierarchical output with confidence and differential rejection logic.

## 7. Explainability Contract
Every prediction must return:
- `hierarchyPath`
- `candidateList`
- `supportingEvidence`
- `counterEvidence`
- `whyNotTopAlternatives`
- `biasChecks`
- `privacySafeFieldsUsed`
- `finalRecommendation`

## 8. Local Backup Requirement (Mandatory)
Neo4j is not the single source of truth.

Required local files:
- `backup/kg_snapshot.jsonl`
- `backup/evidence_cards.jsonl`
- `backup/inference_audit.jsonl`
- `backup/schema_version.json`

Rules:
- Every graph write has equivalent JSONL backup append.
- Version all records with schema and pipeline version.
- Support replay mode to rebuild graph from JSON backups.

## 9. APIs Needed
Minimum required:
1. Neo4j Aura API/driver
2. Bytez API (`BYTEZ_API_KEY`)

Recommended open research APIs:
- PubMed E-utilities
- Europe PMC
- Crossref
- OpenAlex (or Semantic Scholar)

Optional premium clinical sources (license dependent):
- VisualDx
- UpToDate
- DynaMed

## 10. Research Agent Scope
Build a research agent focused on:
- Indian skin morphology and differential nuances
- Evidence extraction from trusted clinical sources
- Contradiction detection across references
- KG-ready triples with provenance metadata

## 11. Evaluation Plan (Publication-Oriented)
- Hierarchical accuracy at each level
- Top-1 and Top-3 disease accuracy
- Calibration (ECE, Brier)
- Differential elimination quality
- Fairness gaps across skin-tone strata
- External robustness (site/device/time shift)
- Explainability quality (citation precision, clinician utility)
- Privacy compliance pass rate

## 12. Current Status
Completed:
- Project direction finalized around privacy-minimal, feature-only graph design
- Novel algorithm concept (HYPERDERM-Graph) defined
- API strategy and acquisition roadmap outlined
- Backup architecture requirement specified

Next implementation priorities:
1. Neo4j schema + constraints for generalized ontology
2. JSON backup writer + replay loader
3. CHYVA feature interface (no image persistence)
4. Bytez prompt + strict JSON output schema
5. End-to-end hierarchical inference endpoint
6. Evaluation harness for fairness, calibration, and robustness

## 13. Practical Note
This system is for clinical decision support and research. It should include abstain/escalation pathways and should not be positioned as autonomous medical diagnosis without clinical oversight.
