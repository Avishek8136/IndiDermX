# HyperDerm: Agentic Dermatology Diagnostic System

A conversational AI system for dermatology diagnosis using Neo4j graph database, MedGemma LLM, and a professional chat interface.

## Overview

HyperDerm combines:
- **Neo4j Graph Database**: Knowledge graph of diseases, symptoms, body regions, and clinical evidence
- **MedGemma LLM**: Medical-grade language model for feature extraction and diagnosis generation
- **LangGraph Workflow**: Multi-agent reasoning workflow with supervision, planning, and safety guards
- **Modern Web UI**: Professional chat interface with real-time status, markdown rendering, and image upload

**Problem it solves**: When dermatologists lack time or patients lack access, HyperDerm provides doctor-style conversational diagnosis support with explainability and evidence trails.

## Features

### ✅ Core Functionality
- **Conversational Intake**: Natural Q&A flow that gathers clinical details progressively
- **Graph-Augmented Diagnosis**: Searches Neo4j knowledge graph for matching diseases
- **MedGemma Direct Inference**: When graph has no matches, diagnoses using model-only differential
- **Explainability**: Shows which graph features matched, evidence sources, and reasoning
- **Multi-Round Reasoning**: Improves diagnosis as patient provides more symptoms
- **Medical Safety Guards**: Adds disclaimers and recommends specialist consultation

### 🎨 Frontend
- Sidebar with brand, status banner, image upload, session management
- Chat log with markdown-formatted assistant responses
- Clickable suggested questions that populate the message box
- Real-time backend health polling
- Responsive layout (desktop and mobile)
- Copy-friendly explainability metadata display

### 🔧 Backend Services
- **FastAPI REST endpoints** for chat, diagnosis, image upload
- **MCP (Model Context Protocol) server** for tool integration
- **Workflow orchestration** via LangGraph with conditional routing
- **Neo4j integration** with automatic fallback when unavailable
- **Local JSONL backup** for offline resilience

## Research Components

> Highlighted research track for the HYPERDERM-Graph roadmap.

## Agentic Workflow

HyperDerm is designed as an agentic system, not a single-pass chatbot.

### 🤖 Agentic Orchestration
- A supervisor-style workflow routes each request through specialized steps.
- Different nodes handle extraction, diagnosis, evidence retrieval, confidence assessment, and safety checks.
- The system can ask follow-up questions, abstain when uncertainty is high, and retry with fallback reasoning.
- Tool use is explicit: graph lookup, backup retrieval, and model inference are coordinated as part of the response path.

### Why It Matters
- Produces more clinical-style reasoning than a one-shot prompt.
- Makes the decision path inspectable and easier to debug.
- Supports safer behavior by separating intake, confidence gating, and final response generation.

### 🔬 CHYVA Hyperspherical Latent Projection
- Projects derived visual features into a hyperspherical latent space.
- Uses `mu` for direction and `kappa` for concentration.
- Keeps only privacy-safe, derived representations rather than raw image data.

### 🧠 CHYVA-CI Causal Bias Suppression
- Reduces shortcut reliance in diagnosis scoring.
- Penalizes spurious correlations that can skew predictions across skin tones or presentation styles.
- Supports more robust, clinically grounded reasoning.

### 🧩 Other Research Components
- **Strict Hierarchical Diagnosis**: main class -> subclass -> disease -> differential elimination.
- **Graph-RAG + Local-RAG Fusion**: combines Neo4j evidence with local fallback knowledge.
- **Evidence-Cited Reasoning**: every conclusion should explain what evidence supported or rejected it.
- **Privacy-Minimal Graph Design**: no raw images, no patient identifiers, no person-level nodes.
- **Population-Aware Validation**: fairness, calibration, and external robustness checks for Indian skin presentations.
- **Deterministic Replay Backups**: JSONL snapshots for reproducible graph rebuilds and audit trails.

These components are part of the research-oriented HYPERDERM-Graph direction and can be used to guide future development, evaluation, and publication work.

## Architecture

```
HyperDerm/
├── hyperderm/
│   ├── api/                          # FastAPI application
│   │   ├── app.py                    # FastAPI factory
│   │   ├── routes/                   # Endpoints (chat, health, diagnosis)
│   │   └── dependencies.py           # Dependency injection
│   ├── workflows/
│   │   └── chatbot_graph.py          # LangGraph multi-agent workflow
│   ├── services/
│   │   ├── medgemma_chat_service.py  # LLM calls & answer generation
│   │   ├── diagnosis_service.py      # Neo4j integration
│   │   ├── graph_rag_service.py      # Context retrieval
│   │   ├── local_backup_*.py         # Fallback diagnosis
│   │   └── privacy.py                # Data sanitization
│   ├── infrastructure/
│   │   ├── graph/
│   │   │   └── neo4j_store.py        # Neo4j queries
│   │   ├── backup/
│   │   │   └── jsonl_store.py        # JSONL persistence
│   │   └── clients/
│   │       └── literature.py         # Evidence retrieval
│   ├── domain/
│   │   └── schemas.py                # Data types
│   ├── core/
│   │   └── config.py                 # Settings from .env
│   └── scripts/
│       ├── run_stack.py              # Start all services
│       ├── build_kg.py               # Build Neo4j graph
│       └── run_infer.py              # CLI inference
│
├── frontend_service/
│   ├── app.py                        # Flask for static serving
│   └── static/
│       └── index.html                # Modern chat UI
│
├── tests/
│   └── test_multi_agent_chatbot_graph.py  # Workflow regression tests
│
├── backup/
│   ├── kg_snapshot.jsonl             # Local disease catalog
│   └── bytez_calls.jsonl             # API call log
│
└── .env                              # Configuration
```

### Workflow: Multi-Agent LangGraph

```
┌─────────────────────────────────────────────────────────────┐
│ Chat Message + Session Context                              │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │ extract_query_features  │ (MedGemma: parse intent)
        └────────────┬────────────┘
                     │
        ┌────────────▼─────────────┐
        │ diagnose_neo4j           │ (Neo4j: find candidates)
        └────────────┬─────────────┘
                     │
      ┌──────────────┴──────────────┐
      │ (Conditional routing)       │
      ├─────────────────────────────┤
      │ If candidates exist:        │
      │   → select_primary          │
      │                             │
      │ If no candidates (strict):  │
      │   → allow MedGemma direct   │
      │                             │
      │ Otherwise:                  │
      │   → use local backup        │
      └──────────────┬──────────────┘
                     │
        ┌────────────▼────────────┐
        │ retrieve_evidence       │ (GraphRAG for all candidates)
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │ generate_questions      │ (Ask clinician for more details)
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │ assess_confidence       │ (Decide: diagnose or abstain?)
        └────────────┬────────────┘
                     │
      ┌──────────────┴──────────────┐
      │ should_abstain?             │
      ├─────────────────────────────┤
      │ YES: abstain_answer         │
      │ (Ask for more symptoms)     │
      │                             │
      │ NO: generate_answer         │
      │ (MedGemma: 4-section output)│
      └──────────────┬──────────────┘
                     │
        ┌────────────▼────────────┐
        │ medical_safety_guard    │ (Add disclaimers)
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │ finalize                │ (Package response)
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │ JSON Response to Frontend│
        └─────────────────────────┘
```

## Prerequisites

- **Python 3.13+**
- **Neo4j Aura** (cloud) or **Neo4j Community** (local)
- **Bytez API Key** (for MedGemma LLM access)
- **Node.js** (optional, for frontend development)

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/Avishek8136/IndiDermX.git
cd IndiDermX
```

### 2. Set Up Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env with your credentials:
# - NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
# - BYTEZ_API_KEY
# - STRICT_NEO4J_ONLY=true (recommended)
```

### 4. Build Knowledge Graph (Optional)
```bash
python -m hyperderm.scripts.build_kg
```

This creates the Neo4j graph from CSV disease data. If skipped, local JSONL backup is used.

## Running

### Start All Services
```bash
python -m hyperderm.scripts.run_stack
```

This starts:
- **Backend API** on http://127.0.0.1:8000 (FastAPI)
- **MCP Server** on http://127.0.0.1:9001 (protocol server)
- **Frontend** on http://127.0.0.1:8080 (chat UI)

### Access the Application
Open browser: **http://localhost:8080**

## Configuration

### Key Settings (.env)

| Variable | Description | Example |
|----------|-------------|---------|
| `NEO4J_URI` | Graph database connection | `neo4j+s://xxxx.databases.neo4j.io` |
| `NEO4J_USERNAME` | Database username | `neo4j` |
| `NEO4J_PASSWORD` | Database password | `your_password_here` |
| `BYTEZ_API_KEY` | LLM service API key | `your_bytez_key_here` |
| `BYTEZ_MODEL` | Model identifier | `ASAIs-TDDI-2025/MedTurk-MedGemma-4b` |
| `STRICT_NEO4J_ONLY` | Disable local backup fallback | `true` |
| `BACKUP_DIR` | Local JSONL backup location | `backup` |
| `BYTEZ_TIMEOUT_SECONDS` | API call timeout | `45` |

## API Endpoints

### Chat Endpoint
```http
POST /chat/mcp
Content-Type: application/json

{
  "message": "Red, itchy patches on my arm",
  "session_id": "sess-abc123",
  "image_path": "/uploads/skin.jpg"
}
```

**Response**:
```json
{
  "session_id": "sess-abc123",
  "answer": "Probable condition: Tinea corporis or contact dermatitis...",
  "top_candidate": {
    "disease": "Tinea Corporis",
    "score": 4.2,
    "matched_descriptors": ["scaly", "patchy"]
  },
  "suggested_questions": [
    "Has it spread?",
    "Do you have itching?"
  ],
  "explainability": {
    "reasoning": {
      "confidence": {
        "reason": "sufficient_signal",
        "round": 2
      }
    },
    "neo4j_candidate_matches": [...],
    "graph_context": [...]
  },
  "used_fallback": false
}
```

### Health Check
```http
GET /health

200 OK: {"status": "ok"}
```

### Image Upload
```http
POST /upload-image
Content-Type: multipart/form-data

{
  "file": <binary image data>
}
```

**Response**:
```json
{
  "filename": "IMG_20240101_120000.jpg",
  "image_path": "/uploads/abc123.jpg"
}
```

## Usage Examples

### Example 1: Minimal Input → Engagement
```
User Round 1: "Hi"
Assistant: "Hello! I'm your dermatology assistant. To help me understand your skin concern better, I have a few clinical questions..."
Questions:
- Which body area is most affected?
- How does the lesion look?
- What symptoms do you feel?
```

### Example 2: With Symptoms → Diagnosis
```
User Round 1: "Red patches on my arm, very itchy"
User Round 2: "It's getting worse, some areas are cracked"

Assistant (Round 2):
Probable condition: Tinea Corporis (fungal infection) or severe Contact Dermatitis

Why this matches:
- Red, itchy patches: common in both fungal and allergic reactions
- Worsening with cracking: suggests infection or inflammation progression
- Arm location: common site for fungal infections spreading

Graph evidence used:
- Tinea Corporis (score 4.2): matched on descriptors [scaly, patchy], symptoms [itch], effects [cracking]
- Contact Dermatitis (score 3.5): matched on symptoms [itch], effects [inflammation]

Clinical caution:
These are potential diagnoses only. Please see a dermatologist for KOH test or culture to confirm fungal infection. This is decision support only and not a confirmed medical diagnosis.
```

### Example 3: No Graph Match → Direct MedGemma Inference
```
User Round 1: "Strange lesion, not sure what it is"
User Round 2: "It's on my face, purple color, doesn't itch"

Assistant (Round 2):
Probable condition: Possible vascular lesion, hemangioma, or birthmark (based on purple color and face location)

Why this matches:
- Purple coloration and flat presentation suggest vascular origin
- Non-pruritic nature rules out inflammatory causes
- Facial location common for benign vascular lesions

Graph evidence used:
None - direct model assessment (Neo4j had no strong matching candidates)

Clinical caution:
Any new or changing lesion on the face warrants dermatologist evaluation for accurate diagnosis and possible biopsy. This is decision support only and not a confirmed medical diagnosis.
```

## Testing

### Run All Tests
```bash
pytest tests/test_multi_agent_chatbot_graph.py -v
```

### Key Test Scenarios
- Graph path with valid candidates
- Fallback when Neo4j empty
- Abstain on low confidence
- Force diagnosis after round limit
- Strict mode (MedGemma direct inference)
- Multi-candidate GraphRAG loop

### Test Results
All 9 regression tests passing:
- ✅ Neo4j path with scoring
- ✅ Fallback to local backup
- ✅ Abstain logic on zero/low scores
- ✅ Force diagnosis timeout
- ✅ Neo4j error resilience
- ✅ Strict mode direct inference
- ✅ GraphRAG multiloop
- ✅ Round 1 intake engagement

## Troubleshooting

### Issue: "Rate limited, free account users are limited to 1 request at a time"
**Cause**: Bytez API tier limitation  
**Solution**: 
- Upgrade Bytez account to paid tier
- Increase timeout between requests
- Use alternative LLM provider (see below)

### Issue: "Failed to resolve 'api.bytez.com'"
**Cause**: Network connectivity issue  
**Solution**:
- Check internet connection
- Verify firewall rules
- Check Bytez API status

### Issue: Neo4j connection timeout
**Cause**: Database unreachable  
**Solution**:
- Verify NEO4J_URI and credentials in .env
- Check Neo4j Aura console for connection IP whitelist
- Ensure network allows outbound HTTPS on port 7687

### Issue: Backend health check fails
**Cause**: Services not fully started  
**Solution**:
```bash
# Wait 5-10 seconds for startup
sleep 10
curl http://127.0.0.1:8000/health
```

### Issue: Frontend shows "No strong match" for valid symptoms
**Cause**: Graph candidates have low scores  
**Solution**:
- Add more descriptors/symptoms in message
- Check that graph is initialized with diseases
- Review Neo4j scoring logic in `neo4j_store.py`

## Logs & Debugging

### Enable Debug Logging
Debug logs are enabled by default and show:
- MedGemma API call attempts
- Feature extraction results
- Graph query results
- Fallback activations
- Reasoning decisions

### View API Call History
```bash
tail -50 backup/bytez_calls.jsonl
```

Shows all LLM API calls with timestamps, status, and errors.

## Extending the System

### Add New Diseases to Graph
Edit `scripts/build_kg.py` or upload CSV to Neo4j:
```cypher
CREATE (d:Disease {
  name: "Your Disease",
  mainClass: "Category",
  subClass: "Subcategory"
})
```

### Customize MedGemma Prompts
Edit `services/medgemma_chat_service.py`:
- `generate_chat_answer()`: Main diagnosis prompt
- `extract_query_features()`: Feature extraction prompt
- `_direct_medgemma_diagnosis()`: Fallback prompt

### Add Custom Evidence Sources
Implement in `services/local_rag_service.py`:
- Integrate medical literature APIs
- Add image-based CV matching
- Connect to clinical databases

## Performance Considerations

### Neo4j Query Optimization
- Indexed on disease names and descriptors
- Limit candidate results to top 5
- Cache GraphRAG district results

### Token Efficiency
- Limit candidates passed to LLM (3-5 top matches)
- Summarize graph context to <1000 tokens
- Use simplified prompts for fallback

### Rate Limiting Handling
- Bytez API: 1 request/second on free tier
- Queue requests or upgrade tier
- Local LLM alternative: Use Ollama + MedLLaMA

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **LangGraph** | Structured multi-agent reasoning with state management |
| **Neo4j** | Graph-native queries for disease-symptom relationships |
| **Strict Neo4j-only mode** | Prevent inconsistent local backup answers |
| **MedGemma direct inference** | Pragmatic: don't let perfect be enemy of good |
| **4-section output format** | Clinician-friendly structured diagnosis |
| **Conversational intake** | Guides patients through relevant questions |
| **Medical safety guards** | Passive disclaimers + active recommendations |

## Security & Privacy

### Data Sanitization
- User messages sanitized via `privacy.py`
- PII detection (optional via NER)
- No data stored persistently by default
- Session IDs generated client-side

### Backend Security
- CORS enabled for localhost (customize for production)
- No API key in frontend code
- Environment variables for secrets
- SQL injection handled via ORM/parameterized queries

## Contributing

### Code Style
```bash
# Format with black
black hyperderm/

# Type check with mypy
mypy hyperderm/

# Lint with ruff
ruff check hyperderm/
```

### Adding Features
1. Create feature branch: `git checkout -b feature/my-feature`
2. Add tests to `tests/test_multi_agent_chatbot_graph.py`
3. Ensure all tests pass: `pytest tests/ -q`
4. Submit PR with clear description

## License

MIT License - See LICENSE file

## References

- **Neo4j Documentation**: https://neo4j.com/docs/
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **Bytez API**: https://bytez.com/docs
- **Medical Terminology**: SNOMED CT, ICD-11

## Support

### Common Questions

**Q: Can I use a different LLM?**  
A: Yes. Replace `medgemma_chat_service.py` with your LLM provider (OpenAI, Anthropic, local Ollama).

**Q: Does this work offline?**  
A: Partially. Neo4j becomes unavailable, but local JSONL backup works (disable with `STRICT_NEO4J_ONLY=true`).

**Q: How accurate are the diagnoses?**  
A: This is a **decision support tool, not a substitute for medical expertise**. Always recommend in-person dermatologist evaluation. Graph + LLM together provide better context than either alone.

**Q: Can I run this on my laptop?**  
A: Yes. Use Neo4j Community Edition locally and Ollama for LLM. No cloud dependencies required.

---

## Project Status

**Last Updated**: April 2026  
**Status**: Production-ready with logging & error handling  
**Tests**: 9/9 passing  
**Known Limitations**:
- Bytez free tier rate limiting (1 req/sec)
- Graph requires manual disease entry or CSV import
- No image analysis yet (prepared for future CV integration)

## Contact & Issues

For bugs, feature requests, or questions:
- GitHub Issues: https://github.com/Avishek8136/IndiDermX/issues
- Email: avishekrauniyar07@gmail.com

---

**Disclaimer**: This is an AI-assisted diagnostic tool for educational and supportive purposes only. It is not a substitute for professional medical diagnosis, treatment, or advice. Always consult a qualified dermatologist for any skin concerns.