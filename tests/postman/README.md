# RDF-StarBase Postman Test Suite

This directory contains the Postman collection for testing the RDF-StarBase API.

## Running Tests

### Option 1: Import into Postman

1. Open Postman
2. Click **Import** 
3. Select `RDF-StarBase-API.postman_collection.json`
4. Run the collection

### Option 2: Run with Newman (CLI)

Install Newman globally:

```bash
npm install -g newman
```

Run the tests:

```bash
# Basic run
newman run RDF-StarBase-API.postman_collection.json

# With detailed output
newman run RDF-StarBase-API.postman_collection.json --reporters cli,json --reporter-json-export results.json

# With HTML report
npm install -g newman-reporter-html
newman run RDF-StarBase-API.postman_collection.json -r cli,html --reporter-html-export report.html
```

### Option 3: Run from project root

```bash
cd e:\RDF-StarBase
npm run test:api
```

## Test Categories

| Folder | Tests | Description |
|--------|-------|-------------|
| Health & Info | 3 | API health, root info, stats |
| Triples | 6 | CRUD operations, filtering, competing claims |
| SPARQL | 6 | SELECT, ASK queries, parsing, error handling |
| Registry | 6 | Source management, filtering |
| Graph Visualization | 3 | Nodes, edges, subgraph |
| Performance Tests | 3 | Response time validation |
| **AI Grounding** | **12** | **Query, verify, context, materialize, inferences** |

## AI Grounding Tests

The AI Grounding folder tests the specialized API layer for LLM/RAG consumption:

| Test | Endpoint | Description |
|------|----------|-------------|
| AI Health Check | `GET /ai/health` | Verify AI API is operational |
| AI Query - All Facts | `POST /ai/query` | Retrieve facts with provenance |
| AI Query - High Confidence | `POST /ai/query` | Filter by confidence >= 0.9 |
| AI Query - By Subject | `POST /ai/query` | Filter facts by subject IRI |
| AI Query - Source Filter | `POST /ai/query` | Filter to specific sources |
| AI Verify - Supported Claim | `POST /ai/verify` | Verify claim against KB |
| AI Verify - Unknown Subject | `POST /ai/verify` | Test unknown subject handling |
| AI Context - Entity | `GET /ai/context/{iri}` | Get full entity context |
| AI Context - Confidence | `GET /ai/context/{iri}` | Context with confidence filter |
| AI Materialize | `POST /ai/materialize` | Run RDFS/OWL reasoning |
| AI Inferences | `GET /ai/inferences` | List materialized inferences |
| AI Workflow - RAG | `POST /ai/query` | Simulated RAG pipeline |

## Test Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `baseUrl` | `http://localhost:8000` | API server URL |
| `sourceId` | (auto-set) | First source ID from registry |
| `newSourceId` | (auto-set) | ID of newly created source |

## Prerequisites

Make sure the API server is running with demo data:

```bash
cd e:\RDF-StarBase
python scripts/run_demo.py
```

## Performance Thresholds

- Health check: < 100ms
- Simple queries: < 300ms  
- SPARQL queries: < 500ms
- Bulk operations: < 1000ms
