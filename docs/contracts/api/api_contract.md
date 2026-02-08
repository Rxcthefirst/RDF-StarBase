# REST API Contract

> API specification and examples

## Base URL

```
http://localhost:8000
```

## Authentication

All requests require an API key (except `/health`, `/ready`):

```http
Authorization: Bearer rsb_your_api_key_here
```

---

## Core Endpoints

### Repositories

#### List Repositories
```http
GET /repositories
```

Response:
```json
{
  "repositories": [
    {
      "name": "my-repo",
      "uuid": "550e8400-e29b-41d4-a716-446655440000",
      "created": "2026-01-15T10:30:00Z",
      "triple_count": 125000,
      "graph_count": 5
    }
  ]
}
```

#### Create Repository
```http
POST /repositories
Content-Type: application/json

{
  "name": "my-repo",
  "config": {
    "reasoning_enabled": true,
    "memory_limit_mb": 512
  }
}
```

#### Delete Repository
```http
DELETE /repositories/{name}
```

---

### SPARQL Queries

#### Execute Query
```http
POST /repositories/{name}/sparql
Content-Type: application/sparql-query
Accept: application/sparql-results+json

SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10
```

Response:
```json
{
  "head": {"vars": ["s", "p", "o"]},
  "results": {
    "bindings": [
      {
        "s": {"type": "uri", "value": "http://example.org/alice"},
        "p": {"type": "uri", "value": "http://xmlns.com/foaf/0.1/name"},
        "o": {"type": "literal", "value": "Alice"}
      }
    ]
  }
}
```

#### SPARQL Update
```http
POST /repositories/{name}/sparql/update
Content-Type: application/sparql-update

INSERT DATA { <http://example.org/bob> <http://xmlns.com/foaf/0.1/name> "Bob" }
```

---

### Import/Export

#### Import RDF
```http
POST /repositories/{name}/import
Content-Type: text/turtle

@prefix foaf: <http://xmlns.com/foaf/0.1/> .
<http://example.org/alice> foaf:name "Alice" .
```

Supported formats:
- `text/turtle` — Turtle
- `application/n-triples` — N-Triples
- `application/ld+json` — JSON-LD
- `application/trig` — TriG
- `application/n-quads` — N-Quads
- `application/rdf+xml` — RDF/XML

#### Export Repository
```http
GET /repositories/{name}/export?format=turtle
Accept: text/turtle
```

---

### AI/Grounding Endpoints

#### Query for RAG
```http
POST /ai/query
Content-Type: application/json

{
  "repository": "my-repo",
  "question": "Who knows Alice?",
  "max_results": 10
}
```

#### Verify Claim
```http
POST /ai/verify
Content-Type: application/json

{
  "repository": "my-repo",
  "claim": "Bob is a friend of Alice",
  "threshold": 0.8
}
```

---

### Administration

#### Backup
```http
POST /repositories/{name}/backup
```

Response:
```json
{
  "backup_path": "/data/backups/my-repo-20260205-103000.tar.gz",
  "size_bytes": 15234567
}
```

#### Metrics
```http
GET /metrics
Accept: text/plain
```

Returns Prometheus exposition format.

---

## Error Responses

```json
{
  "error": "NotFound",
  "message": "Repository 'unknown' does not exist",
  "status_code": 404
}
```

| Status | Meaning |
|--------|---------|
| 400 | Bad Request — Invalid input |
| 401 | Unauthorized — Missing/invalid API key |
| 403 | Forbidden — Insufficient permissions |
| 404 | Not Found — Resource doesn't exist |
| 429 | Too Many Requests — Rate limited |
| 500 | Internal Error — Server error |

---

## OpenAPI Spec

Auto-generated OpenAPI available at:
```
GET /openapi.json
GET /docs      # Swagger UI
GET /redoc     # ReDoc UI
```
