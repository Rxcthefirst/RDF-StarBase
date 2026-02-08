# Local Development Setup

> Getting started with RDF-StarBase development

## Prerequisites

- Python 3.10+ (3.12 recommended)
- Node.js 18+ (for UI development)
- Git

## Quick Setup

### 1. Clone Repository

```bash
git clone https://github.com/ontus/rdf-starbase.git
cd rdf-starbase
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Full development install
pip install -e ".[dev,web,query,sql]"
```

This installs:
- `dev` — pytest, black, ruff, mypy
- `web` — FastAPI, uvicorn
- `query` — SPARQL parsing
- `sql` — DuckDB integration

### 4. Verify Installation

```bash
# Run tests
pytest -x -q

# Check imports
python -c "from rdf_starbase import TripleStore; print('OK')"
python -c "from api.web import create_app; print('OK')"
```

---

## Running the Application

### API Server

```bash
# Development mode with auto-reload
uvicorn api.web:app --reload --port 8000

# Or via Python
python -m uvicorn api.web:app --reload
```

Access:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- UI: http://localhost:8000/app/ (if UI built)

### Building the UI

```bash
cd src/ui
npm install
npm run build    # Production build
npm run dev      # Development server (port 5173)
```

---

## Development Workflow

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_auth.py

# With coverage
pytest --cov=src/rdf_starbase --cov-report=html

# Fast subset
pytest -x -q --tb=short
```

### Code Formatting

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/rdf_starbase
```

### Pre-commit (Optional)

```bash
pip install pre-commit
pre-commit install
```

---

## Project Structure

```
rdf-starbase/
├── src/
│   ├── api/              # REST API layer
│   │   ├── auth.py       # Authentication
│   │   ├── web.py        # FastAPI app
│   │   └── repository_api.py
│   ├── rdf_starbase/     # Core engine
│   │   ├── store.py      # TripleStore
│   │   ├── executor.py   # SPARQL
│   │   └── storage/      # Storage layer
│   └── ui/               # React frontend
├── tests/                # Test suite
├── docs/                 # Documentation
├── deploy/               # Docker/K8s configs
└── pyproject.toml        # Package config
```

---

## Common Tasks

### Create a Test Repository

```python
from rdf_starbase import TripleStore

store = TripleStore()
store.add_triple(
    "http://example.org/alice",
    "http://xmlns.com/foaf/0.1/name",
    "Alice"
)
results = store.query("SELECT * WHERE { ?s ?p ?o }")
```

### Run SPARQL Query via API

```bash
curl -X POST http://localhost:8000/repositories/test/sparql \
  -H "Content-Type: application/sparql-query" \
  -d "SELECT * WHERE { ?s ?p ?o } LIMIT 10"
```

### Import Data

```bash
curl -X POST http://localhost:8000/repositories/test/import \
  -H "Content-Type: text/turtle" \
  -d "@data/sample/example.ttl"
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RDFSTARBASE_REPOSITORY_PATH` | `./data/repositories` | Data directory |
| `RDFSTARBASE_LOG_LEVEL` | `info` | Logging level |
| `RDFSTARBASE_HOST` | `127.0.0.1` | Bind address |
| `RDFSTARBASE_PORT` | `8000` | HTTP port |

---

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'rdf_starbase'
```

**Fix:** Ensure you installed in editable mode:
```bash
pip install -e "."
```

### Port Already in Use

```
ERROR: [Errno 98] Address already in use
```

**Fix:** Kill the process or use a different port:
```bash
uvicorn api.web:app --port 8001
```

### Tests Failing

```bash
# Run with verbose output
pytest -v --tb=long

# Run specific failing test
pytest tests/test_file.py::test_function -v
```

---

## IDE Setup

### VS Code

Recommended extensions:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Black Formatter
- Ruff

Settings (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": ".venv/Scripts/python",
  "python.formatting.provider": "black",
  "editor.formatOnSave": true
}
```

### PyCharm

1. Mark `src/` as Sources Root
2. Set interpreter to `.venv`
3. Enable Black formatter
