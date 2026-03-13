# Bulk Load Feature Guide

## Overview

RDF-StarBase provides GraphDB-style bulk loading for astronomical RDF files with:

- **Streaming ingestion** - Handle multi-GB files without memory overflow
- **Compression support** - `.gz`, `.bz2`, `.zip` formats
- **Batch provenance** - Set metadata for entire file imports
- **Smart deduplication** - Append-only with competing claims
- **Transaction support** - Rollback on failure
- **Progress tracking** - Real-time job monitoring

## Quick Start

### 1. Setup Directory Structure

```bash
# Create import directories
mkdir -p data/import/gleif
mkdir -p data/import/fibo

# Copy your RDF files
cp /path/to/L1Data.ttl data/import/gleif/
cp /path/to/prod.ttl.zip data/import/fibo/
```

### 2. Start the Server

```bash
# Local development
source venv/bin/activate
PYTHONPATH=src uvicorn api.web:app --reload

# Docker
cd deploy/compose
docker compose up -d
```

### 3. Submit Bulk Load Job

```bash
curl -X POST http://localhost:8000/repositories/gleif/bulk-load \
  -H "Content-Type: application/json" \
  -d '{
    "files": ["L1Data.ttl", "L2Data.ttl"],
    "provenance": {
      "source": "gleif:2026-02",
      "confidence": 1.0,
      "process": "bulk_import"
    }
  }'
```

Response:
```json
{
  "success": true,
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "files": 2,
  "monitor_url": "/repositories/gleif/bulk-load/jobs/550e8400-..."
}
```

### 4. Monitor Progress

```bash
curl http://localhost:8000/repositories/gleif/bulk-load/jobs/550e8400-...
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": {
    "percent_complete": 45.2,
    "triples_loaded": 1234567,
    "triples_per_second": 25000,
    "elapsed_seconds": 49.4,
    "current_file": "L1Data.ttl"
  }
}
```

## Deduplication Strategy

The bulk loader uses an **append-only** strategy with smart deduplication:

| Scenario | Action | Reason |
|----------|--------|--------|
| Triple + metadata exists | **SKIP** | Exact duplicate |
| Triple exists, different metadata | **ADD** | Competing claim |
| Triple doesn't exist | **ADD** | New assertion |

This preserves provenance tracking while avoiding exact duplicates.

### Example

```python
# First import from CRM
POST /repositories/customers/bulk-load
{
  "files": ["crm_export.ttl"],
  "provenance": {
    "source": "CRM_System",
    "confidence": 0.85
  }
}
# Result: 1000 triples loaded

# Second import from same source (no changes)
POST /repositories/customers/bulk-load
{
  "files": ["crm_export.ttl"],  # Same file
  "provenance": {
    "source": "CRM_System",      # Same source
    "confidence": 0.85           # Same confidence
  }
}
# Result: 0 triples loaded, 1000 skipped (duplicates)

# Import from different source (same data)
POST /repositories/customers/bulk-load
{
  "files": ["data_lake_export.ttl"],  # Same triples
  "provenance": {
    "source": "DataLake",             # Different source!
    "confidence": 0.92                # Different confidence!
  }
}
# Result: 1000 triples loaded (competing claims preserved)
```

## Compression Support

Automatically detects and decompresses:

```bash
# Gzip
data/import/gleif/L1Data.ttl.gz

# Bzip2
data/import/gleif/L1Data.ttl.bz2

# Zip archives (uses first file)
data/import/fibo/prod.ttl.zip
```

## Docker Usage

### Volume Mounts

The docker-compose configuration provides:

```yaml
volumes:
  - ./data/import:/data/import:ro           # Read-only source
  - rdf-starbase-data:/data/repositories    # Persistent stores
  - ./data/export:/data/export              # Backups
```

### Workflow

```bash
# 1. Place files in host directory
cp L1Data.ttl.gz ./data/import/gleif/

# 2. Files are visible inside container at /data/import/
docker compose exec rdf-starbase ls /data/import/gleif/

# 3. Trigger bulk load via API
curl -X POST http://localhost:8000/repositories/gleif/bulk-load \
  -H "Content-Type: application/json" \
  -d '{"files": ["L1Data.ttl.gz"]}'

# 4. Data persists in named volume
docker volume inspect rdf-starbase-data
```

## API Endpoints

### Submit Bulk Load

```
POST /repositories/{name}/bulk-load
```

**Request:**
```json
{
  "files": ["file1.ttl", "file2.ttl.gz"],
  "provenance": {
    "source": "source_name",
    "confidence": 1.0,
    "process": "bulk_import",
    "metadata": {
      "custom_field": "value"
    }
  }
}
```

### Get Job Status

```
GET /repositories/{name}/bulk-load/jobs/{job_id}
```

### List Jobs

```
GET /repositories/{name}/bulk-load/jobs?status=running
```

### Cancel Job

```
DELETE /repositories/{name}/bulk-load/jobs/{job_id}
```

## Benchmarking

### Run Benchmarks

```bash
# GLEIF dataset
python benchmarks/bulk_load_benchmark.py --dataset gleif

# FIBO dataset
python benchmarks/bulk_load_benchmark.py --dataset fibo

# Both
python benchmarks/bulk_load_benchmark.py --dataset all
```

### Expected Performance

| Dataset | Size | Triples | Time | Throughput |
|---------|------|---------|------|------------|
| GLEIF L1 | 8.7 GB | ~30M | ~20 min | 25K t/s |
| GLEIF L2 | 781 MB | ~3M | ~2 min | 25K t/s |
| FIBO | Variable | Variable | Variable | 20K t/s |

*Performance varies by hardware (CPU, disk I/O, memory)*

## Configuration

### Environment Variables

```bash
# Import directory
RDFSTARBASE_IMPORT_PATH=/data/import

# Repository storage
RDFSTARBASE_REPOSITORY_PATH=/data/repositories

# Bulk load tuning
RDFSTARBASE_BULK_BATCH_SIZE=100000      # Triples per batch
RDFSTARBASE_BULK_MAX_WORKERS=4          # Concurrent threads
RDFSTARBASE_BULK_COMPRESSION=gz,bz2,zip # Supported formats
```

### Tuning

**For large files (>5GB):**
- Increase `BULK_BATCH_SIZE` to 200K-500K
- Ensure sufficient RAM (2-4 GB per worker)

**For many small files:**
- Increase `MAX_WORKERS` to 8-16
- Reduce `BATCH_SIZE` to 50K

**For slow storage:**
- Reduce `MAX_WORKERS` to 1-2
- Use SSD if possible

## Transaction Support

Bulk loads run in transactions for safety:

```python
# If job fails, all changes are rolled back
job = service.submit_job(...)

# Transaction is committed only on success
if job.status == "completed":
    # All triples persisted atomically
    pass
elif job.status == "failed":
    # No partial data (rolled back)
    print(f"Error: {job.error_message}")
```

## Security

### API Key Protection

```bash
# Require API key for bulk load operations
curl -X POST http://localhost:8000/repositories/gleif/bulk-load \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"files": ["data.ttl"]}'
```

### Read-Only Mount

Docker mounts import directory as read-only (`:ro`) to prevent accidental modification:

```yaml
volumes:
  - ./data/import:/data/import:ro  # Cannot delete/modify source files
```

## Troubleshooting

### Job stuck in "pending"

```bash
# Check logs
docker compose logs -f

# Verify files exist
docker compose exec rdf-starbase ls /data/import/gleif/
```

### Out of memory

Reduce batch size:
```bash
RDFSTARBASE_BULK_BATCH_SIZE=50000 \
  python -m uvicorn api.web:app
```

### Slow performance

Check disk I/O:
```bash
# Monitor during load
docker stats rdf-starbase
```

### File not found

Ensure correct path (relative to import directory):
```json
{
  "files": ["gleif/L1Data.ttl"]  // Correct
  // NOT: ["data/import/gleif/L1Data.ttl"]
}
```

## Python API

```python
from api.bulk_load_service import BulkLoadService
from rdf_starbase.repositories import RepositoryManager

# Initialize
manager = RepositoryManager("./data/repositories")
service = BulkLoadService(
    repository_manager=manager,
    import_base_path="./data/import",
)

# Submit job
job = service.submit_job(
    repository_name="gleif",
    files=["gleif/L1Data.ttl.gz"],
    batch_provenance={
        "source": "gleif:2026-02",
        "confidence": 1.0,
    }
)

# Monitor
while job.status in ("pending", "running"):
    await asyncio.sleep(1)
    job = service.get_job_status(job.job_id)
    print(f"{job.progress.percent_complete}%")

print(f"Loaded {job.progress.triples_loaded:,} triples")
```

## Next Steps

- See [benchmarks/bulk_load_benchmark.py](../benchmarks/bulk_load_benchmark.py) for complete examples
- Read [storage-spec.md](../storage-spec.md) for columnar architecture details
- Check [ROADMAP.md](../ROADMAP.md) for future enhancements
