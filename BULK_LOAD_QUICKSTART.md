# Bulk Load Quick Start

## Setup GLEIF Data

```bash
# Already have GLEIF files in data/gleif-lei-data/
# Move them to import directory
ln -s ../gleif-lei-data data/import/gleif

# Verify files
ls -lh data/import/gleif/
# Expected: L1Data.ttl (8.7GB), L2Data.ttl (781MB), RepExData.ttl (1.7GB)
```

## Setup FIBO Data

```bash
# Already have FIBO in data/fibo/
ln -s ../fibo data/import/fibo

# Verify
ls -lh data/import/fibo/
# Expected: prod.ttl.zip
```

## Start Server (Local)

```bash
source venv/bin/activate
PYTHONPATH=src uvicorn api.web:app --reload --host 0.0.0.0 --port 8000
```

## Test Bulk Load API

### Create Repository

```bash
curl -X POST http://localhost:8000/repositories \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gleif-test",
    "description": "GLEIF bulk load test",
    "tags": ["gleif", "benchmark"]
  }'
```

### Submit Bulk Load Job

```bash
curl -X POST http://localhost:8000/repositories/gleif-test/bulk-load \
  -H "Content-Type: application/json" \
  -d '{
    "files": ["L1Data.ttl"],
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
  "job_id": "abc123...",
  "status": "pending",
  "monitor_url": "/repositories/gleif-test/bulk-load/jobs/abc123..."
}
```

### Monitor Progress

```bash
# Watch progress
watch -n 2 'curl -s http://localhost:8000/repositories/gleif-test/bulk-load/jobs/abc123... | jq .progress'
```

## Run Benchmarks

```bash
# GLEIF benchmark (8.7GB + 781MB + 1.7GB = 11GB)
python benchmarks/bulk_load_benchmark.py --dataset gleif

# FIBO benchmark
python benchmarks/bulk_load_benchmark.py --dataset fibo

# Both
python benchmarks/bulk_load_benchmark.py --dataset all
```

## Docker Usage

```bash
cd deploy/compose

# Start container
docker compose up -d

# Follow logs
docker compose logs -f

# API available at http://localhost:8000
curl http://localhost:8000/health
```

### Bulk Load via Docker

```bash
# Files already mounted at /data/import/
# Submit job
curl -X POST http://localhost:8000/repositories/gleif/bulk-load \
  -H "Content-Type: application/json" \
  -d '{
    "files": ["L1Data.ttl"],
    "provenance": {"source": "gleif"}
  }'
```

## Expected Performance

**GLEIF L1Data.ttl (8.7 GB, ~30M triples):**
- Parse speed: ~25,000 triples/second
- Total time: ~20 minutes
- Memory: ~4 GB peak

**Hardware used for estimates:**
- CPU: Apple M1/M2 or similar
- RAM: 16 GB
- Disk: SSD

Your mileage may vary!

## Troubleshooting

### Import failed: "File not found"

Files must be in `data/import/{repository}/`:

```bash
# Wrong
curl ... -d '{"files": ["/Users/.../L1Data.ttl"]}'

# Correct
curl ... -d '{"files": ["L1Data.ttl"]}'
```

### Out of memory

Reduce batch size:
```bash
export RDFSTARBASE_BULK_BATCH_SIZE=50000
python -m uvicorn api.web:app
```

### Slow performance

Check disk I/O (should be on SSD):
```bash
df -h data/
```

## Next Steps

- Read [docs/BULK_LOAD_GUIDE.md](docs/BULK_LOAD_GUIDE.md) for full documentation
- See [benchmarks/bulk_load_benchmark.py](benchmarks/bulk_load_benchmark.py) for Python API usage
- Check [deploy/compose/docker-compose.yml](deploy/compose/docker-compose.yml) for configuration
