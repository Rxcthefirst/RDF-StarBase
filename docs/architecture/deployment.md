# Deployment Architecture

> Docker, Kubernetes, and infrastructure assumptions

## Deployment Options

| Option | Use Case | Complexity |
|--------|----------|------------|
| Docker single container | Development, small deployments | Low |
| Docker Compose | Multi-container with orchestration | Medium |
| Kubernetes | Production, scaling, HA | High |

---

## Docker Single Container

The default Docker image runs both API and UI:

```bash
docker run -d \
  --name rdfstarbase \
  -p 8000:8000 \
  -v rdfstarbase-data:/data/repositories \
  rxcthefirst/rdf-starbase:latest
```

### Ports
| Port | Service |
|------|---------|
| 8000 | FastAPI (REST + static UI) |

### Volumes
| Path | Purpose |
|------|---------|
| `/data/repositories` | Persistent storage for all repositories |

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `RDFSTARBASE_HOST` | `0.0.0.0` | Bind address |
| `RDFSTARBASE_PORT` | `8000` | HTTP port |
| `RDFSTARBASE_REPOSITORY_PATH` | `/data/repositories` | Data directory |
| `RDFSTARBASE_SERVE_STATIC` | `true` | Serve UI from API |
| `RDFSTARBASE_LOG_LEVEL` | `info` | Logging verbosity |

---

## Docker Compose

For multi-service setups:

```yaml
# deploy/compose/docker-compose.yml
services:
  rdfstarbase:
    build:
      context: ../..
      dockerfile: deploy/docker/Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - rdfstarbase-data:/data/repositories
    environment:
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  rdfstarbase-data:
```

### Running
```bash
cd deploy/compose
docker-compose up -d
```

---

## Kubernetes

### Architecture
```
┌─────────────────────────────────────────────────┐
│                   Ingress                        │
│              (nginx/traefik)                     │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│                  Service                         │
│           (ClusterIP, port 8000)                │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│               StatefulSet                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │  Pod 0  │  │  Pod 1  │  │  Pod N  │         │
│  │         │  │         │  │         │         │
│  │  PVC 0  │  │  PVC 1  │  │  PVC N  │         │
│  └─────────┘  └─────────┘  └─────────┘         │
└─────────────────────────────────────────────────┘
```

### Key Resources

**StatefulSet** — Stable network IDs, ordered deployment:
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: rdfstarbase
spec:
  serviceName: rdfstarbase
  replicas: 1
  selector:
    matchLabels:
      app: rdfstarbase
  template:
    spec:
      containers:
      - name: rdfstarbase
        image: rxcthefirst/rdf-starbase:latest
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: data
          mountPath: /data/repositories
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

**PersistentVolumeClaim** — Durable storage per pod

**Ingress** — HTTPS termination, path routing

**ServiceMonitor** — Prometheus metrics scraping

### Helm Chart

Located at `deploy/helm/`:
```
deploy/helm/rdfstarbase/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── statefulset.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   └── servicemonitor.yaml
```

Install:
```bash
helm install rdfstarbase ./deploy/helm/rdfstarbase \
  --set replicaCount=1 \
  --set persistence.size=50Gi
```

---

## Health Endpoints

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `GET /health` | Liveness probe | `{"status": "healthy"}` |
| `GET /ready` | Readiness probe | `{"status": "ready"}` |
| `GET /metrics` | Prometheus metrics | Prometheus format |

---

## Scaling Considerations

### Single-Node (Current)
- Vertical scaling via CPU/memory
- Storage scales with disk
- ~10M triples per instance typical

### Future (Multi-Node)
- Read replicas for query scaling
- Sharding by predicate or graph
- External coordination (etcd/Consul)

---

## Backup Strategy

1. **Application snapshots**: `POST /repositories/{name}/backup`
2. **Volume snapshots**: Cloud provider PVC snapshots
3. **Export**: Periodic RDF export to object storage

Recommended: Daily snapshots + weekly exports to S3/GCS.
