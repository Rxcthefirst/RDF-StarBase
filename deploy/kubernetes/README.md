# RDF-StarBase Helm Chart

Deploy RDF-StarBase on Kubernetes with this Helm chart.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2+
- PV provisioner support (if persistence is enabled)

## Installation

### Add the repository (when published)

```bash
helm repo add ontus https://charts.ontus.io
helm repo update
```

### Install from local chart

```bash
# From the repository root
helm install rdfstarbase ./deploy/kubernetes
```

### Install with custom values

```bash
helm install rdfstarbase ./deploy/kubernetes -f my-values.yaml
```

## Quick Start

```bash
# Install with default settings
helm install rdfstarbase ./deploy/kubernetes

# Install with persistence disabled (for testing)
helm install rdfstarbase ./deploy/kubernetes --set persistence.enabled=false

# Install with ingress enabled
helm install rdfstarbase ./deploy/kubernetes \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=rdfstarbase.example.com \
  --set ingress.hosts[0].paths[0].path=/ \
  --set ingress.hosts[0].paths[0].pathType=Prefix
```

## Configuration

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Image repository | `ontus/rdfstarbase` |
| `image.tag` | Image tag | `""` (uses Chart.appVersion) |
| `persistence.enabled` | Enable persistence | `true` |
| `persistence.size` | PVC size | `50Gi` |
| `resources.requests.memory` | Memory request | `1Gi` |
| `resources.limits.memory` | Memory limit | `4Gi` |

### RDF-StarBase Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `config.logLevel` | Log level | `INFO` |
| `config.maxMemory` | Max memory for queries | `2147483648` |
| `config.queryTimeout` | Query timeout (seconds) | `300` |
| `config.reasoningEnabled` | Enable RDFS/OWL reasoning | `true` |
| `config.walEnabled` | Enable Write-Ahead Log | `true` |

### Health Checks

| Parameter | Description | Default |
|-----------|-------------|---------|
| `healthCheck.enabled` | Enable health probes | `true` |
| `healthCheck.livenessProbe.initialDelaySeconds` | Liveness delay | `10` |
| `healthCheck.readinessProbe.initialDelaySeconds` | Readiness delay | `5` |

### Metrics & Monitoring

| Parameter | Description | Default |
|-----------|-------------|---------|
| `metrics.enabled` | Enable Prometheus metrics | `true` |
| `metrics.port` | Metrics port | `9090` |
| `metrics.serviceMonitor.enabled` | Create ServiceMonitor | `false` |

### Multi-tenancy

| Parameter | Description | Default |
|-----------|-------------|---------|
| `multiTenancy.enabled` | Enable multi-tenancy | `false` |
| `multiTenancy.defaultQuotas.repositories` | Default repo limit | `10` |
| `multiTenancy.defaultQuotas.triplesPerRepo` | Default triple limit | `10000000` |

### Federation

| Parameter | Description | Default |
|-----------|-------------|---------|
| `federation.enabled` | Enable federation | `false` |
| `federation.endpoints` | Remote SPARQL endpoints | `[]` |

### Ingress

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ingress.enabled` | Enable ingress | `false` |
| `ingress.className` | Ingress class | `""` |
| `ingress.hosts` | Ingress hosts | See values.yaml |

## Production Deployment

### High Availability

For production, consider:

```yaml
replicaCount: 3

resources:
  requests:
    cpu: 1000m
    memory: 2Gi
  limits:
    cpu: 4000m
    memory: 8Gi

persistence:
  size: 200Gi
  storageClass: fast-ssd

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10

affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
              - key: app.kubernetes.io/name
                operator: In
                values:
                  - rdfstarbase
          topologyKey: kubernetes.io/hostname
```

### Monitoring

Enable Prometheus monitoring:

```yaml
metrics:
  enabled: true
  serviceMonitor:
    enabled: true
    labels:
      release: prometheus
```

### Ingress with TLS

```yaml
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: kg.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: rdfstarbase-tls
      hosts:
        - kg.example.com
```

## Upgrading

```bash
helm upgrade rdfstarbase ./deploy/kubernetes -f my-values.yaml
```

## Uninstalling

```bash
helm uninstall rdfstarbase
```

**Note:** PVCs are not deleted automatically. Remove manually if needed:

```bash
kubectl delete pvc -l app.kubernetes.io/name=rdfstarbase
```

## Troubleshooting

### Pod not starting

Check logs:
```bash
kubectl logs -l app.kubernetes.io/name=rdfstarbase
```

### Health check failing

Check endpoints:
```bash
kubectl port-forward svc/rdfstarbase 8000:8000
curl http://localhost:8000/health/ready
```

### Storage issues

Verify PVC:
```bash
kubectl get pvc -l app.kubernetes.io/name=rdfstarbase
kubectl describe pvc data-rdfstarbase-0
```
