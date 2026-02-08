# Security Model

> Authentication, authorization, audit, and data protection

## Overview

RDF-StarBase implements defense-in-depth security:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Request Flow                                │
├─────────────────────────────────────────────────────────────────┤
│  1. TLS termination (Ingress/Load Balancer)                     │
│  2. API Key validation → api/auth.py                            │
│  3. Rate limiting check                                          │
│  4. Role-based authorization                                     │
│  5. Operation scoping                                            │
│  6. Audit logging                                                │
│  7. Execute request                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Authentication

### API Keys

Primary authentication method. Keys are:
- 256-bit cryptographically random tokens
- SHA-256 hashed before storage
- Prefix-visible for identification (e.g., `rsb_abc123...`)

```python
from api.auth import APIKeyManager

manager = APIKeyManager(storage_path="/data/keys")
key = manager.create_key(
    name="my-service",
    role=Role.WRITER,
    allowed_repos={"production"},
    expires_in_days=90
)
# Returns: "rsb_a1b2c3d4..."  (store securely!)
```

### Request Authentication

Include key in header:
```
Authorization: Bearer rsb_a1b2c3d4...
```

Or query parameter (not recommended):
```
GET /repositories?api_key=rsb_a1b2c3d4...
```

---

## Authorization (RBAC)

### Roles

| Role | Read | Write | Admin |
|------|------|-------|-------|
| **READER** | ✅ | ❌ | ❌ |
| **WRITER** | ✅ | ✅ | ❌ |
| **ADMIN** | ✅ | ✅ | ✅ |

### Operations by Category

**Read Operations:**
- `QUERY` — Execute SPARQL queries
- `DESCRIBE` — Get resource descriptions
- `EXPORT` — Download repository data

**Write Operations:**
- `INSERT` — Add triples
- `DELETE` — Remove triples
- `UPDATE` — Modify data
- `LOAD` — Import from files

**Admin Operations:**
- `CREATE_REPO` — Create repositories
- `DELETE_REPO` — Delete repositories
- `BACKUP` — Create backups
- `RESTORE` — Restore from backup
- `CONFIG` — Modify configuration
- `MANAGE_KEYS` — Create/revoke API keys

### Scoped Tokens

Create tokens limited to specific repos/operations:

```python
key = manager.create_key(
    name="reporting-service",
    role=Role.READER,
    allowed_repos={"analytics", "staging"},
    allowed_operations={Operation.QUERY, Operation.EXPORT}
)
```

---

## Rate Limiting

Per-key limits prevent abuse:

```python
key = manager.create_key(
    name="batch-loader",
    role=Role.WRITER,
    rate_limit_queries=100,      # queries per minute
    rate_limit_ingestion=50000   # triples per minute
)
```

Exceeded limits return `429 Too Many Requests`.

---

## Audit Logging

All operations are logged:

```json
{
  "timestamp": "2026-02-05T10:30:00Z",
  "key_id": "rsb_abc123",
  "key_name": "my-service",
  "operation": "QUERY",
  "repository": "production",
  "client_ip": "192.168.1.100",
  "duration_ms": 45,
  "status": "success"
}
```

### Audit Log Access

```bash
# Export audit log
GET /admin/audit?format=json&from=2026-01-01

# CSV export for compliance
GET /admin/audit?format=csv
```

---

## Multi-Tenancy

Isolated namespaces with resource quotas:

```python
from rdf_starbase.storage.tenancy import TenantManager

tm = TenantManager(base_path="/data")
tm.create_tenant(
    name="acme-corp",
    quota_repos=10,
    quota_storage_gb=50,
    quota_queries_per_day=10000
)
```

Tenants cannot access each other's data.

---

## Data Protection

### At Rest
- Repository data in Parquet files
- Consider volume encryption (LUKS, cloud KMS)
- API keys hashed with SHA-256

### In Transit
- TLS 1.3 recommended (terminate at load balancer)
- No sensitive data in URLs

### Secrets Management
- API keys: store in secrets manager (Vault, AWS Secrets)
- Config: use environment variables, not files

---

## Security Checklist

### Deployment
- [ ] TLS enabled on all endpoints
- [ ] API keys rotated every 90 days
- [ ] Audit logs exported to SIEM
- [ ] Network policies restrict pod communication
- [ ] Secrets in Kubernetes Secrets (not ConfigMaps)

### Development
- [ ] No hardcoded credentials
- [ ] Dependencies scanned for vulnerabilities
- [ ] Security headers enabled (CORS, CSP)

---

## Incident Response

1. **Revoke compromised key**: `DELETE /admin/keys/{key_id}`
2. **Check audit logs**: Identify scope of unauthorized access
3. **Rotate all keys** if systemic compromise suspected
4. **Export audit evidence**: `GET /admin/audit?format=json`

---

## Future Enhancements

- OAuth2/OIDC integration
- SAML for enterprise SSO
- Graph-level access control
- Field-level encryption
