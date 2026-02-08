# ADR-0002: Authentication Approach

**Status:** Accepted  
**Date:** 2025-09-20  
**Authors:** Ontus Team

## Context

RDF-StarBase needs authentication and authorization for:
1. Multi-user deployments
2. API access control
3. Audit compliance
4. Rate limiting

Options considered:
- **API Keys** — Simple, stateless
- **OAuth2/OIDC** — Standard, complex
- **mTLS** — Service-to-service, no user identity
- **JWT only** — Requires external IdP

## Decision

Implement **API Key-based authentication** as the primary method, with:
1. SHA-256 hashed key storage (no plaintext)
2. Role-based access control (RBAC)
3. Operation scoping per key
4. Rate limiting per key
5. Audit logging of all operations

Future: Add OAuth2/OIDC as secondary method.

### Role Model

```
┌─────────────────────────────────────────────────┐
│                    ADMIN                         │
│  ┌──────────────────────────────────────────┐   │
│  │              WRITER                       │   │
│  │  ┌─────────────────────────────────┐     │   │
│  │  │           READER                 │     │   │
│  │  │  • QUERY                         │     │   │
│  │  │  • DESCRIBE                      │     │   │
│  │  │  • EXPORT                        │     │   │
│  │  └─────────────────────────────────┘     │   │
│  │  • INSERT, DELETE, UPDATE, LOAD          │   │
│  └──────────────────────────────────────────┘   │
│  • CREATE_REPO, DELETE_REPO, BACKUP, CONFIG     │
└─────────────────────────────────────────────────┘
```

### Key Scoping

Keys can be restricted to:
- Specific repositories
- Specific operations
- Rate limits (queries/min, triples/min)
- Expiration dates

## Rationale

### Why API Keys First?

| Factor | API Keys | OAuth2 | mTLS |
|--------|----------|--------|------|
| Simplicity | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| No external deps | ✅ | ❌ | ✅ |
| User-friendly | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Revocable | ✅ | ✅ | ⚠️ |
| Audit trail | ✅ | ✅ | ⚠️ |

API keys are:
- Easy to implement
- Easy to understand
- Easy to integrate
- Sufficient for most use cases

### Why Hashed Storage?

Storing plaintext keys is a security risk. If the key database leaks:
- **Plaintext**: Attacker has immediate access
- **Hashed**: Attacker must brute-force

We use SHA-256 with a prefix for identification:
```python
key_id = "rsb_" + secrets.token_hex(4)  # e.g., "rsb_a1b2c3d4"
full_key = key_id + secrets.token_hex(28)  # 256 bits total
stored_hash = hashlib.sha256(full_key.encode()).hexdigest()
```

### Why RBAC Over ABAC?

Attribute-Based Access Control (ABAC) is more flexible but:
- Complex policy language
- Harder to audit
- Overkill for our use case

RBAC is:
- Well-understood
- Easy to audit ("who has ADMIN?")
- Sufficient for repository-level access

## Implementation

### Key Structure

```python
@dataclass
class APIKey:
    key_id: str              # Prefix visible for identification
    key_hash: str            # SHA-256 of full key
    name: str                # Human-readable name
    role: Role               # READER, WRITER, ADMIN
    created_at: datetime
    expires_at: datetime | None
    enabled: bool
    allowed_repos: set[str] | None
    allowed_operations: set[Operation] | None
    rate_limit_queries: int | None
    rate_limit_ingestion: int | None
```

### Request Flow

```
1. Extract key from Authorization header
2. Look up key by prefix (key_id)
3. Verify hash matches
4. Check enabled + not expired
5. Check role permits operation
6. Check scoped permissions
7. Check rate limits
8. Log to audit
9. Execute request
```

## Consequences

### Positive
- Simple to implement and use
- No external dependencies
- Full audit trail
- Granular scoping possible

### Negative
- No SSO (until OAuth2 added)
- Keys must be securely distributed
- No short-lived tokens (like JWT)

### Migration Path

OAuth2/OIDC can be added later:
1. Add OIDC discovery endpoint config
2. Accept JWT in Authorization header
3. Map OIDC claims to internal roles
4. API keys remain as fallback

## Alternatives Considered

1. **OAuth2-only** — Rejected: Requires external IdP setup
2. **mTLS-only** — Rejected: Poor UX for humans
3. **Session cookies** — Rejected: Stateful, not for API
4. **No auth** — Rejected: Unacceptable for production

## References

- [OWASP API Security](https://owasp.org/www-project-api-security/)
- [OAuth 2.0 RFC 6749](https://datatracker.ietf.org/doc/html/rfc6749)
- [API Key Best Practices](https://cloud.google.com/docs/authentication/api-keys)
