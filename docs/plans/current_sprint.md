# Current Sprint

> What to work on now

## Sprint: February 2026

### Theme: API/Engine Separation & Documentation

We recently completed a major refactoring to separate the API layer from the core engine. Current focus is on stabilizing this structure and improving documentation.

---

## Active Tasks

### 1. Documentation Refresh (HIGH) ✅
- [x] Separate `src/api/` from `src/rdf_starbase/`
- [x] Update FOLDER_STRUCTURE.md
- [x] Complete docs/architecture files
- [x] Complete docs/runbooks
- [ ] Update README with new import paths

### 2. Auth Enhancements (HIGH) ✅
- [x] Add OAuth2/OIDC support (src/api/oidc.py)
- [x] Add JWT token validation (included in OIDC module)
- [x] Improve rate limiting configurability (RateLimitConfig class)
- [x] Add audit log rotation (rotate, archive methods)

### 3. Testing & Stability (MEDIUM)
- [ ] Increase test coverage to 80%
- [ ] Add integration tests for API layer
- [ ] Performance regression tests

---

## Completed This Sprint

- ✅ Created `src/api/` package with auth, web, repository_api, ai_grounding
- ✅ Renamed `src/frontend/` → `src/ui/`
- ✅ Reorganized `deploy/` folder structure
- ✅ Added backward-compatible shims with deprecation warnings
- ✅ All 1526 tests passing
- ✅ OAuth2/OIDC authentication module (`src/api/oidc.py`)
- ✅ JWT token validation with JWKS support
- ✅ Provider templates for Keycloak, Azure AD, Okta, Auth0
- ✅ Configurable rate limiting (`RateLimitConfig` class)
- ✅ Audit log rotation and archiving

---

## Import Path Changes

**New paths (preferred):**
```python
from api.auth import APIKeyManager, Role
from api.web import create_app
```

**Deprecated paths (still work):**
```python
from rdf_starbase.web import create_app  # DeprecationWarning
```

---

## Next Sprint Preview

### v2.0.0 — Data Integration Platform

**Priority 1: Data Integration**
- Starchart visual RML/R2RML mapper (standalone `/mapper` UI)
- ONTOP virtualized data (PostgreSQL, MySQL connectors)
- RDFMap ETL pipeline for materializing semistructured data

**Priority 2: Governance & Tooling**
- Governance framework (policies, change management, agent safety)
- Protégé-like ontology editor with provenance tracking
- Embeddings for semantic search
