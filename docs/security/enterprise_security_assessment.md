# Enterprise Security Assessment

> RDF-StarBase UI Security Evaluation for Fannie Mae TSP Compliance

## Executive Summary

**Current Status: 🟡 DEVELOPMENT-READY, NOT ENTERPRISE-READY**

The backend has strong foundations (API keys, OIDC/OAuth support, RBAC), but the **UI lacks browser-based authentication flows** required for enterprise SSO integration. Critical gaps exist in security headers, session management, and CSRF protection.

### TSP Compliance Score

| Category | Current | Required | Gap |
|----------|---------|----------|-----|
| **Authentication** | 50% | 100% | Missing browser OAuth flow |
| **Authorization** | 80% | 100% | Need fine-grained claims mapping |
| **Session Management** | 0% | 100% | No browser sessions |
| **Security Headers** | 20% | 100% | Missing CSP, HSTS, etc. |
| **Audit/Logging** | 60% | 100% | Need security event logging |
| **CSRF Protection** | 0% | 100% | Not implemented |

---

## 1. Current Architecture Analysis

### 1.1 What Exists Today

**Backend (`src/api/`):**
```
✅ API Key authentication (256-bit tokens, SHA-256 hashed)
✅ OIDC/OAuth2 JWT validation
✅ Role-based access control (READER/WRITER/ADMIN)
✅ Operation scoping (per-key permissions)
✅ Rate limiting (per-key)
✅ Provider templates (Keycloak, Azure AD, Okta, Auth0)
✅ JWKS caching and auto-discovery
```

**UI (`src/ui/`):**
```
❌ No login page
❌ No OAuth redirect flow
❌ No session cookies
❌ No CSRF tokens
❌ No authenticated state management
❌ Hardcoded "open access" - anyone can use UI
```

### 1.2 The Critical Gap

The UI currently operates **without authentication**. It can:
1. Query any repository
2. Create API keys (requires no auth!)
3. View security configuration
4. Execute arbitrary SPARQL

This is suitable for local development but **fails every enterprise security standard**.

---

## 2. Fannie Mae TSP Requirements Mapping

### 2.1 Authentication & Identity (TSP-AUTH)

| Requirement | Status | Implementation Path |
|-------------|--------|---------------------|
| **SSO Integration** | ⚠️ Backend ready | Need UI OAuth flow |
| **MFA Support** | ⚠️ IdP handles | Verify claims propagate |
| **Session Timeout** | ❌ Missing | Implement with cookies |
| **Account Lockout** | ❌ Missing | Add failed attempt tracking |
| **Password Policy** | N/A | Delegated to IdP |

### 2.2 Authorization (TSP-AUTHZ)

| Requirement | Status | Implementation Path |
|-------------|--------|---------------------|
| **RBAC** | ✅ Done | Roles: READER/WRITER/ADMIN |
| **Least Privilege** | ✅ Done | Scoped tokens, repo limits |
| **Fine-grained Access** | ⚠️ Partial | Add graph-level permissions |
| **Dynamic Authorization** | ❌ Missing | Add context-aware policies |

### 2.3 Data Protection (TSP-DATA)

| Requirement | Status | Implementation Path |
|-------------|--------|---------------------|
| **Encryption at Rest** | ⚠️ Infra | Document volume encryption |
| **Encryption in Transit** | ⚠️ Infra | Enforce TLS, add HSTS |
| **Key Management** | ✅ Done | SHA-256 hashed API keys |
| **Data Classification** | ❌ Missing | Add sensitivity labels |

### 2.4 Logging & Monitoring (TSP-LOG)

| Requirement | Status | Implementation Path |
|-------------|--------|---------------------|
| **Audit Trail** | ✅ Partial | Extend to security events |
| **Security Events** | ❌ Missing | Login, failures, revocations |
| **Log Integrity** | ❌ Missing | Add tamper detection |
| **Retention Policy** | ✅ Done | Configurable rotation |

---

## 3. Required Security Headers

### 3.1 Current State

```python
# src/api/web.py line 519-525
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # ❌ CRITICAL: Wide open
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Missing headers:**
- Content-Security-Policy
- Strict-Transport-Security
- X-Frame-Options
- X-Content-Type-Options
- X-XSS-Protection
- Referrer-Policy
- Permissions-Policy

### 3.2 Required Implementation

```python
@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    
    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    
    # Prevent MIME sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    # Force HTTPS (enable in production)
    if PRODUCTION:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # CSP - restrict script/style sources
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self' https://*.auth0.com https://*.okta.com; "
        "frame-ancestors 'none'; "
    )
    
    # Modern XSS protection
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    # Referrer policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response
```

---

## 4. UI Authentication Flow Design

### 4.1 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Browser Authentication Flow                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   User → UI (/login) → Redirect to IdP (Okta/Azure AD/Keycloak)         │
│                              ↓                                          │
│   IdP authenticates → Redirect to /callback with auth code               │
│                              ↓                                          │
│   Backend exchanges code → Gets tokens → Creates session cookie          │
│                              ↓                                          │
│   UI loads with authenticated context → API calls include session        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

Session Cookie Properties:
  - HttpOnly: true (no JS access)
  - Secure: true (HTTPS only)
  - SameSite: Strict (CSRF protection)
  - Max-Age: 3600 (1 hour, sliding)
```

### 4.2 New UI Components Required

```
src/ui/src/
├── components/
│   ├── Auth/
│   │   ├── LoginPage.jsx       # SSO redirect trigger
│   │   ├── CallbackPage.jsx    # Handle OAuth callback
│   │   ├── LogoutButton.jsx    # End session
│   │   ├── AuthContext.jsx     # React context for auth state
│   │   ├── ProtectedRoute.jsx  # Route guard component
│   │   └── SessionTimeout.jsx  # Inactivity warning
│   └── ...
├── hooks/
│   └── useAuth.js              # Auth state hook
└── App.jsx                     # Wrap with AuthProvider
```

### 4.3 Backend Endpoints Required

```python
# New endpoints for browser OAuth flow
@router.get("/auth/login")
async def login_redirect(provider: str = "default"):
    """Redirect to IdP authorization endpoint."""
    # Build authorization URL with state, nonce
    # Return redirect to IdP
    
@router.get("/auth/callback")
async def oauth_callback(code: str, state: str):
    """Handle OAuth callback, exchange code for tokens."""
    # Validate state
    # Exchange code for tokens
    # Validate ID token
    # Create session
    # Set HttpOnly cookie
    # Redirect to UI
    
@router.post("/auth/logout")
async def logout():
    """End session and clear cookies."""
    # Invalidate session
    # Clear cookies
    # Optionally logout from IdP (OIDC RP-initiated logout)
    
@router.get("/auth/session")
async def get_session():
    """Get current session info (for UI state)."""
    # Return user info, role, expiry
```

---

## 5. CSRF Protection Strategy

### 5.1 Double-Submit Cookie Pattern

For state-changing operations when using session cookies:

```python
from fastapi import Request, Depends
import secrets

async def csrf_protect(request: Request):
    """Verify CSRF token for POST/PUT/DELETE."""
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        cookie_token = request.cookies.get("csrf_token")
        header_token = request.headers.get("X-CSRF-Token")
        
        if not cookie_token or not header_token:
            raise HTTPException(403, "CSRF token missing")
        if not secrets.compare_digest(cookie_token, header_token):
            raise HTTPException(403, "CSRF token mismatch")
```

### 5.2 UI Implementation

```jsx
// On app load, get CSRF token from cookie
const csrfToken = document.cookie
  .split('; ')
  .find(row => row.startsWith('csrf_token='))
  ?.split('=')[1];

// Include in all mutating requests
fetch('/api/repositories', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-CSRF-Token': csrfToken,
  },
  credentials: 'include',
  body: JSON.stringify(data),
});
```

---

## 6. Session Management Implementation

### 6.1 Server-Side Sessions

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
import secrets

@dataclass
class Session:
    session_id: str
    user_id: str
    username: str
    role: Role
    provider: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str

class SessionManager:
    def __init__(self, storage_path: Path):
        self.sessions: dict[str, Session] = {}
        self.storage = storage_path
        
    def create_session(
        self,
        user_id: str,
        username: str,
        role: Role,
        provider: str,
        ip_address: str,
        user_agent: str,
        ttl: timedelta = timedelta(hours=1),
    ) -> str:
        session_id = secrets.token_urlsafe(32)
        now = datetime.utcnow()
        
        self.sessions[session_id] = Session(
            session_id=session_id,
            user_id=user_id,
            username=username,
            role=role,
            provider=provider,
            created_at=now,
            expires_at=now + ttl,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        
        return session_id
    
    def get_session(self, session_id: str) -> Session | None:
        session = self.sessions.get(session_id)
        if not session:
            return None
        if datetime.utcnow() > session.expires_at:
            del self.sessions[session_id]
            return None
        # Sliding expiration
        session.last_activity = datetime.utcnow()
        return session
    
    def invalidate_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
```

### 6.2 Cookie Configuration

```python
response.set_cookie(
    key="session_id",
    value=session_id,
    httponly=True,      # No JavaScript access
    secure=True,        # HTTPS only
    samesite="strict",  # CSRF protection
    max_age=3600,       # 1 hour
    path="/",
    domain=None,        # Current domain only
)

response.set_cookie(
    key="csrf_token",
    value=csrf_token,
    httponly=False,     # JS needs to read this
    secure=True,
    samesite="strict",
    max_age=3600,
)
```

---

## 7. Security Event Logging

### 7.1 Events to Capture

```python
class SecurityEventType(Enum):
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    SESSION_EXPIRED = "session_expired"
    SESSION_INVALID = "session_invalid"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

@dataclass
class SecurityEvent:
    timestamp: datetime
    event_type: SecurityEventType
    user_id: str | None
    username: str | None
    ip_address: str
    user_agent: str
    resource: str | None
    details: dict[str, Any]
    severity: str  # INFO, WARNING, CRITICAL
```

### 7.2 Integration with SIEM

```python
class SecurityLogger:
    def log_event(self, event: SecurityEvent):
        # Local file logging
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event.__dict__, default=str) + "\n")
        
        # SIEM webhook (Splunk, Azure Sentinel, etc.)
        if self.siem_webhook:
            httpx.post(self.siem_webhook, json=event.__dict__)
```

---

## 8. Implementation Roadmap

### Phase 1: Security Headers (1 day)
- [ ] Add security headers middleware
- [ ] Configure strict CORS for production
- [ ] Add CSP with proper directives
- [ ] Enable HSTS

### Phase 2: Browser OAuth Flow (3-5 days)
- [ ] Backend: `/auth/login`, `/auth/callback`, `/auth/logout`
- [ ] Backend: Session manager with Redis/file storage
- [ ] UI: LoginPage with provider selection
- [ ] UI: CallbackPage for OAuth redirect handling
- [ ] UI: AuthContext and ProtectedRoute components
- [ ] UI: Session timeout warning

### Phase 3: CSRF Protection (1 day)
- [ ] Double-submit cookie implementation
- [ ] UI: Include CSRF token in all mutating requests
- [ ] Tests for CSRF protection

### Phase 4: Security Events (2 days)
- [ ] SecurityLogger class
- [ ] Hook into auth/session events
- [ ] SIEM export endpoint
- [ ] Dashboard for security overview

### Phase 5: Enterprise Hardening (2-3 days)
- [ ] Account lockout after N failed attempts
- [ ] Session binding (IP + User-Agent)
- [ ] Concurrent session limits
- [ ] Forced logout on password change (from IdP)
- [ ] Graph-level permissions

---

## 9. Configuration Example

### Production `config.yaml`

```yaml
security:
  # Require authentication for all endpoints
  require_auth: true
  
  # Session settings
  session:
    ttl_minutes: 60
    sliding_expiration: true
    max_concurrent_sessions: 5
    bind_to_ip: true
    
  # CORS - restrict to known origins
  cors:
    allowed_origins:
      - "https://rdfstarbase.example.com"
      - "https://admin.example.com"
    allow_credentials: true
    
  # Rate limiting
  rate_limits:
    login_attempts: 5/minute
    api_requests: 1000/minute
    
  # Account lockout
  lockout:
    max_failed_attempts: 5
    lockout_duration_minutes: 30
    
  # OIDC providers
  oidc:
    default_provider: okta
    providers:
      okta:
        issuer: https://example.okta.com
        client_id: ${OKTA_CLIENT_ID}
        client_secret: ${OKTA_CLIENT_SECRET}
        scopes: ["openid", "profile", "email", "groups"]
        
  # Audit settings
  audit:
    security_events: true
    siem_webhook: https://splunk.example.com/hec
    retention_days: 365
```

---

## 10. Acceptance Criteria for Enterprise Ready

### Authentication
- [ ] Users cannot access UI without SSO login
- [ ] Session cookies are HttpOnly, Secure, SameSite=Strict
- [ ] Sessions expire after inactivity
- [ ] Logout terminates session completely
- [ ] Multiple IdP support (Okta, Azure AD, Keycloak)

### Authorization
- [ ] UI respects user role (hide admin features for readers)
- [ ] API enforces permissions per endpoint
- [ ] Scoped access to specific repositories

### Security Headers
- [ ] CSP prevents XSS
- [ ] X-Frame-Options prevents clickjacking
- [ ] HSTS enforces TLS

### Audit
- [ ] All logins/logouts logged with IP, timestamp, user
- [ ] Failed attempts tracked
- [ ] Security events exportable to SIEM

### Testing
- [ ] Penetration test passed
- [ ] OWASP Top 10 mitigations verified
- [ ] Automated security scanning in CI

---

## Conclusion

RDF-StarBase has solid **backend security primitives** but requires significant **UI-layer work** to meet enterprise standards. The key investment is implementing browser-based OAuth/OIDC flow with proper session management.

**Estimated effort:** 2-3 weeks for full enterprise compliance
**Recommended approach:** Start with Phase 1-2 (security headers + OAuth flow) to unlock enterprise pilots

