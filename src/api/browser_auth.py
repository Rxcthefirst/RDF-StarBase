"""
Browser-based OAuth2/OIDC Authentication for RDF-StarBase UI.

Provides:
- OAuth2 Authorization Code flow with PKCE
- Secure session management with HttpOnly cookies
- CSRF protection via double-submit cookie pattern
- Session timeout and sliding expiration

This module handles browser-based authentication while api/oidc.py handles
API token validation. Both work together for enterprise SSO integration.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode, urljoin

from fastapi import APIRouter, HTTPException, Request, Response, Depends
from fastapi.responses import RedirectResponse


# Optional dependencies
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None


# =============================================================================
# Session Management
# =============================================================================

@dataclass
class BrowserSession:
    """A browser session for authenticated users."""
    
    session_id: str
    user_id: str
    username: str
    email: str | None
    role: str  # "reader", "writer", "admin"
    provider_id: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    csrf_token: str
    
    # Additional claims from IdP
    groups: list[str] = field(default_factory=list)
    allowed_repos: list[str] | None = None
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["expires_at"] = self.expires_at.isoformat()
        data["last_activity"] = self.last_activity.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrowserSession":
        """Deserialize from storage."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        data["last_activity"] = datetime.fromisoformat(data["last_activity"])
        return cls(**data)


class SessionManager:
    """Manages browser sessions with persistence."""
    
    def __init__(
        self,
        storage_path: Path,
        session_ttl: timedelta = timedelta(hours=1),
        max_sessions_per_user: int = 5,
        sliding_expiration: bool = True,
    ):
        self.storage_path = storage_path
        self.session_ttl = session_ttl
        self.max_sessions_per_user = max_sessions_per_user
        self.sliding_expiration = sliding_expiration
        self.sessions: dict[str, BrowserSession] = {}
        self._load_sessions()
    
    def _load_sessions(self) -> None:
        """Load sessions from disk."""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                for session_data in data:
                    session = BrowserSession.from_dict(session_data)
                    if not session.is_expired():
                        self.sessions[session.session_id] = session
                self._save_sessions()
            except Exception:
                self.sessions = {}
    
    def _save_sessions(self) -> None:
        """Persist sessions to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [s.to_dict() for s in self.sessions.values() if not s.is_expired()]
        self.storage_path.write_text(json.dumps(data, indent=2))
    
    def create_session(
        self,
        user_id: str,
        username: str,
        email: str | None,
        role: str,
        provider_id: str,
        ip_address: str,
        user_agent: str,
        groups: list[str] | None = None,
        allowed_repos: list[str] | None = None,
    ) -> BrowserSession:
        """Create a new browser session."""
        # Enforce max sessions per user
        user_sessions = [
            s for s in self.sessions.values()
            if s.user_id == user_id and not s.is_expired()
        ]
        if len(user_sessions) >= self.max_sessions_per_user:
            # Remove oldest session
            oldest = min(user_sessions, key=lambda s: s.created_at)
            del self.sessions[oldest.session_id]
        
        now = datetime.utcnow()
        session = BrowserSession(
            session_id=secrets.token_urlsafe(32),
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            provider_id=provider_id,
            created_at=now,
            expires_at=now + self.session_ttl,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent,
            csrf_token=secrets.token_urlsafe(32),
            groups=groups or [],
            allowed_repos=allowed_repos,
        )
        
        self.sessions[session.session_id] = session
        self._save_sessions()
        return session
    
    def get_session(self, session_id: str) -> BrowserSession | None:
        """Get and validate a session."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        if session.is_expired():
            del self.sessions[session_id]
            self._save_sessions()
            return None
        
        # Sliding expiration - extend on activity
        if self.sliding_expiration:
            session.last_activity = datetime.utcnow()
            session.expires_at = session.last_activity + self.session_ttl
            self._save_sessions()
        
        return session
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session (logout)."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save_sessions()
            return True
        return False
    
    def invalidate_all_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user (e.g., password change)."""
        to_delete = [
            sid for sid, s in self.sessions.items()
            if s.user_id == user_id
        ]
        for sid in to_delete:
            del self.sessions[sid]
        self._save_sessions()
        return len(to_delete)


# =============================================================================
# OAuth2 PKCE Flow State
# =============================================================================

@dataclass
class AuthorizationState:
    """State for OAuth2 authorization flow."""
    
    state: str  # CSRF protection
    code_verifier: str  # PKCE
    code_challenge: str  # PKCE
    provider_id: str
    redirect_uri: str
    created_at: datetime
    
    def is_expired(self) -> bool:
        """Authorization state expires after 10 minutes."""
        return datetime.utcnow() > self.created_at + timedelta(minutes=10)


class AuthStateManager:
    """Manages OAuth2 authorization flow state."""
    
    def __init__(self):
        self.pending: dict[str, AuthorizationState] = {}
    
    def create_state(
        self,
        provider_id: str,
        redirect_uri: str,
    ) -> AuthorizationState:
        """Create new authorization state with PKCE."""
        state = secrets.token_urlsafe(32)
        code_verifier = secrets.token_urlsafe(64)
        
        # SHA256 hash for PKCE code challenge
        code_challenge = hashlib.sha256(code_verifier.encode()).digest()
        code_challenge_b64 = (
            secrets.base64.urlsafe_b64encode(code_challenge)
            .rstrip(b"=")
            .decode()
        )
        
        auth_state = AuthorizationState(
            state=state,
            code_verifier=code_verifier,
            code_challenge=code_challenge_b64,
            provider_id=provider_id,
            redirect_uri=redirect_uri,
            created_at=datetime.utcnow(),
        )
        
        self.pending[state] = auth_state
        self._cleanup_expired()
        return auth_state
    
    def consume_state(self, state: str) -> AuthorizationState | None:
        """Consume and return state (one-time use)."""
        auth_state = self.pending.pop(state, None)
        if auth_state and not auth_state.is_expired():
            return auth_state
        return None
    
    def _cleanup_expired(self) -> None:
        """Remove expired states."""
        expired = [
            s for s, state in self.pending.items()
            if state.is_expired()
        ]
        for s in expired:
            del self.pending[s]


# =============================================================================
# Browser Auth Router
# =============================================================================

def create_browser_auth_router(
    oidc_manager: Any = None,  # OIDCManager from api/oidc.py, optional
    session_storage_path: Path | None = None,
) -> tuple[APIRouter, SessionManager]:
    """Create router for browser-based OAuth authentication.
    
    Args:
        oidc_manager: Optional OIDC manager for OAuth providers
        session_storage_path: Optional path for session storage
        
    Returns:
        Tuple of (router, session_manager)
    """
    router = APIRouter(prefix="/auth", tags=["authentication"])
    state_manager = AuthStateManager()
    
    # Create session manager
    if session_storage_path is None:
        data_dir = Path(os.getenv("RDFSTARBASE_DATA_DIR", "./data"))
        session_storage_path = data_dir / "security" / "browser_sessions.json"
    session_storage_path.parent.mkdir(parents=True, exist_ok=True)
    session_manager = SessionManager(storage_path=session_storage_path)
    
    # Cookie settings
    COOKIE_SECURE = os.getenv("RDFSTARBASE_PRODUCTION", "false").lower() == "true"
    COOKIE_SAMESITE = "strict"
    COOKIE_PATH = "/"
    SESSION_COOKIE_NAME = "rdfstarbase_session"
    CSRF_COOKIE_NAME = "rdfstarbase_csrf"
    
    def get_client_info(request: Request) -> tuple[str, str]:
        """Extract client IP and user agent."""
        # Handle proxies
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "unknown")
        return ip, user_agent
    
    @router.get("/login")
    async def login(
        request: Request,
        provider: str = "default",
        redirect_to: str = "/",
    ):
        """
        Initiate OAuth2 login flow.
        
        Redirects browser to identity provider's authorization endpoint.
        """
        if not oidc_manager:
            raise HTTPException(503, "OIDC not configured")
        
        # Get provider config
        providers = oidc_manager.list_providers()
        if provider == "default":
            enabled = [p for p in providers if p.enabled]
            if not enabled:
                raise HTTPException(503, "No OIDC providers enabled")
            provider_config = enabled[0]
        else:
            provider_config = oidc_manager.get_provider(provider)
            if not provider_config:
                raise HTTPException(404, f"Provider '{provider}' not found")
        
        # Build callback URL
        base_url = str(request.base_url).rstrip("/")
        callback_url = f"{base_url}/auth/callback"
        
        # Create authorization state with PKCE
        auth_state = state_manager.create_state(
            provider_id=provider_config.provider_id,
            redirect_uri=callback_url,
        )
        
        # Build authorization URL
        # Note: Actual implementation depends on OIDC provider
        auth_params = {
            "response_type": "code",
            "client_id": provider_config.audience,
            "redirect_uri": callback_url,
            "scope": "openid profile email",
            "state": auth_state.state,
            "code_challenge": auth_state.code_challenge,
            "code_challenge_method": "S256",
        }
        
        # Get authorization endpoint from OIDC discovery
        auth_endpoint = f"{provider_config.issuer}/protocol/openid-connect/auth"
        # TODO: Use actual discovery endpoint
        
        auth_url = f"{auth_endpoint}?{urlencode(auth_params)}"
        
        # Store redirect_to in session for post-login
        response = RedirectResponse(auth_url, status_code=302)
        response.set_cookie(
            key="rdfstarbase_redirect",
            value=redirect_to,
            httponly=True,
            secure=COOKIE_SECURE,
            samesite=COOKIE_SAMESITE,
            max_age=600,  # 10 minutes
        )
        return response
    
    @router.get("/callback")
    async def oauth_callback(
        request: Request,
        code: str,
        state: str,
        error: str | None = None,
        error_description: str | None = None,
    ):
        """
        Handle OAuth2 callback from identity provider.
        
        Exchanges authorization code for tokens and creates session.
        """
        if error:
            raise HTTPException(401, f"OAuth error: {error_description or error}")
        
        # Validate state
        auth_state = state_manager.consume_state(state)
        if not auth_state:
            raise HTTPException(400, "Invalid or expired state parameter")
        
        # Get provider config
        provider_config = oidc_manager.get_provider(auth_state.provider_id)
        if not provider_config:
            raise HTTPException(500, "Provider configuration lost")
        
        # Exchange code for tokens (requires httpx)
        if not HAS_HTTPX:
            raise HTTPException(500, "httpx required for OAuth flow")
        
        # Token endpoint
        token_endpoint = f"{provider_config.issuer}/protocol/openid-connect/token"
        # TODO: Use actual discovery endpoint
        
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                token_endpoint,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": auth_state.redirect_uri,
                    "client_id": provider_config.audience,
                    "code_verifier": auth_state.code_verifier,
                },
            )
        
        if token_response.status_code != 200:
            raise HTTPException(401, "Token exchange failed")
        
        tokens = token_response.json()
        id_token = tokens.get("id_token")
        access_token = tokens.get("access_token")
        
        if not id_token:
            raise HTTPException(401, "No ID token received")
        
        # Validate ID token and extract claims
        try:
            validated = oidc_manager.validate_token(
                id_token,
                provider_id=provider_config.provider_id,
            )
        except Exception as e:
            raise HTTPException(401, f"Token validation failed: {e}")
        
        # Create browser session
        ip, user_agent = get_client_info(request)
        session = session_manager.create_session(
            user_id=validated.subject,
            username=validated.username or validated.subject,
            email=validated.claims.get("email"),
            role=validated.role.value,
            provider_id=provider_config.provider_id,
            ip_address=ip,
            user_agent=user_agent,
            groups=validated.claims.get("groups", []),
            allowed_repos=list(validated.allowed_repos) if validated.allowed_repos else None,
        )
        
        # Get redirect destination
        redirect_to = request.cookies.get("rdfstarbase_redirect", "/")
        
        # Build response with secure cookies
        response = RedirectResponse(redirect_to, status_code=302)
        
        # Session cookie (HttpOnly - not accessible to JS)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session.session_id,
            httponly=True,
            secure=COOKIE_SECURE,
            samesite=COOKIE_SAMESITE,
            max_age=int(session_manager.session_ttl.total_seconds()),
            path=COOKIE_PATH,
        )
        
        # CSRF cookie (readable by JS for header inclusion)
        response.set_cookie(
            key=CSRF_COOKIE_NAME,
            value=session.csrf_token,
            httponly=False,  # JS needs to read this
            secure=COOKIE_SECURE,
            samesite=COOKIE_SAMESITE,
            max_age=int(session_manager.session_ttl.total_seconds()),
            path=COOKIE_PATH,
        )
        
        # Clear redirect cookie
        response.delete_cookie("rdfstarbase_redirect")
        
        return response
    
    @router.post("/logout")
    async def logout(request: Request):
        """
        End the current session.
        
        Clears session and cookies. Optionally initiates OIDC RP-initiated logout.
        """
        session_id = request.cookies.get(SESSION_COOKIE_NAME)
        
        if session_id:
            session_manager.invalidate_session(session_id)
        
        response = RedirectResponse("/", status_code=302)
        response.delete_cookie(SESSION_COOKIE_NAME, path=COOKIE_PATH)
        response.delete_cookie(CSRF_COOKIE_NAME, path=COOKIE_PATH)
        return response
    
    @router.get("/session")
    async def get_session_info(request: Request):
        """
        Get current session information.
        
        Used by UI to check authentication state and user info.
        """
        session_id = request.cookies.get(SESSION_COOKIE_NAME)
        
        if not session_id:
            return {
                "authenticated": False,
                "user": None,
            }
        
        session = session_manager.get_session(session_id)
        
        if not session:
            return {
                "authenticated": False,
                "user": None,
            }
        
        return {
            "authenticated": True,
            "user": {
                "id": session.user_id,
                "username": session.username,
                "email": session.email,
                "role": session.role,
                "groups": session.groups,
                "allowed_repos": session.allowed_repos,
            },
            "session": {
                "created_at": session.created_at.isoformat(),
                "expires_at": session.expires_at.isoformat(),
                "provider": session.provider_id,
            },
        }
    
    @router.get("/providers")
    async def list_auth_providers():
        """
        List available authentication providers.
        
        Used by UI to show login options.
        """
        if not oidc_manager:
            return {"providers": []}
        
        try:
            providers = oidc_manager.list_providers()
            return {
                "providers": [
                    {
                        "id": p.provider_id,
                        "name": p.provider_id.replace("-", " ").title(),
                        "enabled": p.enabled,
                    }
                    for p in providers
                    if p.enabled
                ]
            }
        except Exception:
            return {"providers": []}
    
    return router, session_manager


# =============================================================================
# CSRF Protection Dependency
# =============================================================================

async def verify_csrf(request: Request):
    """
    Verify CSRF token for state-changing requests.
    
    Uses double-submit cookie pattern:
    - CSRF token is in cookie (set by server)
    - Same token must be in X-CSRF-Token header (set by JS)
    """
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        cookie_token = request.cookies.get("rdfstarbase_csrf")
        header_token = request.headers.get("X-CSRF-Token")
        
        if not cookie_token or not header_token:
            raise HTTPException(403, "CSRF token missing")
        
        if not secrets.compare_digest(cookie_token, header_token):
            raise HTTPException(403, "CSRF token mismatch")


# =============================================================================
# Session Authentication Dependency
# =============================================================================

async def get_browser_session(
    request: Request,
    session_manager: SessionManager,
) -> BrowserSession | None:
    """
    Get authenticated session from request.
    
    Can be used as a FastAPI dependency.
    """
    session_id = request.cookies.get("rdfstarbase_session")
    if not session_id:
        return None
    return session_manager.get_session(session_id)


async def require_browser_session(
    request: Request,
    session_manager: SessionManager,
) -> BrowserSession:
    """
    Require authenticated session.
    
    Raises 401 if not authenticated.
    """
    session = await get_browser_session(request, session_manager)
    if not session:
        raise HTTPException(401, "Authentication required")
    return session
