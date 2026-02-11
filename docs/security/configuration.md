# Enterprise Security Configuration Guide

This guide covers the enterprise authentication and security setup for RDF-StarBase.

## Overview

RDF-StarBase supports enterprise-grade authentication through:
- **OIDC/OAuth2 SSO**: Keycloak, Azure AD, Okta, Auth0
- **API Key Authentication**: For programmatic access
- **Role-Based Access Control (RBAC)**: Reader, Writer, Admin roles

## Quick Start

### 1. Enable Authentication in UI

Copy the environment template and enable auth:

```bash
cd src/ui
cp .env.example .env.local
```

Edit `.env.local`:
```env
VITE_REQUIRE_AUTH=true
VITE_SESSION_TIMEOUT=1800000  # 30 minutes
```

### 2. Configure an OIDC Provider

**Via API:**
```bash
# Configure Keycloak as OIDC provider
curl -X POST http://localhost:8000/security/oidc/providers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "keycloak",
    "display_name": "Corporate SSO",
    "issuer": "https://keycloak.example.com/realms/myrealm",
    "client_id": "rdf-starbase",
    "client_secret": "your-client-secret",
    "redirect_uri": "http://localhost:5173/auth/callback",
    "enabled": true
  }'
```

**Via UI:**
1. Navigate to Security tab
2. Go to OIDC Providers section
3. Click "Add Provider"
4. Fill in provider details

### 3. Configure CORS for Production

Set allowed origins in your deployment:

```bash
# Environment variable
export RDFSTARBASE_CORS_ORIGINS="https://app.example.com,https://admin.example.com"

# Or in production mode (auto-restricts)
export RDFSTARBASE_PRODUCTION=true
```

## OIDC Provider Templates

### Keycloak

```json
{
  "name": "keycloak",
  "display_name": "Keycloak SSO",
  "issuer": "https://keycloak.example.com/realms/{realm}",
  "client_id": "rdf-starbase",
  "authorization_endpoint": "https://keycloak.example.com/realms/{realm}/protocol/openid-connect/auth",
  "token_endpoint": "https://keycloak.example.com/realms/{realm}/protocol/openid-connect/token",
  "userinfo_endpoint": "https://keycloak.example.com/realms/{realm}/protocol/openid-connect/userinfo",
  "jwks_uri": "https://keycloak.example.com/realms/{realm}/protocol/openid-connect/certs",
  "scopes": ["openid", "profile", "email"],
  "claims_mapping": {
    "username_claim": "preferred_username",
    "email_claim": "email",
    "roles_claim": "realm_access.roles"
  }
}
```

### Azure AD

```json
{
  "name": "azure_ad",
  "display_name": "Microsoft Entra ID",
  "issuer": "https://login.microsoftonline.com/{tenant_id}/v2.0",
  "client_id": "your-application-id",
  "authorization_endpoint": "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize",
  "token_endpoint": "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
  "jwks_uri": "https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys",
  "scopes": ["openid", "profile", "email", "User.Read"],
  "claims_mapping": {
    "username_claim": "preferred_username",
    "email_claim": "email",
    "roles_claim": "roles"
  }
}
```

### Okta

```json
{
  "name": "okta",
  "display_name": "Okta SSO",
  "issuer": "https://{domain}.okta.com/oauth2/default",
  "client_id": "your-client-id",
  "authorization_endpoint": "https://{domain}.okta.com/oauth2/default/v1/authorize",
  "token_endpoint": "https://{domain}.okta.com/oauth2/default/v1/token",
  "userinfo_endpoint": "https://{domain}.okta.com/oauth2/default/v1/userinfo",
  "jwks_uri": "https://{domain}.okta.com/oauth2/default/v1/keys",
  "scopes": ["openid", "profile", "email", "groups"],
  "claims_mapping": {
    "username_claim": "preferred_username",
    "email_claim": "email",
    "roles_claim": "groups"
  }
}
```

### Auth0

```json
{
  "name": "auth0",
  "display_name": "Auth0 SSO",
  "issuer": "https://{domain}.auth0.com/",
  "client_id": "your-client-id",
  "authorization_endpoint": "https://{domain}.auth0.com/authorize",
  "token_endpoint": "https://{domain}.auth0.com/oauth/token",
  "userinfo_endpoint": "https://{domain}.auth0.com/userinfo",
  "jwks_uri": "https://{domain}.auth0.com/.well-known/jwks.json",
  "scopes": ["openid", "profile", "email"],
  "claims_mapping": {
    "username_claim": "nickname",
    "email_claim": "email",
    "roles_claim": "https://rdfstarbase/roles"
  }
}
```

## Role Mapping

RDF-StarBase supports three roles:

| Role | Permissions |
|------|-------------|
| `reader` | Query data, view repositories, read-only access |
| `writer` | Create/update triples, import data, manage own resources |
| `admin` | Full access including security configuration, user management |

### Mapping OIDC Groups to Roles

Configure group-to-role mapping:

```json
{
  "role_mapping": {
    "rdf-admins": "admin",
    "rdf-editors": "writer",
    "rdf-viewers": "reader"
  },
  "default_role": "reader"
}
```

## Security Headers

The API automatically sets security headers in production mode:

| Header | Value | Purpose |
|--------|-------|---------|
| `Content-Security-Policy` | `default-src 'self'...` | Prevent XSS |
| `X-Frame-Options` | `DENY` | Prevent clickjacking |
| `X-Content-Type-Options` | `nosniff` | Prevent MIME sniffing |
| `X-XSS-Protection` | `1; mode=block` | Legacy XSS protection |
| `Strict-Transport-Security` | `max-age=31536000...` | Force HTTPS |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | Control referrer |
| `Permissions-Policy` | `geolocation=()...` | Disable unnecessary APIs |

## API Key Authentication

For programmatic/service access:

### Create API Key

```bash
curl -X POST http://localhost:8000/security/keys \
  -H "Authorization: Bearer {admin-token}" \
  -H "Content-Type: application/json" \
  -d '{"name": "CI Pipeline", "role": "writer", "expires_days": 365}'
```

Response:
```json
{
  "key_id": "uuid-here",
  "api_key": "rdfstar_XXXXXXXXXX",  // Only shown once!
  "name": "CI Pipeline",
  "role": "writer",
  "expires_at": "2026-01-01T00:00:00Z"
}
```

### Use API Key

```bash
curl http://localhost:8000/repositories \
  -H "X-API-Key: rdfstar_XXXXXXXXXX"
```

## Session Management

Browser sessions use:
- HTTP-only secure cookies
- CSRF tokens for state-changing operations
- Configurable session timeout
- Automatic session refresh on activity

### Session Timeout Warning

Users see a warning 5 minutes before session expiry with options to:
- Extend session (re-authenticates silently if possible)
- Logout

## Fannie Mae TSP Compliance

For TSP compliance, ensure:

1. **MFA Required**: Configure in your OIDC provider
2. **Session Limits**: Set `VITE_SESSION_TIMEOUT=900000` (15 min)
3. **Audit Logging**: Enable via `RDFSTARBASE_AUDIT_LOG=true`
4. **TLS 1.2+**: Configure at load balancer/reverse proxy
5. **No Local Storage**: Auth tokens stored in HTTP-only cookies

## Deployment Checklist

- [ ] OIDC provider configured and tested
- [ ] CORS origins restricted to known domains
- [ ] TLS/HTTPS enabled at reverse proxy
- [ ] Session timeout configured appropriately
- [ ] Audit logging enabled
- [ ] Security headers verified (use securityheaders.com)
- [ ] API keys created for service accounts
- [ ] Role mapping verified with test users
- [ ] Backup/recovery for OIDC provider config
