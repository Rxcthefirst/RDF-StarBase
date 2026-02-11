/**
 * Authentication Context and Provider for RDF-StarBase UI
 * 
 * Provides:
 * - Authentication state management
 * - Session checking
 * - Login/logout functions
 * - CSRF token handling
 * - Protected route wrapper
 */

import { createContext, useContext, useState, useEffect, useCallback } from 'react'

// API base URL
const API_BASE = import.meta.env.DEV ? '/api' : ''

// =============================================================================
// Auth Context
// =============================================================================

const AuthContext = createContext(null)

/**
 * Authentication state shape:
 * {
 *   isAuthenticated: boolean,
 *   isLoading: boolean,
 *   user: { id, username, email, role, groups, allowed_repos } | null,
 *   session: { created_at, expires_at, provider } | null,
 *   error: string | null,
 * }
 */

export function AuthProvider({ children, requireAuth = false, sessionTimeout = 30 * 60 * 1000 }) {
  const [state, setState] = useState({
    isAuthenticated: false,
    // Only show loading state if auth is required
    isLoading: requireAuth,
    user: null,
    session: null,
    error: null,
    requireAuth,
  })
  
  // Get CSRF token from cookie
  const getCsrfToken = useCallback(() => {
    const match = document.cookie.match(/rdfstarbase_csrf=([^;]+)/)
    return match ? match[1] : null
  }, [])
  
  // Check current session with timeout
  const checkSession = useCallback(async () => {
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 second timeout
      
      const response = await fetch(`${API_BASE}/auth/session`, {
        credentials: 'include',
        signal: controller.signal,
      })
      
      clearTimeout(timeoutId)
      
      if (!response.ok) {
        throw new Error('Session check failed')
      }
      
      const data = await response.json()
      
      setState(prev => ({
        ...prev,
        isAuthenticated: data.authenticated,
        isLoading: false,
        user: data.user,
        session: data.session,
        error: null,
      }))
    } catch (error) {
      // Don't block UI on auth check failure when requireAuth is false
      setState(prev => ({
        ...prev,
        isAuthenticated: false,
        isLoading: false,
        user: null,
        session: null,
        error: prev.requireAuth ? error.message : null,
      }))
    }
  }, [])
  
  // Initial session check (non-blocking when requireAuth is false)
  useEffect(() => {
    checkSession()
  }, [checkSession])
  
  // Periodic session refresh (every 5 minutes)
  useEffect(() => {
    if (!state.isAuthenticated) return
    
    const interval = setInterval(checkSession, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [state.isAuthenticated, checkSession])
  
  // Login - redirect to OAuth flow
  const login = useCallback((provider = 'default', redirectTo = window.location.pathname) => {
    const params = new URLSearchParams({
      provider,
      redirect_to: redirectTo,
    })
    window.location.href = `${API_BASE}/auth/login?${params}`
  }, [])
  
  // Logout
  const logout = useCallback(async () => {
    try {
      const csrfToken = getCsrfToken()
      await fetch(`${API_BASE}/auth/logout`, {
        method: 'POST',
        credentials: 'include',
        headers: csrfToken ? { 'X-CSRF-Token': csrfToken } : {},
      })
    } catch (error) {
      console.error('Logout error:', error)
    }
    
    // Clear state regardless
    setState({
      isAuthenticated: false,
      isLoading: false,
      user: null,
      session: null,
      error: null,
    })
    
    // Redirect to home
    window.location.href = '/'
  }, [getCsrfToken])
  
  // Authenticated fetch wrapper (includes CSRF token)
  const authFetch = useCallback(async (url, options = {}) => {
    const csrfToken = getCsrfToken()
    
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers,
    }
    
    // Include CSRF token for state-changing requests
    if (['POST', 'PUT', 'DELETE', 'PATCH'].includes(options.method?.toUpperCase())) {
      if (csrfToken) {
        headers['X-CSRF-Token'] = csrfToken
      }
    }
    
    return fetch(url, {
      ...options,
      credentials: 'include',
      headers,
    })
  }, [getCsrfToken])
  
  // Check if user has required role
  const hasRole = useCallback((requiredRole) => {
    if (!state.user) return false
    
    const roleHierarchy = { reader: 1, writer: 2, admin: 3 }
    const userLevel = roleHierarchy[state.user.role] || 0
    const requiredLevel = roleHierarchy[requiredRole] || 0
    
    return userLevel >= requiredLevel
  }, [state.user])
  
  // Check if user can access repository
  const canAccessRepo = useCallback((repoName) => {
    if (!state.user) return false
    if (!state.user.allowed_repos) return true // No restrictions
    return state.user.allowed_repos.includes(repoName)
  }, [state.user])
  
  const value = {
    ...state,
    requireAuth,
    login,
    logout,
    checkSession,
    authFetch,
    hasRole,
    canAccessRepo,
    getCsrfToken,
  }
  
  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

// =============================================================================
// Auth Hook
// =============================================================================

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

// =============================================================================
// Protected Route Component
// =============================================================================

export function ProtectedRoute({ children, requiredRole = null, fallback = null }) {
  const { isAuthenticated, isLoading, hasRole, login, requireAuth } = useAuth()
  
  // If auth is not required, render children immediately
  if (!requireAuth) {
    return children
  }
  
  // Show loading state (only when requireAuth=true)
  if (isLoading) {
    return fallback || (
      <div className="auth-loading">
        <div className="spinner"></div>
        <p>Checking authentication...</p>
      </div>
    )
  }
  
  // Not authenticated - redirect to login
  if (!isAuthenticated) {
    login('default', window.location.pathname)
    return fallback || (
      <div className="auth-redirect">
        <p>Redirecting to login...</p>
      </div>
    )
  }
  
  // Check role if required
  if (requiredRole && !hasRole(requiredRole)) {
    return (
      <div className="auth-forbidden">
        <h2>Access Denied</h2>
        <p>You don't have permission to view this page.</p>
        <p>Required role: <strong>{requiredRole}</strong></p>
      </div>
    )
  }
  
  return children
}

// =============================================================================
// Login Page Component
// =============================================================================

export function LoginPage({ onSuccess }) {
  const { isAuthenticated, isLoading, login } = useAuth()
  const [providers, setProviders] = useState([])
  const [loadingProviders, setLoadingProviders] = useState(true)
  const [error, setError] = useState(null)
  
  // Fetch available providers
  useEffect(() => {
    async function fetchProviders() {
      try {
        const response = await fetch(`${API_BASE}/auth/providers`)
        if (response.ok) {
          const data = await response.json()
          setProviders(data.providers || [])
        }
      } catch (err) {
        setError('Failed to load authentication providers')
      } finally {
        setLoadingProviders(false)
      }
    }
    fetchProviders()
  }, [])
  
  // If already authenticated, redirect
  useEffect(() => {
    if (isAuthenticated && !isLoading) {
      if (onSuccess) {
        onSuccess()
      } else {
        window.location.href = '/'
      }
    }
  }, [isAuthenticated, isLoading, onSuccess])
  
  if (isLoading || loadingProviders) {
    return (
      <div className="login-page loading">
        <div className="spinner"></div>
        <p>Loading...</p>
      </div>
    )
  }
  
  if (isAuthenticated) {
    return (
      <div className="login-page">
        <p>Already authenticated. Redirecting...</p>
      </div>
    )
  }
  
  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-header">
          <h1>🌟 RDF-StarBase</h1>
          <p>Enterprise Knowledge Graph Platform</p>
        </div>
        
        {error && <div className="login-error">{error}</div>}
        
        <div className="login-providers">
          {providers.length === 0 ? (
            <div className="no-providers">
              <p>No authentication providers configured.</p>
              <p className="hint">Contact your administrator to set up SSO.</p>
            </div>
          ) : (
            <>
              <p>Sign in with your organization:</p>
              {providers.map((provider) => (
                <button
                  key={provider.id}
                  className="provider-btn"
                  onClick={() => login(provider.id)}
                >
                  🔐 Sign in with {provider.name}
                </button>
              ))}
            </>
          )}
        </div>
        
        <div className="login-footer">
          <p>Protected by enterprise SSO</p>
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// Session Timeout Warning Component
// =============================================================================

export function SessionTimeoutWarning({ warningMinutes = 5 }) {
  const { session, checkSession, logout } = useAuth()
  const [showWarning, setShowWarning] = useState(false)
  const [remainingSeconds, setRemainingSeconds] = useState(0)
  
  useEffect(() => {
    if (!session?.expires_at) return
    
    const checkExpiry = () => {
      const expiresAt = new Date(session.expires_at)
      const now = new Date()
      const remaining = Math.floor((expiresAt - now) / 1000)
      
      if (remaining <= 0) {
        logout()
        return
      }
      
      if (remaining <= warningMinutes * 60) {
        setShowWarning(true)
        setRemainingSeconds(remaining)
      } else {
        setShowWarning(false)
      }
    }
    
    checkExpiry()
    const interval = setInterval(checkExpiry, 1000)
    return () => clearInterval(interval)
  }, [session, warningMinutes, logout])
  
  if (!showWarning) return null
  
  const minutes = Math.floor(remainingSeconds / 60)
  const seconds = remainingSeconds % 60
  
  return (
    <div className="session-warning">
      <div className="warning-content">
        <p>⚠️ Your session will expire in {minutes}:{seconds.toString().padStart(2, '0')}</p>
        <button onClick={checkSession} className="btn primary">
          Extend Session
        </button>
        <button onClick={logout} className="btn secondary">
          Logout
        </button>
      </div>
    </div>
  )
}

// =============================================================================
// User Menu Component
// =============================================================================

export function UserMenu() {
  const { isAuthenticated, user, logout } = useAuth()
  const [menuOpen, setMenuOpen] = useState(false)
  
  if (!isAuthenticated || !user) {
    return null
  }
  
  return (
    <div className="user-menu">
      <button 
        className="user-menu-trigger"
        onClick={() => setMenuOpen(!menuOpen)}
      >
        <span className="user-avatar">👤</span>
        <span className="user-name">{user.username}</span>
        <span className="user-role">{user.role}</span>
      </button>
      
      {menuOpen && (
        <div className="user-menu-dropdown">
          <div className="menu-header">
            <strong>{user.username}</strong>
            {user.email && <span className="email">{user.email}</span>}
          </div>
          <div className="menu-items">
            <div className="menu-item info">
              Role: <span className={`role-badge ${user.role}`}>{user.role}</span>
            </div>
            {user.groups?.length > 0 && (
              <div className="menu-item info">
                Groups: {user.groups.join(', ')}
              </div>
            )}
            <hr />
            <button onClick={logout} className="menu-item logout">
              🚪 Sign Out
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default AuthContext
