import { useState, useEffect, useCallback } from 'react'
import {
  ShieldCheckIcon, ShieldIcon, KeyIcon, PlusIcon, TrashIcon, CopyIcon,
  CheckIcon, AlertTriangleIcon, ClockIcon, SearchIcon, RefreshIcon,
  GlobeIcon, UserCheckIcon, EyeIcon, EyeOffIcon, UsersIcon,
  ShieldOffIcon, LockIcon, CheckCircleIcon, InfoIcon, CloseIcon,
  ToggleLeftIcon, ToggleRightIcon
} from './Icons'

// API base URL
const API_BASE = import.meta.env.DEV ? '/api' : ''

async function fetchJson(endpoint, options = {}) {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(error.detail || 'Request failed')
  }
  return response.json()
}

// ── Toast notification ─────────────────────────────────────────────────────
function Toast({ message, type = 'info', onClose }) {
  useEffect(() => {
    const timer = setTimeout(onClose, 3500)
    return () => clearTimeout(timer)
  }, [onClose])

  const icons = {
    success: <CheckCircleIcon size={16} />,
    error: <AlertTriangleIcon size={16} />,
    info: <InfoIcon size={16} />,
    warning: <AlertTriangleIcon size={16} />,
  }

  return (
    <div className={`sec-toast sec-toast--${type}`}>
      <span className="sec-toast__icon">{icons[type]}</span>
      <span className="sec-toast__msg">{message}</span>
      <button className="sec-toast__close" onClick={onClose}><CloseIcon size={14} /></button>
    </div>
  )
}

// ── Confirmation modal ─────────────────────────────────────────────────────
function ConfirmModal({ title, message, detail, confirmLabel = 'Confirm', danger, onConfirm, onCancel }) {
  return (
    <div className="sec-modal-backdrop" onClick={onCancel}>
      <div className="sec-modal" onClick={e => e.stopPropagation()}>
        <div className={`sec-modal__header ${danger ? 'sec-modal__header--danger' : ''}`}>
          {danger ? <AlertTriangleIcon size={20} /> : <InfoIcon size={20} />}
          <h3>{title}</h3>
        </div>
        <div className="sec-modal__body">
          <p>{message}</p>
          {detail && <p className="sec-modal__detail">{detail}</p>}
        </div>
        <div className="sec-modal__actions">
          <button className="sec-btn sec-btn--ghost" onClick={onCancel}>Cancel</button>
          <button className={`sec-btn ${danger ? 'sec-btn--danger' : 'sec-btn--primary'}`} onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Status card ────────────────────────────────────────────────────────────
function StatusCard({ icon, label, value, variant = 'default' }) {
  return (
    <div className={`sec-stat-card sec-stat-card--${variant}`}>
      <div className="sec-stat-card__icon">{icon}</div>
      <div className="sec-stat-card__content">
        <span className="sec-stat-card__value">{value}</span>
        <span className="sec-stat-card__label">{label}</span>
      </div>
    </div>
  )
}

// ── Relative time helper ───────────────────────────────────────────────────
function relativeTime(isoString) {
  if (!isoString) return 'Never'
  const date = new Date(isoString)
  const now = new Date()
  const diffMs = now - date
  const diffMin = Math.floor(diffMs / 60000)
  const diffHr = Math.floor(diffMin / 60)
  const diffDay = Math.floor(diffHr / 24)
  if (diffMin < 1) return 'Just now'
  if (diffMin < 60) return `${diffMin}m ago`
  if (diffHr < 24) return `${diffHr}h ago`
  if (diffDay < 30) return `${diffDay}d ago`
  return date.toLocaleDateString()
}

function expiryLabel(isoString) {
  if (!isoString) return { text: 'Never expires', warn: false }
  const date = new Date(isoString)
  const now = new Date()
  const diffDay = Math.ceil((date - now) / 86400000)
  if (diffDay < 0) return { text: 'Expired', warn: true }
  if (diffDay === 0) return { text: 'Expires today', warn: true }
  if (diffDay <= 7) return { text: `${diffDay}d remaining`, warn: true }
  if (diffDay <= 30) return { text: `${diffDay}d remaining`, warn: false }
  return { text: date.toLocaleDateString(), warn: false }
}

// ── Main component ─────────────────────────────────────────────────────────
export default function Security({ theme }) {
  const [status, setStatus] = useState(null)
  const [keys, setKeys] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [newKey, setNewKey] = useState(null)
  const [keyRevealed, setKeyRevealed] = useState(false)
  const [toasts, setToasts] = useState([])
  const [confirmModal, setConfirmModal] = useState(null)
  const [searchFilter, setSearchFilter] = useState('')
  const [roleFilter, setRoleFilter] = useState('all')

  // OIDC state
  const [oidcStatus, setOidcStatus] = useState(null)
  const [oidcProviders, setOidcProviders] = useState([])

  // Form state
  const [showForm, setShowForm] = useState(false)
  const [keyName, setKeyName] = useState('')
  const [keyRole, setKeyRole] = useState('reader')
  const [expiresDays, setExpiresDays] = useState('')
  const [rateLimit, setRateLimit] = useState('')
  const [creating, setCreating] = useState(false)

  // Active section
  const [activeSection, setActiveSection] = useState('keys') // 'keys' | 'oidc'

  const toast = useCallback((message, type = 'info') => {
    const id = Date.now()
    setToasts(prev => [...prev, { id, message, type }])
  }, [])

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  useEffect(() => {
    loadSecurityData()
  }, [])

  const loadSecurityData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [statusRes, keysRes] = await Promise.all([
        fetchJson('/security/status'),
        fetchJson('/security/keys'),
      ])
      setStatus(statusRes)
      setKeys(keysRes)

      // Load OIDC status (non-blocking)
      try {
        const [oidcStatusRes, oidcProvidersRes] = await Promise.all([
          fetchJson('/security/oidc/status'),
          fetchJson('/security/oidc/providers').catch(() => []),
        ])
        setOidcStatus(oidcStatusRes)
        setOidcProviders(oidcProvidersRes || [])
      } catch {
        // OIDC not available — that's fine
        setOidcStatus({ available: false })
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const createKey = async (e) => {
    e.preventDefault()
    if (!keyName.trim()) return

    setCreating(true)
    setError(null)
    try {
      const params = new URLSearchParams({ name: keyName, role: keyRole })
      if (expiresDays) params.append('expires_days', expiresDays)
      if (rateLimit) params.append('rate_limit', rateLimit)

      const result = await fetchJson(`/security/keys?${params}`, { method: 'POST' })
      setNewKey(result)
      setKeyRevealed(false)
      setKeyName('')
      setExpiresDays('')
      setRateLimit('')
      setShowForm(false)
      toast('API key created successfully', 'success')
      loadSecurityData()
    } catch (err) {
      toast(err.message, 'error')
    } finally {
      setCreating(false)
    }
  }

  const revokeKey = (key) => {
    setConfirmModal({
      title: 'Revoke API Key',
      message: `Are you sure you want to revoke the key "${key.name}"?`,
      detail: 'This action is permanent and cannot be undone. Any services using this key will immediately lose access.',
      confirmLabel: 'Revoke Key',
      danger: true,
      onConfirm: async () => {
        setConfirmModal(null)
        try {
          await fetchJson(`/security/keys/${key.key_id}`, { method: 'DELETE' })
          toast(`Key "${key.name}" revoked`, 'success')
          loadSecurityData()
        } catch (err) {
          toast(err.message, 'error')
        }
      },
      onCancel: () => setConfirmModal(null),
    })
  }

  const toggleOidcProvider = async (providerId, enabled) => {
    try {
      await fetchJson(`/security/oidc/providers/${providerId}/toggle?enabled=${enabled}`, { method: 'POST' })
      toast(`Provider ${enabled ? 'enabled' : 'disabled'}`, 'success')
      loadSecurityData()
    } catch (err) {
      toast(err.message, 'error')
    }
  }

  const removeOidcProvider = (provider) => {
    setConfirmModal({
      title: 'Remove Identity Provider',
      message: `Remove OIDC provider "${provider.provider_id}"?`,
      detail: 'Users authenticating via this provider will no longer be able to access the system.',
      confirmLabel: 'Remove Provider',
      danger: true,
      onConfirm: async () => {
        setConfirmModal(null)
        try {
          await fetchJson(`/security/oidc/providers/${provider.provider_id}`, { method: 'DELETE' })
          toast('Provider removed', 'success')
          loadSecurityData()
        } catch (err) {
          toast(err.message, 'error')
        }
      },
      onCancel: () => setConfirmModal(null),
    })
  }

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text)
      toast('Copied to clipboard', 'success')
    } catch {
      toast('Failed to copy', 'error')
    }
  }

  // ── Derived data ──────────────────────────────────────────────────────────
  const activeKeys = keys.filter(k => !k.is_expired)
  const expiredKeys = keys.filter(k => k.is_expired)
  const expiringKeys = keys.filter(k => {
    if (!k.expires_at || k.is_expired) return false
    return (new Date(k.expires_at) - new Date()) < 7 * 86400000
  })

  const filteredKeys = keys.filter(k => {
    const matchesSearch = !searchFilter ||
      k.name.toLowerCase().includes(searchFilter.toLowerCase()) ||
      k.key_id.toLowerCase().includes(searchFilter.toLowerCase())
    const matchesRole = roleFilter === 'all' || k.role === roleFilter
    return matchesSearch && matchesRole
  })

  // ── Loading state ─────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="sec-panel">
        <div className="sec-panel__loading">
          <div className="spinner" />
          <p>Loading security configuration...</p>
        </div>
      </div>
    )
  }

  return (
    <div className={`sec-panel ${theme}`}>
      {/* Toast stack */}
      <div className="sec-toast-stack">
        {toasts.map(t => (
          <Toast key={t.id} message={t.message} type={t.type} onClose={() => removeToast(t.id)} />
        ))}
      </div>

      {/* Confirmation modal */}
      {confirmModal && <ConfirmModal {...confirmModal} />}

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="sec-header">
        <div className="sec-header__title">
          <div className="sec-header__icon">
            <ShieldCheckIcon size={24} />
          </div>
          <div>
            <h2>Security Center</h2>
            <p className="sec-header__subtitle">Manage API keys, access control, and identity providers</p>
          </div>
        </div>
        <div className="sec-header__actions">
          <button className="sec-btn sec-btn--ghost" onClick={loadSecurityData} title="Refresh">
            <RefreshIcon size={16} />
          </button>
        </div>
      </div>

      {/* ── Status banner ──────────────────────────────────────────────────── */}
      <div className={`sec-banner ${status?.enabled ? 'sec-banner--secure' : 'sec-banner--warning'}`}>
        <div className="sec-banner__icon">
          {status?.enabled ? <ShieldCheckIcon size={20} /> : <ShieldOffIcon size={20} />}
        </div>
        <div className="sec-banner__content">
          <strong>{status?.enabled ? 'Authentication Enabled' : 'Authentication Disabled'}</strong>
          <span>{status?.enabled
            ? `${activeKeys.length} active key${activeKeys.length !== 1 ? 's' : ''} configured`
            : 'All API endpoints are publicly accessible. Create an API key to enable authentication.'
          }</span>
        </div>
        {!status?.enabled && (
          <button className="sec-btn sec-btn--primary sec-btn--sm" onClick={() => setShowForm(true)}>
            <PlusIcon size={14} /> Create First Key
          </button>
        )}
      </div>

      {/* ── Stats row ──────────────────────────────────────────────────────── */}
      <div className="sec-stats">
        <StatusCard
          icon={<KeyIcon size={20} />}
          label="Active Keys"
          value={activeKeys.length}
          variant={activeKeys.length > 0 ? 'success' : 'muted'}
        />
        <StatusCard
          icon={<ClockIcon size={20} />}
          label="Expiring Soon"
          value={expiringKeys.length}
          variant={expiringKeys.length > 0 ? 'warning' : 'muted'}
        />
        <StatusCard
          icon={<AlertTriangleIcon size={20} />}
          label="Expired"
          value={expiredKeys.length}
          variant={expiredKeys.length > 0 ? 'danger' : 'muted'}
        />
        <StatusCard
          icon={<GlobeIcon size={20} />}
          label="Identity Providers"
          value={oidcStatus?.available ? (oidcStatus.enabled_count ?? 0) : 'N/A'}
          variant={oidcStatus?.available && oidcStatus.enabled_count > 0 ? 'info' : 'muted'}
        />
      </div>

      {/* ── Section tabs ───────────────────────────────────────────────────── */}
      <div className="sec-tabs">
        <button
          className={`sec-tabs__btn ${activeSection === 'keys' ? 'sec-tabs__btn--active' : ''}`}
          onClick={() => setActiveSection('keys')}
        >
          <KeyIcon size={16} /> API Keys
          <span className="sec-tabs__count">{keys.length}</span>
        </button>
        <button
          className={`sec-tabs__btn ${activeSection === 'oidc' ? 'sec-tabs__btn--active' : ''}`}
          onClick={() => setActiveSection('oidc')}
        >
          <GlobeIcon size={16} /> Identity Providers
          {oidcStatus?.available && <span className="sec-tabs__count">{oidcStatus.provider_count ?? 0}</span>}
        </button>
      </div>

      {/* ── New key alert ──────────────────────────────────────────────────── */}
      {newKey && (
        <div className="sec-new-key">
          <div className="sec-new-key__header">
            <KeyIcon size={18} />
            <h3>API Key Created</h3>
            <button className="sec-btn sec-btn--ghost sec-btn--sm" onClick={() => setNewKey(null)}>
              <CloseIcon size={16} />
            </button>
          </div>
          <div className="sec-new-key__warning">
            <AlertTriangleIcon size={14} />
            <span>Store this key in a secure location. It will not be displayed again.</span>
          </div>
          <div className="sec-new-key__display">
            <code className={keyRevealed ? '' : 'sec-new-key--masked'}>
              {keyRevealed ? newKey.key : '\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022'}
            </code>
            <button className="sec-btn sec-btn--ghost sec-btn--sm" onClick={() => setKeyRevealed(!keyRevealed)} title={keyRevealed ? 'Hide' : 'Reveal'}>
              {keyRevealed ? <EyeOffIcon size={16} /> : <EyeIcon size={16} />}
            </button>
            <button className="sec-btn sec-btn--ghost sec-btn--sm" onClick={() => copyToClipboard(newKey.key)} title="Copy to clipboard">
              <CopyIcon size={16} />
            </button>
          </div>
          <div className="sec-new-key__meta">
            <span><strong>Name:</strong> {newKey.name}</span>
            <span><strong>Role:</strong> <span className={`sec-role sec-role--${newKey.role}`}>{newKey.role}</span></span>
            {newKey.expires_at && <span><strong>Expires:</strong> {new Date(newKey.expires_at).toLocaleDateString()}</span>}
          </div>
        </div>
      )}

      {/* ── API Keys Section ───────────────────────────────────────────────── */}
      {activeSection === 'keys' && (
        <div className="sec-section">

          {/* Toolbar */}
          <div className="sec-toolbar">
            <div className="sec-toolbar__search">
              <SearchIcon size={16} />
              <input
                type="text"
                placeholder="Search keys..."
                value={searchFilter}
                onChange={e => setSearchFilter(e.target.value)}
              />
            </div>
            <div className="sec-toolbar__filters">
              <select value={roleFilter} onChange={e => setRoleFilter(e.target.value)}>
                <option value="all">All Roles</option>
                <option value="reader">Reader</option>
                <option value="writer">Writer</option>
                <option value="admin">Admin</option>
              </select>
            </div>
            <button className="sec-btn sec-btn--primary" onClick={() => setShowForm(!showForm)}>
              <PlusIcon size={16} /> {showForm ? 'Cancel' : 'Create Key'}
            </button>
          </div>

          {/* Create key form (collapsible) */}
          {showForm && (
            <div className="sec-create-form">
              <div className="sec-create-form__header">
                <KeyIcon size={18} />
                <h3>Create New API Key</h3>
              </div>
              <form onSubmit={createKey}>
                <div className="sec-form-grid">
                  <label className="sec-field">
                    <span className="sec-field__label">Key Name <span className="sec-field__required">*</span></span>
                    <input
                      type="text"
                      value={keyName}
                      onChange={e => setKeyName(e.target.value)}
                      placeholder="e.g., Production API, CI/CD Pipeline"
                      required
                      autoFocus
                    />
                    <span className="sec-field__hint">A descriptive name to identify this key</span>
                  </label>
                  <label className="sec-field">
                    <span className="sec-field__label">Access Role</span>
                    <select value={keyRole} onChange={e => setKeyRole(e.target.value)}>
                      <option value="reader">Reader — Query &amp; export only</option>
                      <option value="writer">Writer — Read, insert, update, delete</option>
                      <option value="admin">Admin — Full system access</option>
                    </select>
                  </label>
                  <label className="sec-field">
                    <span className="sec-field__label">Expiration</span>
                    <input
                      type="number"
                      value={expiresDays}
                      onChange={e => setExpiresDays(e.target.value)}
                      placeholder="No expiration"
                      min="1"
                    />
                    <span className="sec-field__hint">Days until key expires (blank = never)</span>
                  </label>
                  <label className="sec-field">
                    <span className="sec-field__label">Rate Limit</span>
                    <input
                      type="number"
                      value={rateLimit}
                      onChange={e => setRateLimit(e.target.value)}
                      placeholder="Unlimited"
                      min="1"
                    />
                    <span className="sec-field__hint">Maximum queries per minute</span>
                  </label>
                </div>
                <div className="sec-create-form__actions">
                  <button type="button" className="sec-btn sec-btn--ghost" onClick={() => setShowForm(false)}>Cancel</button>
                  <button type="submit" className="sec-btn sec-btn--primary" disabled={creating || !keyName.trim()}>
                    {creating ? (
                      <><div className="sec-spinner" /> Creating...</>
                    ) : (
                      <><KeyIcon size={16} /> Create API Key</>
                    )}
                  </button>
                </div>
              </form>
            </div>
          )}

          {/* Keys table */}
          {filteredKeys.length === 0 ? (
            <div className="sec-empty">
              <KeyIcon size={40} />
              <h3>{keys.length === 0 ? 'No API keys yet' : 'No matching keys'}</h3>
              <p>{keys.length === 0
                ? 'Create your first API key to enable authenticated access to the platform.'
                : 'Try adjusting your search or filter criteria.'
              }</p>
              {keys.length === 0 && (
                <button className="sec-btn sec-btn--primary" onClick={() => setShowForm(true)}>
                  <PlusIcon size={16} /> Create API Key
                </button>
              )}
            </div>
          ) : (
            <div className="sec-table-wrap">
              <table className="sec-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Key ID</th>
                    <th>Role</th>
                    <th>Created</th>
                    <th>Expiration</th>
                    <th>Last Used</th>
                    <th style={{ width: '1%' }}></th>
                  </tr>
                </thead>
                <tbody>
                  {filteredKeys.map(key => {
                    const expiry = expiryLabel(key.expires_at)
                    return (
                      <tr key={key.key_id} className={key.is_expired ? 'sec-table__row--expired' : ''}>
                        <td>
                          <div className="sec-key-name">
                            <KeyIcon size={14} />
                            <span>{key.name}</span>
                          </div>
                        </td>
                        <td>
                          <code className="sec-key-id">{key.key_id.substring(0, 12)}...</code>
                          <button className="sec-btn-icon" onClick={() => copyToClipboard(key.key_id)} title="Copy Key ID">
                            <CopyIcon size={13} />
                          </button>
                        </td>
                        <td><span className={`sec-role sec-role--${key.role}`}>{key.role}</span></td>
                        <td><span className="sec-table__date" title={new Date(key.created_at).toLocaleString()}>{relativeTime(key.created_at)}</span></td>
                        <td>
                          <span className={`sec-table__expiry ${expiry.warn ? 'sec-table__expiry--warn' : ''} ${key.is_expired ? 'sec-table__expiry--expired' : ''}`}>
                            {key.is_expired && <AlertTriangleIcon size={12} />}
                            {expiry.text}
                          </span>
                        </td>
                        <td><span className="sec-table__date">{relativeTime(key.last_used)}</span></td>
                        <td>
                          <button
                            className="sec-btn sec-btn--danger-ghost sec-btn--sm"
                            onClick={() => revokeKey(key)}
                            title="Revoke this key"
                          >
                            <TrashIcon size={14} /> Revoke
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Usage guide */}
          <div className="sec-guide">
            <div className="sec-guide__header">
              <InfoIcon size={16} />
              <h4>Authentication Guide</h4>
            </div>
            <div className="sec-guide__content">
              <div className="sec-guide__example">
                <span className="sec-guide__label">HTTP Header</span>
                <div className="sec-guide__code">
                  <code>curl -H "X-API-Key: YOUR_KEY" https://your-instance/repositories/myrepo/sparql</code>
                  <button className="sec-btn-icon" onClick={() => copyToClipboard('curl -H "X-API-Key: YOUR_KEY" https://your-instance/repositories/myrepo/sparql')} title="Copy">
                    <CopyIcon size={13} />
                  </button>
                </div>
              </div>
              <div className="sec-guide__roles">
                <div className="sec-guide__role">
                  <span className="sec-role sec-role--reader">reader</span>
                  <span>Query &amp; export data. Cannot modify repositories or configuration.</span>
                </div>
                <div className="sec-guide__role">
                  <span className="sec-role sec-role--writer">writer</span>
                  <span>Full data access. Query, insert, update, delete, bulk-load.</span>
                </div>
                <div className="sec-guide__role">
                  <span className="sec-role sec-role--admin">admin</span>
                  <span>Unrestricted access including security settings and system configuration.</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── OIDC Section ───────────────────────────────────────────────────── */}
      {activeSection === 'oidc' && (
        <div className="sec-section">
          {!oidcStatus?.available ? (
            <div className="sec-empty">
              <GlobeIcon size={40} />
              <h3>OIDC / SSO Not Available</h3>
              <p>Single Sign-On support requires the <code>auth</code> extras package.</p>
              <div className="sec-guide__code" style={{ marginTop: '1rem' }}>
                <code>pip install "rdf-starbase[auth]"</code>
                <button className="sec-btn-icon" onClick={() => copyToClipboard('pip install "rdf-starbase[auth]"')} title="Copy">
                  <CopyIcon size={13} />
                </button>
              </div>
            </div>
          ) : (
            <>
              <div className="sec-oidc-grid">
                {oidcProviders.length === 0 ? (
                  <div className="sec-empty">
                    <UsersIcon size={40} />
                    <h3>No Identity Providers</h3>
                    <p>Connect an OIDC-compatible identity provider to enable SSO authentication.</p>
                    <p className="sec-empty__hint">
                      Supported: Keycloak, Azure AD, Okta, Auth0, and any OIDC-compliant provider.
                    </p>
                  </div>
                ) : (
                  oidcProviders.map(p => (
                    <div key={p.provider_id} className={`sec-provider-card ${!p.enabled ? 'sec-provider-card--disabled' : ''}`}>
                      <div className="sec-provider-card__header">
                        <GlobeIcon size={18} />
                        <div className="sec-provider-card__name">
                          <strong>{p.provider_id}</strong>
                          <span className="sec-provider-card__issuer">{p.issuer}</span>
                        </div>
                        <div className="sec-provider-card__actions">
                          <button
                            className="sec-btn-icon"
                            onClick={() => toggleOidcProvider(p.provider_id, !p.enabled)}
                            title={p.enabled ? 'Disable' : 'Enable'}
                          >
                            {p.enabled ? <ToggleRightIcon size={22} /> : <ToggleLeftIcon size={22} />}
                          </button>
                          <button
                            className="sec-btn-icon sec-btn-icon--danger"
                            onClick={() => removeOidcProvider(p)}
                            title="Remove provider"
                          >
                            <TrashIcon size={14} />
                          </button>
                        </div>
                      </div>
                      <div className="sec-provider-card__meta">
                        <span className={`sec-provider-card__status ${p.enabled ? 'sec-provider-card__status--active' : ''}`}>
                          {p.enabled ? <><CheckCircleIcon size={12} /> Active</> : <><ShieldOffIcon size={12} /> Disabled</>}
                        </span>
                        {p.role_claim && <span>Role claim: <code>{p.role_claim}</code></span>}
                      </div>
                    </div>
                  ))
                )}
              </div>

              {/* OIDC setup guide */}
              <div className="sec-guide" style={{ marginTop: '1.5rem' }}>
                <div className="sec-guide__header">
                  <InfoIcon size={16} />
                  <h4>Connecting an Identity Provider</h4>
                </div>
                <div className="sec-guide__content">
                  <p className="sec-guide__text">Use the REST API to register OIDC providers. Pre-built configurations are available for major platforms:</p>
                  <div className="sec-guide__providers">
                    {[
                      { name: 'Keycloak', endpoint: '/security/oidc/providers/keycloak' },
                      { name: 'Azure AD', endpoint: '/security/oidc/providers/azure' },
                      { name: 'Okta', endpoint: '/security/oidc/providers/okta' },
                      { name: 'Auth0', endpoint: '/security/oidc/providers/auth0' },
                    ].map(p => (
                      <div key={p.name} className="sec-guide__provider-chip">
                        <GlobeIcon size={14} />
                        <span>{p.name}</span>
                        <code>{p.endpoint}</code>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* ── Error display ──────────────────────────────────────────────────── */}
      {error && (
        <div className="sec-error">
          <AlertTriangleIcon size={16} />
          <span>{error}</span>
          <button className="sec-btn-icon" onClick={() => setError(null)}><CloseIcon size={14} /></button>
        </div>
      )}
    </div>
  )
}
