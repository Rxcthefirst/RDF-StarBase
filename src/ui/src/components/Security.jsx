import { useState, useEffect } from 'react'

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

/**
 * Security Panel - API Key Management
 * 
 * Allows users to:
 * - View security status
 * - Create new API keys
 * - List existing keys
 * - Revoke keys
 */
export default function Security({ theme }) {
  const [status, setStatus] = useState(null)
  const [keys, setKeys] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [newKey, setNewKey] = useState(null) // Newly created key (shown once)
  
  // Form state
  const [keyName, setKeyName] = useState('')
  const [keyRole, setKeyRole] = useState('reader')
  const [expiresDays, setExpiresDays] = useState('')
  const [rateLimit, setRateLimit] = useState('')
  const [creating, setCreating] = useState(false)

  useEffect(() => {
    loadSecurityData()
  }, [])

  const loadSecurityData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [statusRes, keysRes] = await Promise.all([
        fetchJson('/security/status'),
        fetchJson('/security/keys')
      ])
      setStatus(statusRes)
      setKeys(keysRes)
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
      setKeyName('')
      setExpiresDays('')
      setRateLimit('')
      loadSecurityData()
    } catch (err) {
      setError(err.message)
    } finally {
      setCreating(false)
    }
  }

  const revokeKey = async (keyId) => {
    if (!confirm(`Revoke key ${keyId}? This cannot be undone.`)) return
    
    try {
      await fetchJson(`/security/keys/${keyId}`, { method: 'DELETE' })
      loadSecurityData()
    } catch (err) {
      setError(err.message)
    }
  }

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text)
  }

  if (loading) {
    return <div className="security-panel loading">Loading security configuration...</div>
  }

  return (
    <div className={`security-panel ${theme}`}>
      <div className="security-header">
        <h2>🔐 Security & API Keys</h2>
        <p className="security-status">
          {status?.enabled ? (
            <span className="status-enabled">✓ Security Enabled ({status.key_count} keys)</span>
          ) : (
            <span className="status-disabled">⚠ No API keys configured - endpoints are open</span>
          )}
        </p>
      </div>

      {error && <div className="error-message">{error}</div>}

      {/* Newly created key - show only once */}
      {newKey && (
        <div className="new-key-alert">
          <h3>🔑 New API Key Created</h3>
          <p className="warning">⚠️ Copy this key now! It will not be shown again.</p>
          <div className="key-display">
            <code>{newKey.key}</code>
            <button onClick={() => copyToClipboard(newKey.key)} className="btn-copy">
              📋 Copy
            </button>
          </div>
          <div className="key-details">
            <span>Name: {newKey.name}</span>
            <span>Role: {newKey.role}</span>
            {newKey.expires_at && <span>Expires: {new Date(newKey.expires_at).toLocaleDateString()}</span>}
          </div>
          <button onClick={() => setNewKey(null)} className="btn secondary">
            Dismiss
          </button>
        </div>
      )}

      {/* Create new key form */}
      <div className="create-key-section">
        <h3>Create New API Key</h3>
        <form onSubmit={createKey} className="key-form">
          <div className="form-row">
            <label>
              Name
              <input
                type="text"
                value={keyName}
                onChange={(e) => setKeyName(e.target.value)}
                placeholder="e.g., Production API, Analytics Bot"
                required
              />
            </label>
            <label>
              Role
              <select value={keyRole} onChange={(e) => setKeyRole(e.target.value)}>
                <option value="reader">Reader (read-only)</option>
                <option value="writer">Writer (read/write)</option>
                <option value="admin">Admin (full access)</option>
              </select>
            </label>
          </div>
          <div className="form-row">
            <label>
              Expires (days)
              <input
                type="number"
                value={expiresDays}
                onChange={(e) => setExpiresDays(e.target.value)}
                placeholder="Never"
                min="1"
              />
            </label>
            <label>
              Rate Limit (queries/min)
              <input
                type="number"
                value={rateLimit}
                onChange={(e) => setRateLimit(e.target.value)}
                placeholder="Unlimited"
                min="1"
              />
            </label>
          </div>
          <button type="submit" className="btn primary" disabled={creating || !keyName.trim()}>
            {creating ? 'Creating...' : '+ Create API Key'}
          </button>
        </form>
      </div>

      {/* Existing keys */}
      <div className="keys-list-section">
        <h3>Existing API Keys</h3>
        {keys.length === 0 ? (
          <p className="no-keys">No API keys configured. Create one to enable authentication.</p>
        ) : (
          <table className="keys-table">
            <thead>
              <tr>
                <th>Key ID</th>
                <th>Name</th>
                <th>Role</th>
                <th>Created</th>
                <th>Expires</th>
                <th>Last Used</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {keys.map((key) => (
                <tr key={key.key_id} className={key.is_expired ? 'expired' : ''}>
                  <td><code>{key.key_id}...</code></td>
                  <td>{key.name}</td>
                  <td><span className={`role-badge ${key.role}`}>{key.role}</span></td>
                  <td>{new Date(key.created_at).toLocaleDateString()}</td>
                  <td>{key.expires_at ? new Date(key.expires_at).toLocaleDateString() : 'Never'}</td>
                  <td>{key.last_used ? new Date(key.last_used).toLocaleDateString() : 'Never'}</td>
                  <td>
                    <button 
                      onClick={() => revokeKey(key.key_id)} 
                      className="btn-revoke"
                      title="Revoke this key"
                    >
                      🗑️ Revoke
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Usage instructions */}
      <div className="usage-section">
        <h3>Using API Keys</h3>
        <p>Include your API key in the <code>X-API-Key</code> header:</p>
        <pre className="code-example">
{`curl -H "X-API-Key: YOUR_KEY_HERE" \\
  http://localhost:8000/repositories/myrepo/sparql \\
  -d '{"query": "SELECT * WHERE { ?s ?p ?o } LIMIT 10"}'`}
        </pre>
        <div className="role-descriptions">
          <div className="role-desc">
            <strong>Reader:</strong> Query and export data. Cannot modify.
          </div>
          <div className="role-desc">
            <strong>Writer:</strong> Full data access. Can query, insert, update, delete.
          </div>
          <div className="role-desc">
            <strong>Admin:</strong> Full access including security and system configuration.
          </div>
        </div>
      </div>
    </div>
  )
}
