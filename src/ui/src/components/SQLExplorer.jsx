import { useState, useEffect } from 'react'
import {
  DatabaseIcon, PlayIcon, TableIcon, CodeIcon, RefreshIcon,
  ChevronDownIcon, CopyIcon, CheckCircleIcon, InfoIcon
} from './Icons'

// API helper
const API_BASE = import.meta.env.DEV ? '/api' : ''

async function fetchJson(endpoint, options = {}) {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options.headers },
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(error.detail || 'Request failed')
  }
  return response.json()
}

// ============================================================================
// Query Helper Card
// ============================================================================
function QueryHelperCard({ helper, onUse }) {
  const categoryColors = {
    basic: 'var(--accent-color)',
    analytics: 'var(--success)',
    schema: 'var(--accent-secondary)',
    provenance: 'var(--warning)',
    quality: 'var(--error)',
    temporal: 'var(--accent-color)',
    entity: 'var(--success)',
    organization: 'var(--accent-secondary)',
    internals: 'var(--text-muted)',
    search: 'var(--accent-color)',
  }

  return (
    <div className="query-helper-card" onClick={() => onUse(helper.sql)}>
      <div className="helper-header">
        <span className="helper-name">{helper.name}</span>
        <span 
          className="helper-category"
          style={{ background: categoryColors[helper.category] || 'var(--bg-elevated)' }}
        >
          {helper.category}
        </span>
      </div>
      <p className="helper-description">{helper.description}</p>
      <code className="helper-preview">{helper.sql.slice(0, 80)}...</code>
    </div>
  )
}

// ============================================================================
// Table Browser Sidebar
// ============================================================================
function TableBrowser({ tables, onSelectTable, selectedTable }) {
  return (
    <div className="table-browser">
      <h3><TableIcon size={16} /> Tables</h3>
      <div className="table-list">
        {tables.map(table => (
          <div
            key={table.name}
            className={`table-item ${selectedTable === table.name ? 'selected' : ''}`}
            onClick={() => onSelectTable(table.name)}
          >
            <span className="table-name">{table.name}</span>
            <span className="table-count">{table.row_count?.toLocaleString() || 0} rows</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ============================================================================
// Schema Panel
// ============================================================================
function SchemaPanel({ table, schema }) {
  if (!table || !schema) return null
  
  return (
    <div className="schema-panel">
      <h4>Schema: {table}</h4>
      <table className="schema-table">
        <thead>
          <tr>
            <th>Column</th>
            <th>Type</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(schema).map(([col, type]) => (
            <tr key={col}>
              <td className="col-name">{col}</td>
              <td className="col-type">{type}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ============================================================================
// Results Table
// ============================================================================
function ResultsTable({ result }) {
  const [copied, setCopied] = useState(false)

  if (!result) return null

  const copyToClipboard = () => {
    const text = result.rows.map(row => row.join('\t')).join('\n')
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="results-panel">
      <div className="results-header">
        <div className="results-info">
          <span className="result-count">{result.row_count} rows</span>
          <span className="result-time">{result.execution_time_ms?.toFixed(1)}ms</span>
        </div>
        <button className="btn secondary small" onClick={copyToClipboard}>
          {copied ? <CheckCircleIcon size={14} /> : <CopyIcon size={14} />}
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      
      <div className="results-table-wrapper">
        <table className="results-table">
          <thead>
            <tr>
              {result.columns.map((col, i) => (
                <th key={i}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.rows.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j} title={typeof cell === 'string' && cell.length > 50 ? cell : undefined}>
                    {cell === null ? <span className="null-value">NULL</span> : String(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ============================================================================
// Main SQL Explorer Component
// ============================================================================
export default function SQLExplorer({ currentRepo }) {
  const [sql, setSql] = useState('SELECT subject, predicate, object, source, confidence \nFROM triples \nLIMIT 20')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [tables, setTables] = useState([])
  const [selectedTable, setSelectedTable] = useState(null)
  const [schema, setSchema] = useState(null)
  const [helpers, setHelpers] = useState([])
  const [showHelpers, setShowHelpers] = useState(true)

  // Load tables and helpers when repo changes
  useEffect(() => {
    if (!currentRepo) {
      setTables([])
      setResult(null)
      return
    }
    
    // Load tables for the current repository
    fetchJson(`/repositories/${currentRepo}/sql/tables`)
      .then(data => setTables(data.tables || []))
      .catch(err => {
        console.error('Failed to load tables:', err)
        setTables([])
      })
    
    // Load query helpers (these are global)
    fetchJson('/sql/helpers')
      .then(data => setHelpers(data.queries || []))
      .catch(err => console.error('Failed to load helpers:', err))
  }, [currentRepo])

  // Load schema when table selected
  useEffect(() => {
    if (selectedTable && currentRepo) {
      fetchJson(`/repositories/${currentRepo}/sql/schema/${selectedTable}`)
        .then(data => setSchema(data.columns))
        .catch(err => console.error('Failed to load schema:', err))
    }
  }, [selectedTable, currentRepo])

  const executeQuery = async () => {
    if (!sql.trim() || !currentRepo) return
    
    setLoading(true)
    setError(null)
    
    try {
      const data = await fetchJson(`/repositories/${currentRepo}/sql/query`, {
        method: 'POST',
        body: JSON.stringify({ sql, limit: 1000 })
      })
      setResult(data)
    } catch (err) {
      setError(err.message)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  const refreshTables = async () => {
    if (!currentRepo) return
    try {
      const data = await fetchJson(`/repositories/${currentRepo}/sql/tables`)
      setTables(data.tables || [])
    } catch (err) {
      setError('Failed to refresh tables: ' + err.message)
    }
  }

  const useHelper = (helperSql) => {
    setSql(helperSql)
    setShowHelpers(false)
  }

  // Show message if no repository selected
  if (!currentRepo) {
    return (
      <div className="sql-explorer">
        <div className="no-repo-message">
          <DatabaseIcon size={48} />
          <h2>No Repository Selected</h2>
          <p>Select a repository from the header dropdown to use the SQL Explorer.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="sql-explorer">
      {/* Left Sidebar - Table Browser */}
      <aside className="sql-sidebar">
        <div className="sidebar-header">
          <h2><DatabaseIcon size={18} /> SQL Explorer</h2>
          <button className="icon-btn" onClick={refreshTables} title="Refresh tables">
            <RefreshIcon size={16} />
          </button>
        </div>
        
        <TableBrowser 
          tables={tables} 
          onSelectTable={setSelectedTable}
          selectedTable={selectedTable}
        />
        
        <SchemaPanel table={selectedTable} schema={schema} />
        
        <div className="sql-info">
          <InfoIcon size={14} />
          <span>DuckDB SQL interface for analytical queries on the triplestore</span>
        </div>
      </aside>

      {/* Main Content */}
      <main className="sql-main">
        {/* Query Editor */}
        <div className="sql-editor-panel">
          <div className="editor-header">
            <h3><CodeIcon size={16} /> SQL Query</h3>
            <div className="editor-actions">
              <button 
                className="btn secondary small"
                onClick={() => setShowHelpers(!showHelpers)}
              >
                <ChevronDownIcon size={14} style={{ transform: showHelpers ? 'rotate(180deg)' : 'none' }} />
                Query Helpers
              </button>
              <button 
                className="btn primary"
                onClick={executeQuery}
                disabled={loading || !sql.trim()}
              >
                <PlayIcon size={16} />
                {loading ? 'Running...' : 'Execute'}
              </button>
            </div>
          </div>
          
          <textarea
            className="sql-editor"
            value={sql}
            onChange={(e) => setSql(e.target.value)}
            placeholder="Enter SQL query..."
            spellCheck={false}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                executeQuery()
              }
            }}
          />
          <div className="editor-hint">
            Press <kbd>Ctrl</kbd>+<kbd>Enter</kbd> to execute
          </div>
        </div>

        {/* Query Helpers Collapsible */}
        {showHelpers && (
          <div className="query-helpers-panel">
            <h3>Query Templates</h3>
            <p className="helpers-intro">
              Click a template to load it into the editor. These queries help you explore claims and provenance in the triplestore.
            </p>
            <div className="helpers-grid">
              {helpers.map((helper, i) => (
                <QueryHelperCard key={i} helper={helper} onUse={useHelper} />
              ))}
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="sql-error">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Results */}
        {result && <ResultsTable result={result} />}
      </main>
    </div>
  )
}
