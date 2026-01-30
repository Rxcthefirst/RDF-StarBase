import { useState, useEffect, useCallback } from 'react'
import { ChevronDownIcon, ChevronRightIcon, LayersIcon, ListIcon, RefreshIcon } from './Icons'

// API base URL - in dev mode with Vite, use proxy; in production, use root
const API_BASE = import.meta.env.DEV ? '/api' : ''

async function runQuery(repoName, query) {
  const response = await fetch(`${API_BASE}/repositories/${repoName}/sparql`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query })
  })
  if (!response.ok) throw new Error('Failed to fetch')
  return response.json()
}

export default function SchemaBrowser({ repositoryName, onInsert, theme }) {
  const [schema, setSchema] = useState({ classes: [], properties: [] })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [expandedSections, setExpandedSections] = useState({ classes: true, properties: true })
  const [searchTerm, setSearchTerm] = useState('')

  const loadSchema = useCallback(async () => {
    if (!repositoryName) return
    
    try {
      setLoading(true)
      setError(null)
      
      // Query for classes (rdf:type usage)
      const classesQuery = `
        SELECT ?class (COUNT(?s) AS ?count)
        WHERE {
          ?s a ?class .
        }
        GROUP BY ?class
        ORDER BY DESC(?count)
        LIMIT 50
      `
      
      // Query for properties
      const propsQuery = `
        SELECT ?prop (COUNT(*) AS ?count)
        WHERE {
          ?s ?prop ?o .
        }
        GROUP BY ?prop
        ORDER BY DESC(?count)
        LIMIT 100
      `
      
      const [classesRes, propsRes] = await Promise.all([
        runQuery(repositoryName, classesQuery),
        runQuery(repositoryName, propsQuery)
      ])
      
      const classes = (classesRes.results || []).map(r => ({
        uri: r.class,
        label: getLocalName(r.class),
        count: r.count
      }))
      
      const properties = (propsRes.results || []).map(r => ({
        uri: r.prop,
        label: getLocalName(r.prop),
        count: r.count
      }))
      
      setSchema({ classes, properties })
    } catch (err) {
      setError('Failed to load schema')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }, [repositoryName])

  useEffect(() => {
    loadSchema()
  }, [loadSchema])

  const getLocalName = (uri) => {
    if (!uri) return uri
    if (uri.includes('#')) return uri.split('#').pop()
    return uri.split('/').pop()
  }

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }))
  }

  const handleInsert = (uri, type) => {
    if (onInsert) {
      // Generate a useful snippet based on type
      if (type === 'class') {
        onInsert(`?s a <${uri}> .`)
      } else {
        onInsert(`?s <${uri}> ?o .`)
      }
    }
  }

  const filteredClasses = schema.classes.filter(c => 
    c.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
    c.uri.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const filteredProperties = schema.properties.filter(p => 
    p.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
    p.uri.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const isDark = theme === 'dark'

  return (
    <div className={`schema-browser ${isDark ? 'dark' : 'light'}`}>
      <div className="schema-header">
        <h3>Schema Browser</h3>
        <button 
          className="icon-btn" 
          onClick={loadSchema} 
          disabled={loading}
          title="Refresh schema"
        >
          <RefreshIcon size={16} />
        </button>
      </div>

      <input
        type="text"
        className="schema-search"
        placeholder="Search classes & properties..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
      />

      {loading && <div className="schema-loading">Loading schema...</div>}
      {error && <div className="schema-error">{error}</div>}

      {!loading && !error && (
        <>
          {/* Classes Section */}
          <div className="schema-section">
            <div 
              className="schema-section-header" 
              onClick={() => toggleSection('classes')}
            >
              {expandedSections.classes ? <ChevronDownIcon size={16} /> : <ChevronRightIcon size={16} />}
              <LayersIcon size={16} />
              <span>Classes ({filteredClasses.length})</span>
            </div>
            {expandedSections.classes && (
              <div className="section-content">
                {filteredClasses.length === 0 ? (
                  <div className="empty-message">No classes found</div>
                ) : (
                  filteredClasses.map((c, i) => (
                    <div 
                      key={i} 
                      className="schema-item"
                      onClick={() => handleInsert(c.uri, 'class')}
                      title={`Click to insert: ?s a <${c.uri}> .`}
                    >
                      <span className="item-label">{c.label}</span>
                      <span className="item-count">{c.count}</span>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>

          {/* Properties Section */}
          <div className="schema-section">
            <div 
              className="schema-section-header" 
              onClick={() => toggleSection('properties')}
            >
              {expandedSections.properties ? <ChevronDownIcon size={16} /> : <ChevronRightIcon size={16} />}
              <ListIcon size={16} />
              <span>Properties ({filteredProperties.length})</span>
            </div>
            {expandedSections.properties && (
              <div className="section-content">
                {filteredProperties.length === 0 ? (
                  <div className="empty-message">No properties found</div>
                ) : (
                  filteredProperties.map((p, i) => (
                    <div 
                      key={i} 
                      className="schema-item"
                      onClick={() => handleInsert(p.uri, 'property')}
                      title={`Click to insert: ?s <${p.uri}> ?o .`}
                    >
                      <span className="item-label">{p.label}</span>
                      <span className="item-count">{p.count}</span>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </>
      )}

      <style>{`
        .schema-browser {
          display: flex;
          flex-direction: column;
          height: 100%;
          overflow: hidden;
        }
        
        .schema-browser.dark {
          background: #1e1e2e;
          color: #cdd6f4;
        }
        
        .schema-browser.light {
          background: #f8f9fa;
          color: #1e1e1e;
        }
        
        .schema-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0.5rem 0.75rem;
          border-bottom: 1px solid var(--border-color);
        }
        
        .schema-header h3 {
          margin: 0;
          font-size: 0.9rem;
          font-weight: 600;
        }
        
        .icon-btn {
          background: none;
          border: none;
          color: inherit;
          cursor: pointer;
          padding: 0.25rem;
          border-radius: 4px;
          opacity: 0.7;
          transition: opacity 0.2s;
        }
        
        .icon-btn:hover {
          opacity: 1;
        }
        
        .icon-btn:disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }
        
        .schema-search {
          margin: 0.4rem 0.5rem;
          padding: 0.4rem 0.6rem;
          border: 1px solid var(--border-color);
          border-radius: 4px;
          font-size: 0.8rem;
          background: var(--input-bg);
          color: inherit;
        }
        
        .schema-browser.dark .schema-search {
          background: #313244;
          border-color: #45475a;
        }
        
        .schema-browser.light .schema-search {
          background: #ffffff;
          border-color: #dee2e6;
        }
        
        .schema-search:focus {
          outline: none;
          border-color: var(--accent-color);
        }
        
        .schema-loading,
        .schema-error {
          padding: 1rem;
          text-align: center;
          font-size: 0.85rem;
        }
        
        .schema-error {
          color: #f38ba8;
        }
        
        .schema-section {
          border-bottom: 1px solid var(--border-color);
          margin: 0;
        }
        
        .schema-section-header {
          display: flex;
          align-items: center;
          gap: 0.4rem;
          padding: 0.4rem 0.6rem;
          cursor: pointer;
          font-size: 0.8rem;
          font-weight: 500;
          user-select: none;
        }
        
        .schema-browser.dark .schema-section-header:hover {
          background: #313244;
        }
        
        .schema-browser.light .schema-section-header:hover {
          background: #e9ecef;
        }
        
        .section-content {
          max-height: 280px;
          overflow-y: auto;
          padding: 0;
        }
        
        .schema-item {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.25rem 0.6rem 0.25rem 1.5rem;
          font-size: 0.78rem;
          cursor: pointer;
          transition: background 0.15s;
        }
        
        .schema-browser.dark .schema-item:hover {
          background: #313244;
        }
        
        .schema-browser.light .schema-item:hover {
          background: #e9ecef;
        }
        
        .item-label {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          max-width: 200px;
        }
        
        .item-count {
          font-size: 0.75rem;
          padding: 0.1rem 0.4rem;
          border-radius: 10px;
          flex-shrink: 0;
        }
        
        .schema-browser.dark .item-count {
          background: #45475a;
          color: #a6adc8;
        }
        
        .schema-browser.light .item-count {
          background: #dee2e6;
          color: #495057;
        }
        
        .empty-message {
          padding: 0.75rem 1rem;
          font-size: 0.8rem;
          opacity: 0.6;
          font-style: italic;
        }
      `}</style>
    </div>
  )
}
