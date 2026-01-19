import { useState, useEffect, useCallback, useRef } from 'react'
import * as d3 from 'd3'
import SparqlEditor from './components/SparqlEditor'
import SchemaBrowser from './components/SchemaBrowser'
import ImportExport from './components/ImportExport'
import {
  DatabaseIcon, PlayIcon, PlusIcon, TrashIcon, FolderIcon,
  TableIcon, NetworkIcon, CodeIcon, SunIcon, MoonIcon,
  SearchIcon, SettingsIcon, BookIcon, ZapIcon, GlobeIcon,
  ChevronDownIcon, CloseIcon, RefreshIcon
} from './components/Icons'
import './index.css'

// API base URL - in dev mode with Vite, use proxy; in production, use root
const API_BASE = import.meta.env.DEV ? '/api' : ''

// Fetch helpers
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

const getLocalName = (uri) => {
  if (!uri) return uri
  if (uri.includes('#')) return uri.split('#').pop()
  return uri.split('/').pop()
}

// ============================================================================
// Graph Visualization Component
// ============================================================================
function GraphView({ nodes, edges, onNodeClick, onEdgeClick, theme }) {
  const svgRef = useRef(null)
  const [tooltip, setTooltip] = useState(null)
  
  useEffect(() => {
    if (!svgRef.current) return
    
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()
    
    if (!nodes.length) {
      const isDark = theme === 'dark'
      svg.append('text')
        .attr('x', '50%')
        .attr('y', '50%')
        .attr('text-anchor', 'middle')
        .attr('fill', isDark ? 'rgba(205, 214, 244, 0.5)' : 'rgba(30, 30, 30, 0.5)')
        .attr('font-size', '1.25rem')
        .text('No graph data to display')
      svg.append('text')
        .attr('x', '50%')
        .attr('y', '55%')
        .attr('text-anchor', 'middle')
        .attr('fill', isDark ? 'rgba(205, 214, 244, 0.3)' : 'rgba(30, 30, 30, 0.3)')
        .attr('font-size', '0.875rem')
        .text('Run a query with URI relationships')
      return
    }
    
    const width = svgRef.current.clientWidth
    const height = svgRef.current.clientHeight
    const isDark = theme === 'dark'
    
    const zoom = d3.zoom()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => g.attr('transform', event.transform))
    
    svg.call(zoom)
    
    const g = svg.append('g')
    
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(edges).id(d => d.id).distance(150))
      .force('charge', d3.forceManyBody().strength(-400))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(50))
    
    // Arrow marker
    svg.append('defs').append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 28)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('fill', isDark ? '#89b4fa' : '#1e66f5')
      .attr('d', 'M0,-5L10,0L0,5')
    
    // Links
    const link = g.append('g')
      .selectAll('line')
      .data(edges)
      .join('line')
      .attr('stroke', isDark ? '#585b70' : '#9ca0b0')
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.6)
      .attr('marker-end', 'url(#arrow)')
      .on('mouseover', (event, d) => {
        setTooltip({ x: event.pageX + 10, y: event.pageY + 10, content: d.predicate })
        d3.select(event.currentTarget).attr('stroke-opacity', 1).attr('stroke-width', 3)
      })
      .on('mouseout', (event) => {
        setTooltip(null)
        d3.select(event.currentTarget).attr('stroke-opacity', 0.6).attr('stroke-width', 2)
      })
      .on('click', (event, d) => onEdgeClick && onEdgeClick(d))
    
    // Edge labels
    const edgeLabels = g.append('g')
      .selectAll('text')
      .data(edges)
      .join('text')
      .text(d => d.label || getLocalName(d.predicate))
      .attr('font-size', '11px')
      .attr('fill', isDark ? '#a6adc8' : '#5c5f77')
      .attr('text-anchor', 'middle')
      .attr('pointer-events', 'none')
    
    // Nodes
    const node = g.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(d3.drag()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.1).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on('drag', (event, d) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0)
        }))
      .on('click', (event, d) => onNodeClick && onNodeClick(d))
    
    node.append('circle')
      .attr('r', 18)
      .attr('fill', isDark ? '#313244' : '#e6e9ef')
      .attr('stroke', isDark ? '#89b4fa' : '#1e66f5')
      .attr('stroke-width', 2)
    
    node.append('text')
      .attr('dy', 35)
      .attr('text-anchor', 'middle')
      .attr('fill', isDark ? '#cdd6f4' : '#4c4f69')
      .attr('font-size', '12px')
      .attr('font-weight', '500')
      .text(d => {
        const label = d.label || getLocalName(d.id)
        return label.length > 20 ? label.substring(0, 18) + '...' : label
      })
    
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y)
      
      edgeLabels
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2 - 8)
      
      node.attr('transform', d => `translate(${d.x},${d.y})`)
    })
    
    return () => simulation.stop()
  }, [nodes, edges, theme, onNodeClick, onEdgeClick])
  
  return (
    <div className="graph-container">
      <svg ref={svgRef} className="graph-svg" />
      {tooltip && (
        <div className="tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
          {tooltip.content}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Results Table Component
// ============================================================================
function ResultsTable({ results, columns, theme }) {
  if (!results || results.length === 0) {
    return <div className="empty-results">No results</div>
  }
  
  return (
    <div className="table-wrapper">
      <table className={`results-table ${theme}`}>
        <thead>
          <tr>
            {columns.map(col => <th key={col}>{col}</th>)}
          </tr>
        </thead>
        <tbody>
          {results.map((row, i) => (
            <tr key={i}>
              {columns.map(col => (
                <td key={col} title={String(row[col] ?? '')}>
                  {String(row[col] ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ============================================================================
// Create Project Modal
// ============================================================================
function CreateProjectModal({ isOpen, onClose, onCreate, theme }) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [error, setError] = useState(null)
  const [creating, setCreating] = useState(false)

  if (!isOpen) return null

  const handleCreate = async () => {
    if (!name.trim()) {
      setError('Project name is required')
      return
    }
    try {
      setCreating(true)
      setError(null)
      await onCreate(name.trim(), description.trim())
      setName('')
      setDescription('')
      onClose()
    } catch (err) {
      setError(err.message)
    } finally {
      setCreating(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className={`modal ${theme}`} onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Create New Repository</h2>
          <button className="icon-btn" onClick={onClose}><CloseIcon size={20} /></button>
        </div>
        <div className="modal-body">
          <div className="form-group">
            <label>Repository Name</label>
            <input
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="my-knowledge-graph"
              autoFocus
            />
            <small>Letters, numbers, hyphens, and underscores only</small>
          </div>
          <div className="form-group">
            <label>Description (optional)</label>
            <textarea
              value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder="A brief description of your repository..."
              rows={3}
            />
          </div>
          {error && <div className="error-message">{error}</div>}
        </div>
        <div className="modal-footer">
          <button className="btn secondary" onClick={onClose} disabled={creating}>Cancel</button>
          <button className="btn primary" onClick={handleCreate} disabled={creating}>
            {creating ? 'Creating...' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// Main App Component
// ============================================================================
function App() {
  // Theme state
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('rdf-starbase-theme')
    return saved || 'dark'
  })
  
  // Repository state
  const [repositories, setRepositories] = useState([])
  const [currentRepo, setCurrentRepo] = useState(null)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [stats, setStats] = useState(null)
  
  // Query state
  const [sparqlQuery, setSparqlQuery] = useState('SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100')
  const [queryResults, setQueryResults] = useState(null)
  const [executing, setExecuting] = useState(false)
  const [error, setError] = useState(null)
  
  // View state
  const [viewMode, setViewMode] = useState('table') // 'table' | 'graph' | 'json'
  const [sidePanel, setSidePanel] = useState('schema') // 'schema' | 'import' | null
  const [graphNodes, setGraphNodes] = useState([])
  const [graphEdges, setGraphEdges] = useState([])
  const [selectedNode, setSelectedNode] = useState(null)
  const [selectedEdge, setSelectedEdge] = useState(null)
  const [nodeProperties, setNodeProperties] = useState(null)
  
  // API state
  const [apiStatus, setApiStatus] = useState('checking')
  const [loading, setLoading] = useState(true)

  // Theme effect
  useEffect(() => {
    localStorage.setItem('rdf-starbase-theme', theme)
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  // Load repositories
  const loadRepositories = useCallback(async () => {
    try {
      const data = await fetchJson('/repositories')
      setRepositories(data.repositories || [])
      return data.repositories || []
    } catch (err) {
      console.error('Failed to load repositories:', err)
      return []
    }
  }, [])

  // Load stats
  const loadStats = useCallback(async (repoName) => {
    if (!repoName) {
      setStats(null)
      return
    }
    try {
      const data = await fetchJson(`/repositories/${repoName}/stats`)
      setStats(data.stats)
    } catch (err) {
      console.error('Failed to load stats:', err)
    }
  }, [])

  // Create repository
  const createRepository = useCallback(async (name, description) => {
    await fetchJson('/repositories', {
      method: 'POST',
      body: JSON.stringify({ name, description, tags: [] }),
    })
    await loadRepositories()
    setCurrentRepo(name)
  }, [loadRepositories])

  // Delete repository
  const deleteRepository = useCallback(async (name) => {
    if (!confirm(`Delete repository "${name}"? This cannot be undone.`)) return
    try {
      await fetchJson(`/repositories/${name}?force=true`, { method: 'DELETE' })
      await loadRepositories()
      if (currentRepo === name) {
        setCurrentRepo(null)
        setGraphNodes([])
        setGraphEdges([])
        setStats(null)
        setQueryResults(null)
      }
    } catch (err) {
      setError(err.message)
    }
  }, [currentRepo, loadRepositories])

  // Build graph from results
  const buildGraph = useCallback((results, columns) => {
    const nodeSet = new Set()
    const edgeList = []
    const hasTriples = columns.includes('s') && columns.includes('p') && columns.includes('o')
    
    if (hasTriples) {
      for (const row of results) {
        const { s, p, o } = row
        if (!s || !p || !o) continue
        nodeSet.add(s)
        if (typeof o === 'string' && (o.startsWith('http') || o.startsWith('urn:'))) {
          nodeSet.add(o)
          edgeList.push({ source: s, target: o, predicate: p, label: getLocalName(p) })
        }
      }
    } else {
      for (const row of results) {
        for (const col of columns) {
          const val = row[col]
          if (typeof val === 'string' && (val.startsWith('http') || val.startsWith('urn:'))) {
            nodeSet.add(val)
          }
        }
      }
    }
    
    return {
      nodes: [...nodeSet].map(id => ({ id, label: getLocalName(id) })),
      edges: edgeList
    }
  }, [])

  // Execute SPARQL
  const executeSparql = useCallback(async (query = null) => {
    const q = query || sparqlQuery
    if (!currentRepo) {
      setError('Please select or create a repository first')
      return
    }
    
    try {
      setError(null)
      setExecuting(true)
      
      const result = await fetchJson(`/repositories/${currentRepo}/sparql`, {
        method: 'POST',
        body: JSON.stringify({ query: q }),
      })
      
      setQueryResults(result)
      
      if (result.type === 'select' && result.results) {
        const { nodes, edges } = buildGraph(result.results, result.columns)
        setGraphNodes(nodes)
        setGraphEdges(edges)
        if (edges.length > 0 && viewMode !== 'json') {
          setViewMode('graph')
        }
      } else if (result.type === 'construct' && result.triples) {
        const nodeSet = new Set()
        const edgeList = []
        for (const t of result.triples) {
          nodeSet.add(t.subject)
          if (t.object.startsWith('http') || t.object.startsWith('urn:')) {
            nodeSet.add(t.object)
            edgeList.push({ source: t.subject, target: t.object, predicate: t.predicate, label: getLocalName(t.predicate) })
          }
        }
        setGraphNodes([...nodeSet].map(id => ({ id, label: getLocalName(id) })))
        setGraphEdges(edgeList)
      } else {
        setGraphNodes([])
        setGraphEdges([])
      }
      
      if (result.type === 'update') {
        await loadStats(currentRepo)
        await loadRepositories()
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setExecuting(false)
    }
  }, [currentRepo, sparqlQuery, buildGraph, loadStats, loadRepositories, viewMode])

  // Initialize
  useEffect(() => {
    async function init() {
      try {
        setLoading(true)
        await fetchJson('/health')
        setApiStatus('online')
        const repos = await loadRepositories()
        if (repos.length > 0) {
          setCurrentRepo(repos[0].name)
        }
      } catch (err) {
        setApiStatus('offline')
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    init()
  }, [loadRepositories])

  // Load stats when repo changes
  useEffect(() => {
    if (currentRepo && apiStatus === 'online') {
      loadStats(currentRepo)
      executeSparql('SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100')
    }
  }, [currentRepo, apiStatus])

  // Handle node click - fetch properties with provenance
  const handleNodeClick = useCallback(async (node) => {
    setSelectedNode(node)
    setSelectedEdge(null)
    if (!currentRepo) return
    
    try {
      // Query for all properties of this node with provenance
      const query = `
        SELECT ?p ?o ?source ?confidence WHERE {
          <${node.id}> ?p ?o .
          OPTIONAL {
            << <${node.id}> ?p ?o >> <http://rdf-starbase.dev/source> ?source .
            << <${node.id}> ?p ?o >> <http://rdf-starbase.dev/confidence> ?confidence .
          }
        }
      `
      const response = await fetchJson(`/repositories/${currentRepo}/query`, {
        method: 'POST',
        body: JSON.stringify({ query }),
      })
      setNodeProperties(response.results || [])
    } catch (err) {
      console.error('Failed to load node properties:', err)
      setNodeProperties([])
    }
  }, [currentRepo])

  // Handle edge click
  const handleEdgeClick = useCallback(async (edge) => {
    setSelectedEdge(edge)
    setSelectedNode(null)
    // Edge already contains source, target, predicate
    // Could query for provenance here if needed
  }, [])

  // Handle schema insert
  const handleSchemaInsert = (snippet) => {
    setSparqlQuery(prev => {
      const whereMatch = prev.match(/WHERE\s*\{/i)
      if (whereMatch) {
        const insertPos = whereMatch.index + whereMatch[0].length
        return prev.slice(0, insertPos) + '\n  ' + snippet + prev.slice(insertPos)
      }
      return prev + '\n' + snippet
    })
  }

  // Sample queries
  const sampleQueries = [
    { icon: '📊', label: 'All Triples', query: 'SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100' },
    { icon: '🔗', label: 'Relationships', query: 'SELECT ?s ?p ?o WHERE { ?s ?p ?o . FILTER(isIRI(?o)) } LIMIT 100' },
    { icon: '📈', label: 'Statistics', query: 'SELECT ?p (COUNT(*) AS ?count) WHERE { ?s ?p ?o } GROUP BY ?p ORDER BY DESC(?count)' },
    { icon: '🏷️', label: 'Classes', query: 'SELECT ?class (COUNT(?s) AS ?count) WHERE { ?s a ?class } GROUP BY ?class ORDER BY DESC(?count)' },
  ]

  // Offline screen
  if (apiStatus === 'offline') {
    return (
      <div className={`app ${theme}`}>
        <div className="offline-screen">
          <div className="offline-content">
            <DatabaseIcon size={64} />
            <h2>API Server Not Running</h2>
            <p>The RDF-StarBase API server is not responding.</p>
            <div className="offline-instructions">
              <h3>To start the server:</h3>
              <code>cd e:\RDF-StarBase</code>
              <code>uvicorn rdf_starbase.repository_api:app --reload</code>
            </div>
            <button className="btn primary" onClick={() => window.location.reload()}>
              <RefreshIcon size={16} /> Retry Connection
            </button>
          </div>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className={`app ${theme}`}>
        <div className="loading-screen">
          <div className="spinner" />
          <p>Connecting to RDF-StarBase...</p>
        </div>
      </div>
    )
  }

  return (
    <div className={`app ${theme}`}>
      <CreateProjectModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onCreate={createRepository}
        theme={theme}
      />

      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <div className="logo">
            <DatabaseIcon size={24} />
            <span>RDF-StarBase</span>
          </div>
        </div>

        <div className="header-center">
          <div className="repo-selector">
            <select
              value={currentRepo || ''}
              onChange={(e) => setCurrentRepo(e.target.value || null)}
            >
              <option value="">Select repository...</option>
              {repositories.map(r => (
                <option key={r.name} value={r.name}>
                  {r.name} ({r.triple_count} triples)
                </option>
              ))}
            </select>
            <button className="icon-btn" onClick={() => setShowCreateModal(true)} title="Create repository">
              <PlusIcon size={18} />
            </button>
            {currentRepo && (
              <button className="icon-btn danger" onClick={() => deleteRepository(currentRepo)} title="Delete repository">
                <TrashIcon size={18} />
              </button>
            )}
          </div>
        </div>

        <div className="header-right">
          {stats && (
            <div className="stats">
              <span><strong>{stats.total_assertions || 0}</strong> triples</span>
              <span><strong>{stats.unique_subjects || 0}</strong> subjects</span>
            </div>
          )}
          <button 
            className="icon-btn theme-toggle" 
            onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? <SunIcon size={18} /> : <MoonIcon size={18} />}
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        {/* Query Panel */}
        <div className="query-panel">
          <div className="query-toolbar">
            <div className="quick-queries">
              {sampleQueries.map((sq, i) => (
                <button
                  key={i}
                  className="quick-query-btn"
                  onClick={() => { setSparqlQuery(sq.query); executeSparql(sq.query) }}
                  title={sq.query}
                >
                  {sq.icon} {sq.label}
                </button>
              ))}
            </div>
            <button 
              className="btn primary execute-btn"
              onClick={() => executeSparql()}
              disabled={executing || !currentRepo}
            >
              <PlayIcon size={16} />
              {executing ? 'Running...' : 'Run Query'}
            </button>
          </div>

          <div className="editor-container">
            <SparqlEditor
              value={sparqlQuery}
              onChange={setSparqlQuery}
              onExecute={() => executeSparql()}
              theme={theme}
              height="180px"
            />
          </div>

          {error && (
            <div className="error-bar">
              <span>{error}</span>
              <button onClick={() => setError(null)}><CloseIcon size={14} /></button>
            </div>
          )}
        </div>

        {/* Results Area */}
        <div className="results-area">
          <div className="results-toolbar">
            <div className="view-tabs">
              <button 
                className={`view-tab ${viewMode === 'table' ? 'active' : ''}`}
                onClick={() => setViewMode('table')}
              >
                <TableIcon size={16} /> Table
              </button>
              <button 
                className={`view-tab ${viewMode === 'graph' ? 'active' : ''}`}
                onClick={() => setViewMode('graph')}
                disabled={graphNodes.length === 0}
              >
                <NetworkIcon size={16} /> Graph
              </button>
              <button 
                className={`view-tab ${viewMode === 'json' ? 'active' : ''}`}
                onClick={() => setViewMode('json')}
              >
                <CodeIcon size={16} /> JSON
              </button>
            </div>

            {queryResults && (
              <div className="result-info">
                {queryResults.type === 'select' && `${queryResults.results?.length || 0} rows`}
                {queryResults.type === 'ask' && (queryResults.result ? '✓ TRUE' : '✗ FALSE')}
                {queryResults.type === 'update' && `${queryResults.affected || 0} affected`}
                {queryResults.type === 'construct' && `${queryResults.triples?.length || 0} triples`}
              </div>
            )}

            <div className="panel-toggles">
              <button 
                className={`icon-btn ${sidePanel === 'schema' ? 'active' : ''}`}
                onClick={() => setSidePanel(s => s === 'schema' ? null : 'schema')}
                title="Schema Browser"
              >
                <BookIcon size={18} />
              </button>
              <button 
                className={`icon-btn ${sidePanel === 'import' ? 'active' : ''}`}
                onClick={() => setSidePanel(s => s === 'import' ? null : 'import')}
                title="Import / Export"
              >
                I/O
              </button>
            </div>
          </div>

          <div className="results-content">
            <div className="results-main">
              {viewMode === 'table' && queryResults && (
                <>
                  {queryResults.type === 'select' && (
                    <ResultsTable results={queryResults.results} columns={queryResults.columns} theme={theme} />
                  )}
                  {queryResults.type === 'ask' && (
                    <div className="ask-result">
                      <span className={queryResults.result ? 'true' : 'false'}>
                        {queryResults.result ? 'TRUE' : 'FALSE'}
                      </span>
                    </div>
                  )}
                  {queryResults.type === 'update' && (
                    <div className="update-result">
                      <ZapIcon size={32} />
                      <h3>Update Executed Successfully</h3>
                      <p>{queryResults.affected || 0} triples affected</p>
                    </div>
                  )}
                  {queryResults.type === 'construct' && (
                    <ResultsTable 
                      results={queryResults.triples?.map(t => ({ subject: t.subject, predicate: t.predicate, object: t.object }))} 
                      columns={['subject', 'predicate', 'object']} 
                      theme={theme}
                    />
                  )}
                </>
              )}

              {viewMode === 'graph' && (
                <div className="graph-wrapper">
                  <GraphView 
                    nodes={graphNodes} 
                    edges={graphEdges} 
                    onNodeClick={handleNodeClick}
                    onEdgeClick={handleEdgeClick}
                    theme={theme} 
                  />
                  {(selectedNode || selectedEdge) && (
                    <div className={`graph-details-panel ${theme}`}>
                      <div className="graph-details-header">
                        <h4>{selectedNode ? 'Node Properties' : 'Edge Details'}</h4>
                        <button className="close-btn" onClick={() => { setSelectedNode(null); setSelectedEdge(null); setNodeProperties(null); }}>×</button>
                      </div>
                      <div className="graph-details-content">
                        {selectedNode && (
                          <>
                            <div className="detail-uri">{getLocalName(selectedNode.id)}</div>
                            <div className="detail-full-uri">{selectedNode.id}</div>
                            {nodeProperties && nodeProperties.length > 0 ? (
                              <table className="properties-table">
                                <thead>
                                  <tr><th>Property</th><th>Value</th><th>Source</th></tr>
                                </thead>
                                <tbody>
                                  {nodeProperties.map((prop, i) => (
                                    <tr key={i}>
                                      <td title={prop.p}>{getLocalName(prop.p)}</td>
                                      <td title={prop.o}>{getLocalName(prop.o)}</td>
                                      <td>{prop.source ? `${getLocalName(prop.source)} (${prop.confidence || 1})` : '-'}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            ) : (
                              <p className="no-properties">No properties found</p>
                            )}
                          </>
                        )}
                        {selectedEdge && (
                          <>
                            <div className="edge-detail">
                              <span className="edge-label">Predicate:</span>
                              <span className="edge-value" title={selectedEdge.predicate}>{getLocalName(selectedEdge.predicate)}</span>
                            </div>
                            <div className="edge-detail">
                              <span className="edge-label">Full URI:</span>
                              <span className="edge-value-small">{selectedEdge.predicate}</span>
                            </div>
                            <div className="edge-detail">
                              <span className="edge-label">Source:</span>
                              <span className="edge-value">{getLocalName(selectedEdge.source?.id || selectedEdge.source)}</span>
                            </div>
                            <div className="edge-detail">
                              <span className="edge-label">Target:</span>
                              <span className="edge-value">{getLocalName(selectedEdge.target?.id || selectedEdge.target)}</span>
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {viewMode === 'json' && queryResults && (
                <div className="json-view">
                  <pre>{JSON.stringify(queryResults, null, 2)}</pre>
                </div>
              )}

              {!queryResults && viewMode !== 'graph' && (
                <div className="empty-results">Run a query to see results</div>
              )}
            </div>

            {sidePanel && (
              <div className="side-panel">
                <button className="close-panel" onClick={() => setSidePanel(null)}>
                  <CloseIcon size={16} />
                </button>
                {sidePanel === 'schema' && (
                  <SchemaBrowser 
                    repositoryName={currentRepo} 
                    onInsert={handleSchemaInsert}
                    theme={theme}
                  />
                )}
                {sidePanel === 'import' && (
                  <ImportExport 
                    repositoryName={currentRepo}
                    onDataChanged={() => { loadStats(currentRepo); loadRepositories() }}
                    theme={theme}
                  />
                )}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
