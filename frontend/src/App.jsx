import { useState, useEffect, useCallback, useRef } from 'react'
import * as d3 from 'd3'
import './index.css'

// API base URL - in dev mode, vite proxies /api to localhost:8000
const API_BASE = '/api'

// Fetch helpers
async function fetchJson(endpoint, options = {}) {
  console.log(`[API] Fetching ${API_BASE}${endpoint}`)
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    console.error(`[API] Error: ${error.detail || response.statusText}`)
    throw new Error(error.detail || 'Request failed')
  }
  const data = await response.json()
  console.log(`[API] Response from ${endpoint}:`, data)
  return data
}

// Graph Visualization Component
function GraphView({ nodes, edges, selectedNode, onNodeClick }) {
  const svgRef = useRef(null)
  const [tooltip, setTooltip] = useState(null)
  
  useEffect(() => {
    if (!svgRef.current) return
    
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()
    
    if (!nodes.length) {
      // Show empty state message
      svg.append('text')
        .attr('x', '50%')
        .attr('y', '50%')
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.5)')
        .attr('font-size', '1.25rem')
        .text('No data in knowledge graph')
      svg.append('text')
        .attr('x', '50%')
        .attr('y', '55%')
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.3)')
        .attr('font-size', '0.875rem')
        .text('Add triples via the API or load demo data')
      return
    }
    
    const width = svgRef.current.clientWidth
    const height = svgRef.current.clientHeight
    
    // Create zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })
    
    svg.call(zoom)
    
    const g = svg.append('g')
    
    // Create simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(edges).id(d => d.id).distance(150))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(40))
    
    // Create arrow marker
    svg.append('defs').append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 25)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('fill', 'rgba(233, 69, 96, 0.5)')
      .attr('d', 'M0,-5L10,0L0,5')
    
    // Create links
    const link = g.append('g')
      .selectAll('line')
      .data(edges)
      .join('line')
      .attr('class', 'link')
      .attr('marker-end', 'url(#arrow)')
      .on('mouseover', (event, d) => {
        setTooltip({
          x: event.pageX + 10,
          y: event.pageY + 10,
          content: d.predicate
        })
      })
      .on('mouseout', () => setTooltip(null))
    
    // Create edge labels
    const edgeLabels = g.append('g')
      .selectAll('text')
      .data(edges)
      .join('text')
      .attr('class', 'edge-label')
      .text(d => d.label || d.predicate.split('/').pop())
    
    // Create nodes
    const node = g.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('class', d => `node ${selectedNode === d.id ? 'selected' : ''}`)
      .call(d3.drag()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on('drag', (event, d) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0)
          d.fx = null
          d.fy = null
        }))
      .on('click', (event, d) => onNodeClick(d))
    
    node.append('circle')
      .attr('r', 15)
    
    node.append('text')
      .attr('dy', 30)
      .text(d => d.label || d.id.split('/').pop())
    
    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y)
      
      edgeLabels
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2)
      
      node.attr('transform', d => `translate(${d.x},${d.y})`)
    })
    
    // Cleanup
    return () => simulation.stop()
  }, [nodes, edges, selectedNode, onNodeClick])
  
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

// Node Info Panel Component
function NodeInfoPanel({ node, triples, onClose }) {
  if (!node) return null
  
  return (
    <div className="info-panel">
      <button className="close-btn" onClick={onClose}>&times;</button>
      <h3>{node.label || node.id.split('/').pop()}</h3>
      <div className="triple-item" style={{ background: 'none', padding: 0 }}>
        <code style={{ fontSize: '0.7rem', wordBreak: 'break-all', color: 'rgba(255,255,255,0.5)' }}>
          {node.id}
        </code>
      </div>
      <div className="section-title" style={{ marginTop: '1rem' }}>Properties</div>
      <div className="triple-list">
        {triples.map((t, i) => (
          <div key={i} className="triple-item">
            <span className="predicate">{t.predicate.split('/').pop()}</span>
            {' → '}
            <span className="object">{t.object}</span>
            <div className="meta">
              Source: {t.source} | Confidence: {(t.confidence * 100).toFixed(0)}%
            </div>
          </div>
        ))}
        {triples.length === 0 && (
          <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '0.875rem' }}>
            No properties found
          </div>
        )}
      </div>
    </div>
  )
}

// Results Table Component  
function ResultsTable({ results, columns }) {
  if (!results || results.length === 0) return null
  
  return (
    <div className="results-container">
      <table className="results-table">
        <thead>
          <tr>
            {columns.map(col => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {results.map((row, i) => (
            <tr key={i}>
              {columns.map(col => (
                <td key={col}>{String(row[col] ?? '')}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// Main App Component
function App() {
  // State
  const [stats, setStats] = useState(null)
  const [nodes, setNodes] = useState([])
  const [edges, setEdges] = useState([])
  const [selectedNode, setSelectedNode] = useState(null)
  const [nodeTriples, setNodeTriples] = useState([])
  const [sources, setSources] = useState([])
  const [sparqlQuery, setSparqlQuery] = useState('SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10')
  const [queryResults, setQueryResults] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('graph')
  const [apiStatus, setApiStatus] = useState('checking') // 'checking', 'online', 'offline'
  
  // Load initial data
  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true)
        setError(null)
        setApiStatus('checking')
        
        console.log('[App] Loading data from API...')
        
        const [statsData, nodesData, edgesData, sourcesData] = await Promise.all([
          fetchJson('/stats'),
          fetchJson('/graph/nodes?limit=50'),
          fetchJson('/graph/edges?limit=200'),
          fetchJson('/sources'),
        ])
        
        setApiStatus('online')
        setStats(statsData)
        setNodes(nodesData.nodes)
        setEdges(edgesData.edges.map(e => ({
          ...e,
          source: e.source,
          target: e.target,
        })))
        setSources(sourcesData.sources)
        console.log('[App] Data loaded successfully', { nodes: nodesData.nodes.length, edges: edgesData.edges.length })
      } catch (err) {
        console.error('[App] Failed to load data:', err)
        setApiStatus('offline')
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    
    loadData()
  }, [])
  
  // Handle node click - load node details
  const handleNodeClick = useCallback(async (node) => {
    setSelectedNode(node.id)
    
    try {
      const triples = await fetchJson(`/triples?subject=${encodeURIComponent(node.id)}`)
      setNodeTriples(triples.triples)
    } catch (err) {
      console.error('Failed to load node triples:', err)
      setNodeTriples([])
    }
  }, [])
  
  // Execute SPARQL query
  const executeSparql = useCallback(async () => {
    try {
      setError(null)
      const result = await fetchJson('/sparql', {
        method: 'POST',
        body: JSON.stringify({ query: sparqlQuery }),
      })
      setQueryResults(result)
    } catch (err) {
      setError(err.message)
    }
  }, [sparqlQuery])
  
  // Sample queries - updated for demo data
  const sampleQueries = [
    { label: 'All Triples', query: 'SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 20' },
    { label: 'Movie Names', query: 'SELECT ?movie ?name WHERE { ?movie <http://schema.org/name> ?name }' },
    { label: 'Nolan Films', query: 'SELECT ?m ?title WHERE { ?m <http://schema.org/director> <http://example.org/person/nolan> . ?m <http://schema.org/name> ?title }' },
    { label: 'Ask Inception', query: 'ASK WHERE { ?s <http://schema.org/name> "Inception" }' },
  ]
  
  // Show API offline screen
  if (apiStatus === 'offline') {
    return (
      <div className="app">
        <div className="api-offline">
          <div className="offline-icon">⚠️</div>
          <h2>API Server Not Running</h2>
          <p>The RDF-StarBase API server is not responding.</p>
          <div className="offline-instructions">
            <h3>To start the server with demo data:</h3>
            <code>cd e:\RDF-StarBase</code>
            <code>python scripts/run_demo.py</code>
            <p style={{marginTop: '1rem', fontSize: '0.875rem', color: 'rgba(255,255,255,0.6)'}}>
              Or start the basic server: <code style={{display: 'inline'}}>uvicorn rdf_starbase.web:app</code>
            </p>
          </div>
          <button className="btn" onClick={() => window.location.reload()}>
            Retry Connection
          </button>
          {error && <div className="error" style={{marginTop: '1rem'}}>{error}</div>}
        </div>
      </div>
    )
  }
  
  if (loading) {
    return (
      <div className="app">
        <div className="loading">Connecting to RDF-StarBase API...</div>
      </div>
    )
  }
  
  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1>
          <svg className="logo" viewBox="0 0 24 24">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
          </svg>
          RDF-StarBase Explorer
        </h1>
        {stats && (
          <div className="stats-bar">
            <span>
              Assertions: <span className="value">{stats.store?.total_assertions || 0}</span>
            </span>
            <span>
              Sources: <span className="value">{stats.registry?.total_sources || 0}</span>
            </span>
            <span>
              Unique Subjects: <span className="value">{stats.store?.unique_subjects || 0}</span>
            </span>
          </div>
        )}
      </header>
      
      <div className="main-content">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="tabs">
            <button 
              className={`tab ${activeTab === 'graph' ? 'active' : ''}`}
              onClick={() => setActiveTab('graph')}
            >
              Graph
            </button>
            <button 
              className={`tab ${activeTab === 'sparql' ? 'active' : ''}`}
              onClick={() => setActiveTab('sparql')}
            >
              SPARQL
            </button>
            <button 
              className={`tab ${activeTab === 'sources' ? 'active' : ''}`}
              onClick={() => setActiveTab('sources')}
            >
              Sources
            </button>
          </div>
          
          {activeTab === 'sparql' && (
            <div className="section">
              <div className="section-title">SPARQL Query</div>
              <textarea
                className="sparql-input"
                value={sparqlQuery}
                onChange={(e) => setSparqlQuery(e.target.value)}
                placeholder="Enter SPARQL query..."
              />
              <div className="btn-group">
                <button className="btn" onClick={executeSparql}>
                  Execute
                </button>
              </div>
              
              <div className="section-title" style={{ marginTop: '1rem' }}>Examples</div>
              {sampleQueries.map((sq, i) => (
                <button
                  key={i}
                  className="btn btn-secondary"
                  style={{ width: '100%', marginBottom: '0.25rem', textAlign: 'left' }}
                  onClick={() => setSparqlQuery(sq.query)}
                >
                  {sq.label}
                </button>
              ))}
              
              {error && <div className="error">{error}</div>}
              
              {queryResults && (
                <div style={{ marginTop: '1rem' }}>
                  <div className="section-title">
                    Results ({queryResults.type})
                    {queryResults.type === 'ask' && (
                      <span className="value" style={{ marginLeft: '0.5rem' }}>
                        {queryResults.result ? 'TRUE' : 'FALSE'}
                      </span>
                    )}
                  </div>
                  {queryResults.type === 'select' && (
                    <ResultsTable 
                      results={queryResults.results} 
                      columns={queryResults.columns} 
                    />
                  )}
                </div>
              )}
            </div>
          )}
          
          {activeTab === 'sources' && (
            <div className="section">
              <div className="section-title">Registered Sources</div>
              <div className="source-list">
                {sources.map((source) => (
                  <div key={source.id} className="source-item">
                    <div className="name">{source.name}</div>
                    <div className="type">
                      {source.source_type} | Status: {source.status}
                    </div>
                    {source.uri && (
                      <div className="type">{source.uri}</div>
                    )}
                  </div>
                ))}
                {sources.length === 0 && (
                  <div style={{ color: 'rgba(255,255,255,0.5)' }}>
                    No sources registered
                  </div>
                )}
              </div>
            </div>
          )}
          
          {activeTab === 'graph' && (
            <div className="section">
              <div className="section-title">Graph Info</div>
              <p style={{ fontSize: '0.875rem', color: 'rgba(255,255,255,0.7)' }}>
                Click on a node to view its properties.
                Drag nodes to rearrange. Scroll to zoom.
              </p>
              <div style={{ marginTop: '1rem', fontSize: '0.875rem' }}>
                <div>Nodes: <span className="value">{nodes.length}</span></div>
                <div>Edges: <span className="value">{edges.length}</span></div>
              </div>
            </div>
          )}
        </aside>
        
        {/* Main Graph Area */}
        <GraphView
          nodes={nodes}
          edges={edges}
          selectedNode={selectedNode}
          onNodeClick={handleNodeClick}
        />
        
        {/* Node Info Panel */}
        {selectedNode && (
          <NodeInfoPanel
            node={nodes.find(n => n.id === selectedNode)}
            triples={nodeTriples}
            onClose={() => {
              setSelectedNode(null)
              setNodeTriples([])
            }}
          />
        )}
      </div>
    </div>
  )
}

export default App
