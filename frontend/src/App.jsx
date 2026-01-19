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

// Repository selector for the selected project
async function fetchRepoJson(repoName, endpoint, options = {}) {
  return fetchJson(`/repositories/${repoName}${endpoint}`, options)
}

// Helper to extract local name from URI (checks # first, then /)
const getLocalName = (uri) => {
  if (!uri) return uri
  if (uri.includes('#')) return uri.split('#').pop()
  return uri.split('/').pop()
}

// Create Project Modal Component
function CreateProjectModal({ isOpen, onClose, onCreate }) {
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
      <div className="modal" onClick={e => e.stopPropagation()}>
        <h2>Create New Project</h2>
        <div className="form-group">
          <label>Project Name</label>
          <input
            type="text"
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder="my-project"
            pattern="[a-zA-Z0-9\-_]+"
            autoFocus
          />
          <small>Letters, numbers, hyphens, and underscores only</small>
        </div>
        <div className="form-group">
          <label>Description (optional)</label>
          <textarea
            value={description}
            onChange={e => setDescription(e.target.value)}
            placeholder="A brief description of your project..."
            rows={3}
          />
        </div>
        {error && <div className="error">{error}</div>}
        <div className="modal-buttons">
          <button className="btn btn-secondary" onClick={onClose} disabled={creating}>
            Cancel
          </button>
          <button className="btn" onClick={handleCreate} disabled={creating}>
            {creating ? 'Creating...' : 'Create Project'}
          </button>
        </div>
      </div>
    </div>
  )
}

// Load Data Modal Component
function LoadDataModal({ isOpen, onClose, onLoadExample, onLoadCustom, currentProject }) {
  const [tab, setTab] = useState('examples')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [customQuery, setCustomQuery] = useState(`PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>

INSERT DATA {
  ex:person1 foaf:name "Alice" .
  ex:person1 foaf:knows ex:person2 .
  ex:person2 foaf:name "Bob" .
}`)

  const exampleDatasets = [
    { id: 'movies', name: 'Movies & Directors', description: 'Film database with ratings from multiple sources' },
    { id: 'techcorp', name: 'TechCorp Enterprise', description: 'Customer data with conflicts from 8 systems' },
    { id: 'knowledge', name: 'Knowledge Graph', description: 'Academic entities with citations' },
    { id: 'rdfstar', name: 'RDF-Star Demo', description: 'Quoted triples and nested annotations' },
  ]

  if (!isOpen) return null

  const handleLoadExample = async (datasetId) => {
    try {
      setLoading(true)
      setError(null)
      await onLoadExample(datasetId)
      onClose()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleLoadCustom = async () => {
    try {
      setLoading(true)
      setError(null)
      await onLoadCustom(customQuery)
      onClose()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal modal-wide" onClick={e => e.stopPropagation()}>
        <h2>Load Data into {currentProject}</h2>
        
        <div className="tabs" style={{ marginBottom: '1rem' }}>
          <button 
            className={`tab ${tab === 'examples' ? 'active' : ''}`}
            onClick={() => setTab('examples')}
          >
            Example Datasets
          </button>
          <button 
            className={`tab ${tab === 'custom' ? 'active' : ''}`}
            onClick={() => setTab('custom')}
          >
            Custom SPARQL
          </button>
        </div>

        {tab === 'examples' && (
          <div className="dataset-grid">
            {exampleDatasets.map(ds => (
              <div key={ds.id} className="dataset-card">
                <div className="dataset-name">{ds.name}</div>
                <div className="dataset-desc">{ds.description}</div>
                <button 
                  className="btn btn-small"
                  onClick={() => handleLoadExample(ds.id)}
                  disabled={loading}
                >
                  {loading ? 'Loading...' : 'Load'}
                </button>
              </div>
            ))}
          </div>
        )}

        {tab === 'custom' && (
          <div className="form-group">
            <label>SPARQL INSERT DATA Statement</label>
            <textarea
              className="sparql-input"
              value={customQuery}
              onChange={e => setCustomQuery(e.target.value)}
              rows={10}
              style={{ fontFamily: 'monospace', fontSize: '0.875rem' }}
            />
            <button 
              className="btn"
              onClick={handleLoadCustom}
              disabled={loading}
              style={{ marginTop: '0.5rem' }}
            >
              {loading ? 'Loading...' : 'Execute INSERT'}
            </button>
          </div>
        )}

        {error && <div className="error">{error}</div>}
        
        <div className="modal-buttons">
          <button className="btn btn-secondary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  )
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
      svg.append('text')
        .attr('x', '50%')
        .attr('y', '50%')
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.5)')
        .attr('font-size', '1.25rem')
        .text('No graph data to display')
      svg.append('text')
        .attr('x', '50%')
        .attr('y', '55%')
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.3)')
        .attr('font-size', '0.875rem')
        .text('Run a query with URI relationships')
      return
    }
    
    const width = svgRef.current.clientWidth
    const height = svgRef.current.clientHeight
    
    const zoom = d3.zoom()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })
    
    svg.call(zoom)
    
    const g = svg.append('g')
    
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(edges).id(d => d.id).distance(150))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(40))
    
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
    
    const edgeLabels = g.append('g')
      .selectAll('text')
      .data(edges)
      .join('text')
      .attr('class', 'edge-label')
      .text(d => d.label || getLocalName(d.predicate))
    
    const node = g.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('class', d => `node ${selectedNode === d.id ? 'selected' : ''}`)
      .call(d3.drag()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.01).restart()
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
      .on('click', (event, d) => onNodeClick(d))
    
    node.append('circle')
      .attr('r', 15)
    
    node.append('text')
      .attr('dy', 30)
      .text(d => d.label || getLocalName(d.id))
    
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
    
    return () => simulation.stop()
  }, [nodes, edges, selectedNode, onNodeClick])
  
  return (
    <div className="graph-container">
      <svg ref={svgRef} className="graph-svg" />
      {tooltip && (
        <div 
          className="tooltip" 
          style={{ left: tooltip.x, top: tooltip.y }}
        >
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
      <h3>{node.label || getLocalName(node.id)}</h3>
      <div className="triple-item" style={{ background: 'none', padding: 0 }}>
        <code style={{ fontSize: '0.7rem', wordBreak: 'break-all', color: 'rgba(255,255,255,0.5)' }}>
          {node.id}
        </code>
      </div>
      <div className="section-title" style={{ marginTop: '1rem' }}>Properties</div>
      <div className="triple-list">
        {triples.map((t, i) => (
          <div key={i} className="triple-item">
            <span className="predicate">{getLocalName(t.predicate)}</span>
            {'  '}
            <span className="object">{t.object}</span>
            <div className="meta">
              Source: {t.source} | Confidence: {(t.confidence * 100).toFixed(0)}%
            </div>
          </div>
        ))}
        {triples.length === 0 && (
          <div style={{ color: 'rgba(255,255,255,0.5)' }}>No properties found</div>
        )}
      </div>
    </div>
  )
}

// Results Table Component  
function ResultsTable({ results, columns }) {
  if (!results || results.length === 0) {
    return <div className="empty-results">No results</div>
  }
  
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

// Raw JSON View Component
function RawView({ data }) {
  return (
    <div className="raw-view">
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  )
}

// Query Results Panel - shows results in different views
function QueryResultsPanel({ 
  queryResults, 
  viewMode, 
  setViewMode, 
  graphNodes, 
  graphEdges,
  selectedNode,
  onNodeClick,
  nodeTriples,
  onCloseNodePanel
}) {
  const canShowGraph = graphNodes.length > 0 || graphEdges.length > 0
  
  return (
    <div className="results-panel">
      <div className="view-toggle">
        <button 
          className={`view-btn ${viewMode === 'graph' ? 'active' : ''}`}
          onClick={() => setViewMode('graph')}
          disabled={!canShowGraph}
          title={canShowGraph ? 'Graph View' : 'No graph relationships in results'}
        >
          <svg viewBox="0 0 24 24" width="16" height="16">
            <circle cx="5" cy="12" r="3" fill="currentColor"/>
            <circle cx="19" cy="6" r="3" fill="currentColor"/>
            <circle cx="19" cy="18" r="3" fill="currentColor"/>
            <line x1="8" y1="12" x2="16" y2="7" stroke="currentColor" strokeWidth="2"/>
            <line x1="8" y1="12" x2="16" y2="17" stroke="currentColor" strokeWidth="2"/>
          </svg>
          Graph
        </button>
        <button 
          className={`view-btn ${viewMode === 'table' ? 'active' : ''}`}
          onClick={() => setViewMode('table')}
        >
          <svg viewBox="0 0 24 24" width="16" height="16">
            <rect x="3" y="3" width="18" height="4" fill="currentColor"/>
            <rect x="3" y="9" width="18" height="3" fill="currentColor" opacity="0.7"/>
            <rect x="3" y="14" width="18" height="3" fill="currentColor" opacity="0.5"/>
            <rect x="3" y="19" width="18" height="2" fill="currentColor" opacity="0.3"/>
          </svg>
          Table
        </button>
        <button 
          className={`view-btn ${viewMode === 'raw' ? 'active' : ''}`}
          onClick={() => setViewMode('raw')}
        >
          <svg viewBox="0 0 24 24" width="16" height="16">
            <text x="4" y="16" fontSize="12" fill="currentColor">{ }</text>
          </svg>
          Raw
        </button>
        
        {queryResults && (
          <span className="result-info">
            {queryResults.type === 'select' && `${queryResults.results?.length || 0} rows`}
            {queryResults.type === 'ask' && (queryResults.result ? ' TRUE' : ' FALSE')}
            {queryResults.type === 'update' && `${queryResults.affected || 0} affected`}
            {queryResults.type === 'construct' && `${queryResults.triples?.length || 0} triples`}
          </span>
        )}
      </div>
      
      <div className="results-content">
        {viewMode === 'graph' && (
          <>
            <GraphView
              nodes={graphNodes}
              edges={graphEdges}
              selectedNode={selectedNode}
              onNodeClick={onNodeClick}
            />
            {selectedNode && (
              <NodeInfoPanel
                node={graphNodes.find(n => n.id === selectedNode)}
                triples={nodeTriples}
                onClose={onCloseNodePanel}
              />
            )}
          </>
        )}
        
        {viewMode === 'table' && queryResults && (
          <>
            {queryResults.type === 'select' && (
              <ResultsTable 
                results={queryResults.results} 
                columns={queryResults.columns} 
              />
            )}
            {queryResults.type === 'ask' && (
              <div className="ask-result">
                <div className={`ask-value ${queryResults.result ? 'true' : 'false'}`}>
                  {queryResults.result ? 'TRUE' : 'FALSE'}
                </div>
              </div>
            )}
            {queryResults.type === 'update' && (
              <div className="update-result">
                <div className="update-message"> Update executed successfully</div>
                <div className="update-details">{queryResults.affected || 0} triples affected</div>
              </div>
            )}
            {queryResults.type === 'construct' && (
              <ResultsTable 
                results={queryResults.triples?.map(t => ({
                  subject: t.subject,
                  predicate: t.predicate,
                  object: t.object
                })) || []} 
                columns={['subject', 'predicate', 'object']} 
              />
            )}
          </>
        )}
        
        {viewMode === 'raw' && queryResults && (
          <RawView data={queryResults} />
        )}
        
        {!queryResults && viewMode !== 'graph' && (
          <div className="empty-results">Run a query to see results</div>
        )}
      </div>
    </div>
  )
}

// Main App Component
function App() {
  const [projects, setProjects] = useState([])
  const [currentProject, setCurrentProject] = useState(null)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showLoadDataModal, setShowLoadDataModal] = useState(false)
  
  const [stats, setStats] = useState(null)
  const [graphNodes, setGraphNodes] = useState([])
  const [graphEdges, setGraphEdges] = useState([])
  const [selectedNode, setSelectedNode] = useState(null)
  const [nodeTriples, setNodeTriples] = useState([])
  const [sources, setSources] = useState([])
  const [sparqlQuery, setSparqlQuery] = useState('SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100')
  const [queryResults, setQueryResults] = useState(null)
  const [loading, setLoading] = useState(true)
  const [executing, setExecuting] = useState(false)
  const [error, setError] = useState(null)
  const [viewMode, setViewMode] = useState('graph')
  const [sidebarTab, setSidebarTab] = useState('projects')
  const [apiStatus, setApiStatus] = useState('checking')
  
  const loadProjects = useCallback(async () => {
    try {
      const data = await fetchJson('/repositories')
      setProjects(data.repositories || [])
      return data.repositories || []
    } catch (err) {
      console.error('[App] Failed to load projects:', err)
      return []
    }
  }, [])
  
  const createProject = useCallback(async (name, description) => {
    await fetchJson('/repositories', {
      method: 'POST',
      body: JSON.stringify({ name, description, tags: [] }),
    })
    await loadProjects()
    setCurrentProject(name)
  }, [loadProjects])
  
  const deleteProject = useCallback(async (name) => {
    if (!confirm(`Delete project "${name}"? This cannot be undone.`)) return
    
    try {
      await fetchJson(`/repositories/${name}?force=true`, { method: 'DELETE' })
      await loadProjects()
      if (currentProject === name) {
        setCurrentProject(null)
        setGraphNodes([])
        setGraphEdges([])
        setStats(null)
        setQueryResults(null)
      }
    } catch (err) {
      setError(err.message)
    }
  }, [currentProject, loadProjects])
  
  const loadProjectStats = useCallback(async (projectName) => {
    if (!projectName) {
      setStats(null)
      return
    }
    
    try {
      const statsData = await fetchRepoJson(projectName, '/stats')
      setStats(statsData.stats)
    } catch (err) {
      console.error('[App] Failed to load project stats:', err)
    }
  }, [])
  
  const buildGraphFromResults = useCallback((results, columns) => {
    const nodeSet = new Set()
    const edgeList = []
    
    const hasTriplePattern = columns.includes('s') && columns.includes('p') && columns.includes('o')
    
    if (hasTriplePattern) {
      for (const row of results) {
        const subject = row.s
        const predicate = row.p
        const obj = row.o
        
        if (!subject || !predicate || !obj) continue
        
        nodeSet.add(subject)
        
        if (typeof obj === 'string' && (obj.startsWith('http') || obj.startsWith('urn:'))) {
          nodeSet.add(obj)
          edgeList.push({
            source: subject,
            target: obj,
            predicate: predicate,
            label: getLocalName(predicate),
          })
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
    
    const nodes = [...nodeSet].map(id => ({
      id,
      label: getLocalName(id)
    }))
    
    return { nodes, edges: edgeList }
  }, [])
  
  const executeSparql = useCallback(async (query = null) => {
    const queryToRun = query || sparqlQuery
    
    if (!currentProject) {
      setError('Please select or create a project first')
      return
    }
    
    try {
      setError(null)
      setExecuting(true)
      
      const result = await fetchRepoJson(currentProject, '/sparql', {
        method: 'POST',
        body: JSON.stringify({ query: queryToRun }),
      })
      
      setQueryResults(result)
      
      if (result.type === 'select' && result.results) {
        const { nodes, edges } = buildGraphFromResults(result.results, result.columns)
        setGraphNodes(nodes)
        setGraphEdges(edges)
        
        if (edges.length > 0) {
          setViewMode('graph')
        } else if (nodes.length === 0) {
          setViewMode('table')
        }
      } else if (result.type === 'construct' && result.triples) {
        const nodeSet = new Set()
        const edgeList = []
        
        for (const triple of result.triples) {
          nodeSet.add(triple.subject)
          if (triple.object.startsWith('http') || triple.object.startsWith('urn:')) {
            nodeSet.add(triple.object)
            edgeList.push({
              source: triple.subject,
              target: triple.object,
              predicate: triple.predicate,
              label: getLocalName(triple.predicate),
            })
          }
        }
        
        setGraphNodes([...nodeSet].map(id => ({ id, label: getLocalName(id) })))
        setGraphEdges(edgeList)
        setViewMode(edgeList.length > 0 ? 'graph' : 'table')
      } else {
        setGraphNodes([])
        setGraphEdges([])
        setViewMode('table')
      }
      
      if (result.type === 'update') {
        await loadProjectStats(currentProject)
        await loadProjects()
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setExecuting(false)
    }
  }, [currentProject, sparqlQuery, buildGraphFromResults, loadProjectStats, loadProjects])
  
  const loadExampleDataset = useCallback(async (datasetId) => {
    if (!currentProject) throw new Error('No project selected')
    
    await fetchRepoJson(currentProject, `/load-example/${datasetId}`, {
      method: 'POST',
    })
    
    await loadProjectStats(currentProject)
    await loadProjects()
    await executeSparql('SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100')
  }, [currentProject, loadProjectStats, loadProjects, executeSparql])
  
  const loadCustomData = useCallback(async (sparqlQuery) => {
    if (!currentProject) throw new Error('No project selected')
    
    await fetchRepoJson(currentProject, '/sparql', {
      method: 'POST',
      body: JSON.stringify({ query: sparqlQuery }),
    })
    
    await loadProjectStats(currentProject)
    await loadProjects()
    await executeSparql('SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100')
  }, [currentProject, loadProjectStats, loadProjects, executeSparql])
  
  useEffect(() => {
    async function init() {
      try {
        setLoading(true)
        setError(null)
        setApiStatus('checking')
        
        await fetchJson('/health')
        setApiStatus('online')
        
        const repos = await loadProjects()
        
        try {
          const sourcesData = await fetchJson('/sources')
          setSources(sourcesData.sources || [])
        } catch (e) {
          console.warn('[App] Could not load sources:', e)
        }
        
        if (repos.length > 0) {
          setCurrentProject(repos[0].name)
        }
      } catch (err) {
        console.error('[App] Failed to initialize:', err)
        setApiStatus('offline')
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    
    init()
  }, [loadProjects])
  
  useEffect(() => {
    if (currentProject && apiStatus === 'online') {
      loadProjectStats(currentProject)
      executeSparql('SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100')
    } else {
      setGraphNodes([])
      setGraphEdges([])
      setQueryResults(null)
    }
  }, [currentProject, apiStatus, loadProjectStats])
  
  const handleNodeClick = useCallback(async (node) => {
    setSelectedNode(node.id)
    
    if (!currentProject) {
      setNodeTriples([])
      return
    }
    
    try {
      const triples = await fetchRepoJson(currentProject, `/triples?subject=${encodeURIComponent(node.id)}`)
      setNodeTriples(triples.triples)
    } catch (err) {
      console.error('Failed to load node triples:', err)
      setNodeTriples([])
    }
  }, [currentProject])
  
  const sampleQueries = [
    { label: '📊 Show All', query: 'SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100' },
    { label: '⭐ With Provenance', query: 'SELECT ?s ?p ?o ?source ?confidence WHERE { ?s ?p ?o } LIMIT 100' },
    { label: '🔗 Links Only', query: 'SELECT ?s ?p ?o WHERE { ?s ?p ?o . FILTER(isIRI(?o)) } LIMIT 100' },
    { label: '❓ ASK', query: 'ASK WHERE { ?s ?p ?o }' },
  ]
  
  if (apiStatus === 'offline') {
    return (
      <div className="app">
        <div className="api-offline">
          <div className="offline-icon"></div>
          <h2>API Server Not Running</h2>
          <p>The RDF-StarBase API server is not responding.</p>
          <div className="offline-instructions">
            <h3>To start the server:</h3>
            <code>cd e:\RDF-StarBase</code>
            <code>uvicorn rdf_starbase.repository_api:app --reload</code>
          </div>
          <button className="btn" onClick={() => window.location.reload()}>Retry Connection</button>
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
      <CreateProjectModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onCreate={createProject}
      />
      
      <LoadDataModal
        isOpen={showLoadDataModal}
        onClose={() => setShowLoadDataModal(false)}
        onLoadExample={loadExampleDataset}
        onLoadCustom={loadCustomData}
        currentProject={currentProject}
      />
      
      <header className="header">
        <h1>
          <svg className="logo" viewBox="0 0 24 24">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
          </svg>
          RDF-StarBase
        </h1>
        
        <div className="project-selector">
          <select
            value={currentProject || ''}
            onChange={(e) => setCurrentProject(e.target.value || null)}
            className="project-dropdown"
          >
            <option value="">Select a project...</option>
            {projects.map(p => (
              <option key={p.name} value={p.name}>
                {p.name} ({p.triple_count} triples)
              </option>
            ))}
          </select>
          <button className="btn btn-small" onClick={() => setShowCreateModal(true)} title="Create New Project">+ New</button>
          <button className="btn btn-small btn-secondary" onClick={() => setShowLoadDataModal(true)} title="Load Data" disabled={!currentProject}> Load</button>
        </div>
        
        {stats && currentProject && (
          <div className="stats-bar">
            <span>Triples: <span className="value">{stats.total_assertions || 0}</span></span>
            <span>Subjects: <span className="value">{stats.unique_subjects || 0}</span></span>
            <span>Sources: <span className="value">{stats.unique_sources || 0}</span></span>
          </div>
        )}
      </header>
      
      <div className="main-content">
        <aside className="sidebar">
          <div className="tabs">
            <button className={`tab ${sidebarTab === 'projects' ? 'active' : ''}`} onClick={() => setSidebarTab('projects')}>Projects</button>
            <button className={`tab ${sidebarTab === 'sources' ? 'active' : ''}`} onClick={() => setSidebarTab('sources')}>Sources</button>
          </div>
          
          {sidebarTab === 'projects' && (
            <div className="section">
              <div className="section-title">
                Your Projects
                <button className="btn btn-small" style={{ marginLeft: 'auto' }} onClick={() => setShowCreateModal(true)}>+ New</button>
              </div>
              <div className="project-list">
                {projects.length === 0 ? (
                  <div className="empty-state">
                    <p>No projects yet</p>
                    <button className="btn" onClick={() => setShowCreateModal(true)}>Create Your First Project</button>
                  </div>
                ) : (
                  projects.map(p => (
                    <div key={p.name} className={`project-item ${currentProject === p.name ? 'active' : ''}`} onClick={() => setCurrentProject(p.name)}>
                      <div className="project-name">{p.name}</div>
                      <div className="project-meta">{p.triple_count} triples</div>
                      <button className="delete-btn" onClick={(e) => { e.stopPropagation(); deleteProject(p.name) }} title="Delete project"></button>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
          
          {sidebarTab === 'sources' && (
            <div className="section">
              <div className="section-title">Registered Sources</div>
              <div className="source-list">
                {sources.map((source) => (
                  <div key={source.id} className="source-item">
                    <div className="name">{source.name}</div>
                    <div className="type">{source.source_type} | {source.status}</div>
                  </div>
                ))}
                {sources.length === 0 && <div style={{ color: 'rgba(255,255,255,0.5)' }}>No sources registered</div>}
              </div>
            </div>
          )}
        </aside>
        
        <main className="query-area">
          <div className="query-editor">
            <div className="query-header">
              <span className="query-label">SPARQL Query</span>
              <div className="quick-queries">
                {sampleQueries.map((sq, i) => (
                  <button key={i} className="btn btn-tiny" onClick={() => { setSparqlQuery(sq.query); executeSparql(sq.query) }} title={sq.query}>{sq.label}</button>
                ))}
              </div>
            </div>
            <textarea
              className="query-input"
              value={sparqlQuery}
              onChange={(e) => setSparqlQuery(e.target.value)}
              placeholder="Enter SPARQL query..."
              onKeyDown={(e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') executeSparql() }}
            />
            <div className="query-actions">
              <button className="btn btn-primary" onClick={() => executeSparql()} disabled={executing || !currentProject}>
                {executing ? ' Running...' : ' Run Query'}
              </button>
              <span className="query-hint">Ctrl+Enter to execute</span>
              {error && <span className="query-error">{error}</span>}
            </div>
          </div>
          
          <QueryResultsPanel
            queryResults={queryResults}
            viewMode={viewMode}
            setViewMode={setViewMode}
            graphNodes={graphNodes}
            graphEdges={graphEdges}
            selectedNode={selectedNode}
            onNodeClick={handleNodeClick}
            nodeTriples={nodeTriples}
            onCloseNodePanel={() => { setSelectedNode(null); setNodeTriples([]) }}
          />
        </main>
      </div>
    </div>
  )
}

export default App
