import { useState, useEffect, useCallback, useRef } from 'react'
import * as d3 from 'd3'
import SparqlEditor from './components/SparqlEditor'
import SchemaBrowser from './components/SchemaBrowser'
import ImportExport from './components/ImportExport'
import Dashboard from './components/Dashboard'
import Security from './components/Security'
import SQLExplorer from './components/SQLExplorer'
import AIGrounding from './components/AIGrounding'
import Starchart from './components/Starchart'
import { UserMenu, useAuth } from './components/Auth'
import {
  DatabaseIcon, PlayIcon, PlusIcon, TrashIcon, FolderIcon,
  TableIcon, NetworkIcon, CodeIcon, SunIcon, MoonIcon,
  SearchIcon, SettingsIcon, BookIcon, ZapIcon, GlobeIcon,
  ChevronDownIcon, CloseIcon, RefreshIcon, HomeIcon, ShieldIcon,
  TerminalIcon, BrainIcon, MapIcon, PaletteIcon, DownloadIcon, InfoIcon
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
  if (typeof uri !== 'string') return String(uri)
  if (uri.includes('#')) return uri.split('#').pop()
  return uri.split('/').pop()
}

// Check if a value is a URI (object property target)
const isURI = (value) => {
  if (!value || typeof value !== 'string') return false
  return value.startsWith('http://') || value.startsWith('https://') || value.startsWith('urn:')
}

// Format a value for display - handles typed literals
const formatValue = (value) => {
  if (!value) return ''
  if (typeof value !== 'string') return String(value)
  
  // If value is just an XSD datatype URI, it's likely a parsing error - show as "(empty)"
  if (value.startsWith('http://www.w3.org/2001/XMLSchema#')) {
    return `(${getLocalName(value)})`
  }
  
  // Handle typed literals like "2023-03-14"^^xsd:date or "2023-03-14"^^<http://...>
  if (value.includes('^^')) {
    const parts = value.split('^^')
    const val = parts[0]
    // Remove surrounding quotes and angle brackets from value
    return val.replace(/^["']|["']$/g, '')
  }
  
  // Handle language-tagged literals like "Hello"@en
  if (value.includes('@') && value.startsWith('"')) {
    const match = value.match(/^"(.+)"@(\w+)$/)
    if (match) return match[1]
  }
  
  // Remove surrounding quotes if present
  if ((value.startsWith('"') && value.endsWith('"')) || 
      (value.startsWith("'") && value.endsWith("'"))) {
    return value.slice(1, -1)
  }
  
  return value
}

// Default colors for node types (Neo4j-inspired palette)
const DEFAULT_TYPE_COLORS = [
  '#6366f1', // Indigo
  '#22c55e', // Green
  '#f59e0b', // Amber
  '#ec4899', // Pink
  '#06b6d4', // Cyan
  '#8b5cf6', // Violet
  '#ef4444', // Red
  '#14b8a6', // Teal
  '#f97316', // Orange
  '#84cc16', // Lime
]

// ============================================================================
// Graph Visualization Component
// ============================================================================
function GraphView({ nodes, edges, onNodeClick, onEdgeClick, theme, graphStyles, selectedNodeId }) {
  const svgRef = useRef(null)
  const [tooltip, setTooltip] = useState(null)
  
  // Get color for a node based on its types
  const getNodeColor = useCallback((node) => {
    if (!node.types || node.types.length === 0) {
      return theme === 'dark' ? '#313244' : '#e6e9ef' // Default color
    }
    
    // Check if any type has a custom color
    for (const type of node.types) {
      if (graphStyles?.nodeColors?.[type]) {
        return graphStyles.nodeColors[type]
      }
    }
    
    // Auto-assign color based on first type
    const firstType = node.types[0]
    const typeIndex = Object.keys(graphStyles?.nodeTypes || {})
      .flatMap(nid => graphStyles.nodeTypes[nid])
      .filter((v, i, a) => a.indexOf(v) === i) // unique types
      .indexOf(firstType)
    
    return DEFAULT_TYPE_COLORS[typeIndex % DEFAULT_TYPE_COLORS.length] || (theme === 'dark' ? '#313244' : '#e6e9ef')
  }, [theme, graphStyles])
  
  // Get label for a node based on style settings
  const getNodeLabel = useCallback((node) => {
    if (!node.types || node.types.length === 0) {
      return node.label || getLocalName(node.id)
    }
    
    // Check if any type has a custom label property
    for (const type of node.types) {
      const labelProp = graphStyles?.labelProperties?.[type]
      if (labelProp && node.data?.[labelProp]) {
        const val = node.data[labelProp]
        return val.length > 20 ? val.substring(0, 18) + '...' : val
      }
    }
    
    return node.label || getLocalName(node.id)
  }, [graphStyles])
  
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
    
    // KeyLines-style organic layout:
    // 1. Run simulation to completion (fast alpha decay)
    // 2. Fix all nodes in place
    // 3. Dragging moves node directly without re-simulating
    
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(edges).id(d => d.id).distance(180).strength(0.5))
      .force('charge', d3.forceManyBody().strength(-500).distanceMax(400))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(60))
      .alphaDecay(0.05)  // Faster decay = settles quicker
      .velocityDecay(0.4)  // Higher friction = less floaty
    
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
    
    // Track if simulation has settled
    let simulationSettled = false
    
    // Update positions function (used during simulation and after)
    const updatePositions = () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y)
      
      edgeLabels
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2 - 8)
      
      node.attr('transform', d => `translate(${d.x},${d.y})`)
    }
    
    // Nodes with KeyLines-style drag behavior
    const node = g.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'grab')
      .attr('class', d => d.id === selectedNodeId ? 'node selected' : 'node')
      .call(d3.drag()
        .on('start', (event, d) => {
          d3.select(event.sourceEvent.target.parentNode).attr('cursor', 'grabbing')
          // Fix position immediately
          d.fx = d.x
          d.fy = d.y
        })
        .on('drag', (event, d) => {
          // Direct position update - no simulation restart
          d.fx = event.x
          d.fy = event.y
          d.x = event.x
          d.y = event.y
          updatePositions()
        })
        .on('end', (event, d) => {
          d3.select(event.sourceEvent.target.parentNode).attr('cursor', 'grab')
          // Keep node fixed where it was dropped (KeyLines behavior)
          d.fx = d.x
          d.fy = d.y
        }))
      .on('click', (event, d) => onNodeClick && onNodeClick(d))
    
    node.append('circle')
      .attr('r', 20)
      .attr('fill', d => getNodeColor(d))
      .attr('stroke', d => d.id === selectedNodeId ? '#fff' : (isDark ? '#89b4fa' : '#1e66f5'))
      .attr('stroke-width', d => d.id === selectedNodeId ? 3 : 2)
    
    node.append('text')
      .attr('dy', 38)
      .attr('text-anchor', 'middle')
      .attr('fill', isDark ? '#cdd6f4' : '#4c4f69')
      .attr('font-size', '12px')
      .attr('font-weight', '500')
      .text(d => {
        const label = getNodeLabel(d)
        return label.length > 20 ? label.substring(0, 18) + '...' : label
      })
    
    // Simulation tick - only runs during initial layout
    simulation.on('tick', updatePositions)
    
    // When simulation ends, fix all nodes in place
    simulation.on('end', () => {
      simulationSettled = true
      // Fix all nodes where they landed
      nodes.forEach(d => {
        d.fx = d.x
        d.fy = d.y
      })
    })
    
    return () => simulation.stop()
  }, [nodes, edges, theme, onNodeClick, onEdgeClick, getNodeColor, getNodeLabel, selectedNodeId])
  
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
  const [reasoningLevel, setReasoningLevel] = useState('none')
  const [materializeOnLoad, setMaterializeOnLoad] = useState(false)
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
      await onCreate(name.trim(), description.trim(), reasoningLevel, materializeOnLoad)
      setName('')
      setDescription('')
      setReasoningLevel('none')
      setMaterializeOnLoad(false)
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
          <div className="form-group">
            <label>Reasoning Level</label>
            <select 
              value={reasoningLevel} 
              onChange={e => setReasoningLevel(e.target.value)}
              className="select-input"
            >
              <option value="none">None - No inference</option>
              <option value="rdfs">RDFS - Basic class/property inference</option>
              <option value="rdfs_plus">RDFS+ - RDFS plus basic OWL</option>
              <option value="owl_rl">OWL 2 RL - Full OWL 2 RL profile</option>
            </select>
            <small>Controls RDFS/OWL reasoning (like GraphDB)</small>
          </div>
          {reasoningLevel !== 'none' && (
            <div className="form-group checkbox-group">
              <label className="checkbox-label">
                <input 
                  type="checkbox" 
                  checked={materializeOnLoad}
                  onChange={e => setMaterializeOnLoad(e.target.checked)}
                />
                <span>Auto-run inference after data loads</span>
              </label>
            </div>
          )}
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
  
  // Navigation state
  const [activeTab, setActiveTab] = useState('dashboard') // 'dashboard' | 'workbench'
  
  // Repository state
  const [repositories, setRepositories] = useState([])
  const [currentRepo, setCurrentRepo] = useState(null)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showImportModal, setShowImportModal] = useState(false)
  const [stats, setStats] = useState(null)
  
  // Query state
  const [sparqlQuery, setSparqlQuery] = useState('SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100')
  const [queryResults, setQueryResults] = useState(null)
  const [executing, setExecuting] = useState(false)
  const [error, setError] = useState(null)
  
  // View state
  const [viewMode, setViewMode] = useState('table') // 'table' | 'graph' | 'json'
  const [sidePanel, setSidePanel] = useState('schema') // 'schema' | 'import' | 'details' | 'styling' | null
  const [graphNodes, setGraphNodes] = useState([])
  const [graphEdges, setGraphEdges] = useState([])
  const [selectedNode, setSelectedNode] = useState(null)
  const [selectedEdge, setSelectedEdge] = useState(null)
  const [nodeProperties, setNodeProperties] = useState(null)
  const [nodeAnnotations, setNodeAnnotations] = useState(null) // RDF-Star annotations
  const [detailsPanel, setDetailsPanel] = useState(false) // Right panel visibility
  const [expandedSections, setExpandedSections] = useState({ objectProps: true, dataProps: true, provenance: false, annotations: false, metadata: false, styling: true })
  const [expandedProps, setExpandedProps] = useState({}) // Track expanded property accordions for competing claims
  const [showQueryHelpers, setShowQueryHelpers] = useState(false) // Query helpers panel
  
  // Graph styling state (Neo4j-like)
  const [graphStyles, setGraphStyles] = useState(() => {
    const saved = localStorage.getItem('rdf-starbase-graph-styles')
    return saved ? JSON.parse(saved) : {
      nodeColors: {},      // { typeUri: '#hexcolor' }
      labelProperties: {}, // { typeUri: propertyUri }
      nodeTypes: {},       // { nodeId: [typeUris] } - populated from query results
      nodeData: {},        // { nodeId: { propUri: value } } - data properties for labels
    }
  })
  const [availableTypes, setAvailableTypes] = useState([]) // Types found in current graph
  
  // Persist graph styles
  useEffect(() => {
    localStorage.setItem('rdf-starbase-graph-styles', JSON.stringify({
      nodeColors: graphStyles.nodeColors,
      labelProperties: graphStyles.labelProperties,
    }))
  }, [graphStyles.nodeColors, graphStyles.labelProperties])
  
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
  const createRepository = useCallback(async (name, description, reasoningLevel = 'none', materializeOnLoad = false) => {
    await fetchJson('/repositories', {
      method: 'POST',
      body: JSON.stringify({ 
        name, 
        description, 
        tags: [],
        reasoning_level: reasoningLevel,
        materialize_on_load: materializeOnLoad,
      }),
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

  // Build graph from results - extracts types and data properties for styling
  const buildGraph = useCallback((results, columns) => {
    const nodeSet = new Set()
    const edgeList = []
    const nodeTypes = {} // { nodeId: Set<typeUri> }
    const nodeData = {}  // { nodeId: { propUri: value } }
    const RDF_TYPE = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
    const hasTriples = columns.includes('s') && columns.includes('p') && columns.includes('o')
    
    if (hasTriples) {
      for (const row of results) {
        const { s, p, o } = row
        if (!s || !p || !o) continue
        // Skip RDF-Star annotation triples (quoted triple subjects/objects)
        if (typeof s === 'string' && s.startsWith('<<')) continue
        if (typeof o === 'string' && o.startsWith('<<')) continue
        nodeSet.add(s)
        
        // Track rdf:type for styling
        if (p === RDF_TYPE || p.endsWith('#type') || p.endsWith('/type')) {
          if (!nodeTypes[s]) nodeTypes[s] = new Set()
          nodeTypes[s].add(o)
        }
        
        // Track data properties (non-URI objects) for label options
        if (typeof o === 'string' && !o.startsWith('http') && !o.startsWith('urn:')) {
          if (!nodeData[s]) nodeData[s] = {}
          nodeData[s][p] = o
        }
        
        if (typeof o === 'string' && (o.startsWith('http') || o.startsWith('urn:') || o.startsWith('mailto:'))) {
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
    
    // Convert Sets to Arrays for state
    const nodeTypesArray = {}
    const allTypes = new Set()
    for (const [nodeId, types] of Object.entries(nodeTypes)) {
      nodeTypesArray[nodeId] = [...types]
      types.forEach(t => allTypes.add(t))
    }
    
    // Update available types for styling panel
    setAvailableTypes([...allTypes])
    setGraphStyles(prev => ({
      ...prev,
      nodeTypes: nodeTypesArray,
      nodeData: nodeData,
    }))
    
    return {
      nodes: [...nodeSet].map(id => ({ 
        id, 
        label: getLocalName(id),
        types: nodeTypesArray[id] || [],
        data: nodeData[id] || {},
      })),
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
        // Don't auto-switch view - respect user's current view choice
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
  }, [currentRepo, sparqlQuery, buildGraph, loadStats, loadRepositories])

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

  // Handle node click - fetch properties with provenance and RDF-Star annotations
  const handleNodeClick = useCallback(async (node) => {
    setSelectedNode(node)
    setSelectedEdge(null)
    setDetailsPanel(true)
    setSidePanel('details') // Auto-switch to details tab
    setExpandedProps({})
    if (!currentRepo) return
    
    try {
      // Query for all properties of this node with RDF-Star provenance
      // Supports both internal provenance and W3C PROV vocabulary
      const propertiesQuery = `
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX rdfstar: <http://rdf-starbase.dev/>
        
        SELECT ?p ?o ?source ?confidence ?timestamp ?metaPred ?metaVal WHERE {
          <${node.id}> ?p ?o .
          OPTIONAL {
            << <${node.id}> ?p ?o >> rdfstar:source ?internalSource .
          }
          OPTIONAL {
            << <${node.id}> ?p ?o >> rdfstar:confidence ?internalConf .
          }
          OPTIONAL {
            << <${node.id}> ?p ?o >> prov:wasDerivedFrom ?provSource .
          }
          OPTIONAL {
            << <${node.id}> ?p ?o >> prov:value ?provValue .
          }
          OPTIONAL {
            << <${node.id}> ?p ?o >> prov:generatedAtTime ?provTime .
          }
          OPTIONAL {
            << <${node.id}> ?p ?o >> ?metaPred ?metaVal .
            FILTER(?metaPred != rdfstar:source && ?metaPred != rdfstar:confidence 
                   && ?metaPred != prov:wasDerivedFrom && ?metaPred != prov:value 
                   && ?metaPred != prov:generatedAtTime)
          }
          BIND(COALESCE(?internalSource, ?provSource) AS ?source)
          BIND(COALESCE(?internalConf, ?provValue) AS ?confidence)
          BIND(?provTime AS ?timestamp)
        }
      `
      const response = await fetchJson(`/repositories/${currentRepo}/sparql`, {
        method: 'POST',
        body: JSON.stringify({ query: propertiesQuery }),
      })
      
      // Group results by property, collecting all values and metadata
      const results = response.results || []
      const propMap = new Map()
      results.forEach(row => {
        const propKey = row.p
        const valueKey = `${row.p}|${row.o}`
        
        if (!propMap.has(propKey)) {
          propMap.set(propKey, {
            predicate: row.p,
            isObjectProperty: isURI(row.o),
            values: new Map()
          })
        }
        
        const prop = propMap.get(propKey)
        if (!prop.values.has(valueKey)) {
          prop.values.set(valueKey, {
            value: row.o,
            sources: [],
            hasProvenance: false
          })
        }
        
        const valEntry = prop.values.get(valueKey)
        if (row.source || row.confidence || row.timestamp || row.metaPred) {
          valEntry.hasProvenance = true
          const sourceEntry = {
            source: row.source,
            confidence: row.confidence,
            timestamp: row.timestamp,
            metadata: []
          }
          if (row.metaPred && row.metaVal) {
            sourceEntry.metadata.push({ predicate: row.metaPred, value: row.metaVal })
          }
          // Check if we already have this source
          const existingSource = valEntry.sources.find(s => 
            s.source === row.source && s.confidence === row.confidence && s.timestamp === row.timestamp
          )
          if (existingSource && row.metaPred) {
            existingSource.metadata.push({ predicate: row.metaPred, value: row.metaVal })
          } else if (!existingSource && (row.source || row.confidence || row.timestamp)) {
            valEntry.sources.push(sourceEntry)
          }
        }
      })
      
      // Convert to array format with competing claims detection
      const processedProps = Array.from(propMap.values()).map(prop => ({
        predicate: prop.predicate,
        isObjectProperty: prop.isObjectProperty,
        values: Array.from(prop.values.values()),
        hasCompetingClaims: prop.values.size > 1,
        hasProvenance: Array.from(prop.values.values()).some(v => v.hasProvenance)
      }))
      
      setNodeProperties(processedProps)
      
      // Query for RDF-Star annotations (statements about statements)
      const annotationsQuery = `
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX rdfstar: <http://rdf-starbase.dev/>
        
        SELECT ?innerS ?innerP ?innerO ?annPred ?annVal WHERE {
          << ?innerS ?innerP ?innerO >> ?annPred ?annVal .
          FILTER(?innerS = <${node.id}> || ?innerO = <${node.id}>)
        }
        LIMIT 100
      `
      try {
        const annResponse = await fetchJson(`/repositories/${currentRepo}/sparql`, {
          method: 'POST',
          body: JSON.stringify({ query: annotationsQuery }),
        })
        setNodeAnnotations(annResponse.results || [])
      } catch {
        setNodeAnnotations([])
      }
    } catch (err) {
      console.error('Failed to load node properties:', err)
      setNodeProperties([])
      setNodeAnnotations([])
    }
  }, [currentRepo])

  // Handle edge click
  const handleEdgeClick = useCallback(async (edge) => {
    setSelectedEdge(edge)
    setSelectedNode(null)
    setDetailsPanel(true)
    // Edge already contains source, target, predicate
    // Could query for provenance here if needed
  }, [])

  // Toggle accordion section
  const toggleSection = (section) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

  // Toggle property expansion (for competing claims)
  const togglePropExpand = (propKey) => {
    setExpandedProps(prev => ({ ...prev, [propKey]: !prev[propKey] }))
  }

  // Expand graph with a new node - fetch its connections and add to graph
  const expandNode = useCallback(async (nodeUri) => {
    if (!currentRepo) return
    
    // Check if node already exists in graph
    if (graphNodes.some(n => n.id === nodeUri)) {
      // Node exists, just highlight it briefly
      return
    }
    
    try {
      // Query for outgoing relationships from this node
      const query = `
        SELECT ?s ?p ?o WHERE {
          { <${nodeUri}> ?p ?o . BIND(<${nodeUri}> AS ?s) }
          UNION
          { ?s ?p <${nodeUri}> . BIND(<${nodeUri}> AS ?o) }
          FILTER(isIRI(?s) && isIRI(?o))
        }
        LIMIT 50
      `
      const response = await fetchJson(`/repositories/${currentRepo}/sparql`, {
        method: 'POST',
        body: JSON.stringify({ query }),
      })
      
      const results = response.results || []
      if (results.length === 0) return
      
      // Build fresh node set from existing nodes (use IDs only to avoid D3 mutation)
      const nodeIds = new Set(graphNodes.map(n => n.id))
      
      // Build fresh edges list - extract IDs from potentially mutated D3 objects
      const existingEdges = graphEdges.map(e => ({
        source: typeof e.source === 'object' ? e.source.id : e.source,
        target: typeof e.target === 'object' ? e.target.id : e.target,
        predicate: e.predicate,
        label: e.label
      }))
      
      const newEdges = [...existingEdges]
      
      for (const row of results) {
        // Add new nodes
        if (!nodeIds.has(row.s)) {
          nodeIds.add(row.s)
        }
        if (!nodeIds.has(row.o)) {
          nodeIds.add(row.o)
        }
        // Check if edge already exists (compare string IDs)
        const edgeExists = newEdges.some(e => 
          e.source === row.s && e.target === row.o && e.predicate === row.p
        )
        if (!edgeExists) {
          newEdges.push({ 
            source: row.s, 
            target: row.o, 
            predicate: row.p, 
            label: getLocalName(row.p) 
          })
        }
      }
      
      // Create fresh node objects (D3 will add x, y, etc.)
      const freshNodes = [...nodeIds].map(id => ({ id, label: getLocalName(id) }))
      
      // Update graph state with fresh arrays
      setGraphNodes(freshNodes)
      setGraphEdges(newEdges)
      
      // Update query results to include new triples
      if (queryResults && queryResults.results) {
        const existingKeys = new Set(queryResults.results.map(r => `${r.s}|${r.p}|${r.o}`))
        const newResults = [...queryResults.results]
        for (const row of results) {
          const key = `${row.s}|${row.p}|${row.o}`
          if (!existingKeys.has(key)) {
            newResults.push(row)
            existingKeys.add(key)
          }
        }
        setQueryResults({ ...queryResults, results: newResults })
      }
    } catch (err) {
      console.error('Failed to expand node:', err)
    }
  }, [currentRepo, graphNodes, graphEdges, queryResults])

  // Check if a URI is already in the graph
  const isNodeInGraph = useCallback((uri) => {
    return graphNodes.some(n => n.id === uri)
  }, [graphNodes])

  // Close details panel
  const closeDetailsPanel = () => {
    setDetailsPanel(false)
    setSelectedNode(null)
    setSelectedEdge(null)
    setNodeProperties(null)
    setNodeAnnotations(null)
  }

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

  // Comprehensive query helpers with categories
  const queryHelpers = [
    // Basic SPARQL
    { 
      category: 'basic', 
      name: 'Select All Triples', 
      description: 'Retrieve all triples in the graph with a limit',
      query: 'SELECT ?s ?p ?o\nWHERE {\n  ?s ?p ?o\n}\nLIMIT 100'
    },
    { 
      category: 'basic', 
      name: 'Distinct Predicates', 
      description: 'List all unique predicates (properties) in the graph',
      query: 'SELECT DISTINCT ?predicate\nWHERE {\n  ?s ?predicate ?o\n}\nORDER BY ?predicate'
    },
    { 
      category: 'basic', 
      name: 'Entity Properties', 
      description: 'Get all properties for a specific entity',
      query: '# Replace <entity_uri> with your entity\nSELECT ?predicate ?object\nWHERE {\n  <http://example.org/entity> ?predicate ?object\n}'
    },
    { 
      category: 'basic', 
      name: 'Filter by Pattern', 
      description: 'Find entities matching a name pattern',
      query: 'SELECT ?entity ?name\nWHERE {\n  ?entity <http://schema.org/name> ?name .\n  FILTER(CONTAINS(LCASE(?name), "example"))\n}'
    },
    // Analytics
    { 
      category: 'analytics', 
      name: 'Predicate Statistics', 
      description: 'Count occurrences of each predicate',
      query: 'SELECT ?predicate (COUNT(*) AS ?count)\nWHERE {\n  ?s ?predicate ?o\n}\nGROUP BY ?predicate\nORDER BY DESC(?count)'
    },
    { 
      category: 'analytics', 
      name: 'Class Distribution', 
      description: 'Count instances per class type',
      query: 'SELECT ?class (COUNT(?entity) AS ?count)\nWHERE {\n  ?entity a ?class\n}\nGROUP BY ?class\nORDER BY DESC(?count)'
    },
    { 
      category: 'analytics', 
      name: 'Top Connected Entities', 
      description: 'Find entities with most relationships',
      query: 'SELECT ?entity (COUNT(*) AS ?connections)\nWHERE {\n  { ?entity ?p ?o } UNION { ?s ?p ?entity }\n}\nGROUP BY ?entity\nORDER BY DESC(?connections)\nLIMIT 20'
    },
    // SPARQL-Star - Provenance
    { 
      category: 'sparql-star', 
      name: 'All Assertions with Provenance', 
      description: 'Find facts annotated with source and confidence using RDF-Star syntax',
      query: 'SELECT ?s ?p ?o ?source ?confidence\nWHERE {\n  <<?s ?p ?o>> <http://example.org/source> ?source ;\n               <http://example.org/confidence> ?confidence .\n}\nLIMIT 50'
    },
    { 
      category: 'sparql-star', 
      name: 'High Confidence Facts', 
      description: 'Filter facts by confidence score using SPARQL-Star',
      query: 'SELECT ?s ?p ?o ?confidence\nWHERE {\n  <<?s ?p ?o>> <http://example.org/confidence> ?confidence .\n  FILTER(?confidence >= 0.9)\n}\nORDER BY DESC(?confidence)'
    },
    { 
      category: 'sparql-star', 
      name: 'Facts by Source', 
      description: 'Find all facts from a specific data source',
      query: '# Replace "WikiData" with your source\nSELECT ?s ?p ?o\nWHERE {\n  <<?s ?p ?o>> <http://example.org/source> "WikiData" .\n}'
    },
    { 
      category: 'sparql-star', 
      name: 'Temporal Annotations', 
      description: 'Find facts with timestamp metadata',
      query: 'SELECT ?s ?p ?o ?timestamp\nWHERE {\n  <<?s ?p ?o>> <http://example.org/timestamp> ?timestamp .\n}\nORDER BY DESC(?timestamp)\nLIMIT 50'
    },
    { 
      category: 'sparql-star', 
      name: 'Annotated Relationships', 
      description: 'Find relationships with any annotations',
      query: 'SELECT ?s ?p ?o ?annotationPred ?annotationVal\nWHERE {\n  ?s ?p ?o .\n  FILTER(isIRI(?o))\n  <<?s ?p ?o>> ?annotationPred ?annotationVal .\n}\nLIMIT 50'
    },
    // Provenance extension filters
    { 
      category: 'provenance', 
      name: 'Filter by Source (Extension)', 
      description: 'Use FILTER_SOURCE extension to filter by data source',
      query: 'SELECT ?s ?p ?o\nWHERE {\n  ?s ?p ?o .\n  FILTER_SOURCE(?source = "IMDB")\n}\nLIMIT 50'
    },
    { 
      category: 'provenance', 
      name: 'Filter by Confidence (Extension)', 
      description: 'Use FILTER_CONFIDENCE extension for threshold filtering',
      query: 'SELECT ?s ?p ?o\nWHERE {\n  ?s ?p ?o .\n  FILTER_CONFIDENCE(?conf >= 0.8)\n}\nLIMIT 50'
    },
    { 
      category: 'provenance', 
      name: 'Compare Sources', 
      description: 'Find facts that exist in multiple sources',
      query: 'SELECT ?s ?p ?o (COUNT(DISTINCT ?source) AS ?sourceCount)\nWHERE {\n  <<?s ?p ?o>> <http://example.org/source> ?source .\n}\nGROUP BY ?s ?p ?o\nHAVING (COUNT(DISTINCT ?source) > 1)\nORDER BY DESC(?sourceCount)'
    },
    // Graph patterns
    { 
      category: 'patterns', 
      name: 'Two-Hop Paths', 
      description: 'Find entities connected through an intermediate node',
      query: 'SELECT ?start ?middle ?end\nWHERE {\n  ?start ?p1 ?middle .\n  ?middle ?p2 ?end .\n  FILTER(isIRI(?middle) && isIRI(?end))\n}\nLIMIT 50'
    },
    { 
      category: 'patterns', 
      name: 'Find Property Paths', 
      description: 'Use property paths to find transitive relationships',
      query: 'SELECT ?ancestor ?descendant\nWHERE {\n  ?descendant <http://www.w3.org/2000/01/rdf-schema#subClassOf>+ ?ancestor .\n}\nLIMIT 50'
    },
    { 
      category: 'patterns', 
      name: 'Optional Properties', 
      description: 'Include optional properties that may not exist',
      query: 'SELECT ?entity ?name ?description\nWHERE {\n  ?entity <http://schema.org/name> ?name .\n  OPTIONAL { ?entity <http://schema.org/description> ?description }\n}\nLIMIT 50'
    },
    // Existence checks
    { 
      category: 'existence', 
      name: 'ASK - Entity Exists', 
      description: 'Check if any entity has a specific name',
      query: 'ASK WHERE {\n  ?entity <http://schema.org/name> "Inception"\n}'
    },
    { 
      category: 'existence', 
      name: 'ASK - Relationship Exists', 
      description: 'Check if a specific relationship exists',
      query: 'ASK WHERE {\n  ?s <http://schema.org/director> ?o\n}'
    },
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

      {/* Import Modal */}
      {showImportModal && (
        <div className="modal-overlay" onClick={() => setShowImportModal(false)}>
          <div className="modal import-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Import Data</h2>
              <button className="close-btn" onClick={() => setShowImportModal(false)}>
                <CloseIcon size={20} />
              </button>
            </div>
            <div className="modal-body">
              <ImportExport
                repositoryName={currentRepo}
                onDataChanged={() => {
                  loadStats(currentRepo)
                  loadRepositories()
                  setShowImportModal(false)
                }}
                theme={theme}
              />
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <div className="logo">
            <DatabaseIcon size={24} />
            <span>RDF-StarBase</span>
          </div>
          
          {/* Tab Navigation */}
          <div className="app-tabs">
            <button 
              className={`tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
              onClick={() => setActiveTab('dashboard')}
            >
              <HomeIcon size={16} /> Dashboard
            </button>
            <button 
              className={`tab-btn ${activeTab === 'workbench' ? 'active' : ''}`}
              onClick={() => setActiveTab('workbench')}
            >
              <CodeIcon size={16} /> Workbench
            </button>
            <button 
              className={`tab-btn ${activeTab === 'security' ? 'active' : ''}`}
              onClick={() => setActiveTab('security')}
            >
              <ShieldIcon size={16} /> Security
            </button>
            <button 
              className={`tab-btn ${activeTab === 'sql' ? 'active' : ''}`}
              onClick={() => setActiveTab('sql')}
            >
              <TerminalIcon size={16} /> SQL
            </button>
            <button 
              className={`tab-btn ${activeTab === 'ai' ? 'active' : ''}`}
              onClick={() => setActiveTab('ai')}
            >
              <BrainIcon size={16} /> AI Grounding
            </button>
            <button 
              className={`tab-btn ${activeTab === 'mapper' ? 'active' : ''}`}
              onClick={() => setActiveTab('mapper')}
            >
              <MapIcon size={16} /> Mapper
            </button>
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
          <UserMenu />
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        {activeTab === 'dashboard' ? (
          <Dashboard 
            repositories={repositories}
            currentRepo={currentRepo}
            onNavigateToWorkbench={() => setActiveTab('workbench')}
            onRunQuery={(query) => {
              setActiveTab('workbench')
              setSparqlQuery(query)
              setTimeout(() => executeSparql(query), 100)
            }}
            onCreateRepo={() => setShowCreateModal(true)}
            onSelectRepo={setCurrentRepo}
            onRefreshRepos={loadRepositories}
            onOpenImport={() => setShowImportModal(true)}
            theme={theme}
          />
        ) : activeTab === 'security' ? (
          <Security theme={theme} />
        ) : activeTab === 'sql' ? (
          <SQLExplorer theme={theme} currentRepo={currentRepo} />
        ) : activeTab === 'ai' ? (
          <AIGrounding theme={theme} currentRepo={currentRepo} />
        ) : activeTab === 'mapper' ? (
          <Starchart theme={theme} currentRepo={currentRepo} />
        ) : (
        <>
        {/* Query Panel */}
        <div className="query-panel">
          <div className="query-toolbar">
            <button 
              className={`btn ${showQueryHelpers ? 'primary' : 'secondary'}`}
              onClick={() => setShowQueryHelpers(!showQueryHelpers)}
            >
              <BookIcon size={16} />
              Query Templates
              <ChevronDownIcon size={14} style={{ transform: showQueryHelpers ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
            </button>
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

          {/* Query Helpers Panel */}
          {showQueryHelpers && (
            <div className="query-helpers-panel">
              <div className="helpers-header">
                <h3><BookIcon size={16} /> SPARQL Query Templates</h3>
                <p className="helpers-intro">
                  Click a template to load it into the editor. Includes standard SPARQL and RDF-Star extensions for provenance.
                </p>
              </div>
              <div className="helpers-grid">
                {queryHelpers.map((helper, i) => (
                  <div 
                    key={i} 
                    className="query-helper-card"
                    onClick={() => { setSparqlQuery(helper.query); setShowQueryHelpers(false) }}
                  >
                    <div className="helper-header">
                      <span className="helper-name">{helper.name}</span>
                      <span className={`helper-category cat-${helper.category}`}>
                        {helper.category}
                      </span>
                    </div>
                    <p className="helper-description">{helper.description}</p>
                    <code className="helper-preview">{helper.query.split('\n')[0]}...</code>
                  </div>
                ))}
              </div>
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
              {viewMode === 'graph' && (
                <>
                  <button 
                    className={`icon-btn ${sidePanel === 'details' ? 'active' : ''}`}
                    onClick={() => setSidePanel(s => s === 'details' ? null : 'details')}
                    title="Node Details"
                  >
                    <InfoIcon size={18} />
                  </button>
                  <button 
                    className={`icon-btn ${sidePanel === 'styling' ? 'active' : ''}`}
                    onClick={() => setSidePanel(s => s === 'styling' ? null : 'styling')}
                    title="Node Styling"
                  >
                    <PaletteIcon size={18} />
                  </button>
                </>
              )}
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
                    graphStyles={graphStyles}
                    selectedNodeId={selectedNode?.id}
                  />
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

            {/* Side Panel with Tabs */}
            {sidePanel && (
              <div className="side-panel">
                <div className="side-panel-tabs">
                  <button 
                    className={`side-tab ${sidePanel === 'schema' ? 'active' : ''}`}
                    onClick={() => setSidePanel('schema')}
                    title="Schema Browser"
                  >
                    <BookIcon size={14} /> Schema
                  </button>
                  <button 
                    className={`side-tab ${sidePanel === 'import' ? 'active' : ''}`}
                    onClick={() => setSidePanel('import')}
                    title="Import / Export"
                  >
                    <DownloadIcon size={14} /> Import
                  </button>
                  {viewMode === 'graph' && (
                    <>
                      <button 
                        className={`side-tab ${sidePanel === 'details' ? 'active' : ''}`}
                        onClick={() => setSidePanel('details')}
                        title="Node Details"
                      >
                        <InfoIcon size={14} /> Details
                      </button>
                      <button 
                        className={`side-tab ${sidePanel === 'styling' ? 'active' : ''}`}
                        onClick={() => setSidePanel('styling')}
                        title="Node Styling"
                      >
                        <PaletteIcon size={14} /> Style
                      </button>
                    </>
                  )}
                  <button 
                    className="side-tab close-tab"
                    onClick={() => setSidePanel(null)}
                    title="Close Panel"
                  >
                    ×
                  </button>
                </div>
                
                <div className="side-panel-content">
                  {/* Schema Browser Tab */}
                  {sidePanel === 'schema' && (
                    <SchemaBrowser 
                      repositoryName={currentRepo} 
                      onInsert={handleSchemaInsert}
                      theme={theme}
                    />
                  )}
                  
                  {/* Import/Export Tab */}
                  {sidePanel === 'import' && (
                    <ImportExport 
                      repositoryName={currentRepo}
                      onDataChanged={() => { loadStats(currentRepo); loadRepositories() }}
                      theme={theme}
                    />
                  )}
                  
                  {/* Node Details Tab (Graph view only) */}
                  {sidePanel === 'details' && viewMode === 'graph' && (
                    <div className="graph-details-tab">
                      {selectedNode ? (
                        <>
                          <div className="detail-uri">{getLocalName(selectedNode.id)}</div>
                          <div className="detail-full-uri">{selectedNode.id}</div>
                          
                          {selectedNode.types?.length > 0 && (
                            <div className="node-types">
                              {selectedNode.types.map((t, i) => (
                                <span 
                                  key={i} 
                                  className="type-badge"
                                  style={{ backgroundColor: graphStyles.nodeColors[t] || DEFAULT_TYPE_COLORS[i % DEFAULT_TYPE_COLORS.length] }}
                                >
                                  {getLocalName(t)}
                                </span>
                              ))}
                            </div>
                          )}
                          
                          {/* Object Properties Accordion */}
                          {(() => {
                            const objectProps = nodeProperties?.filter(p => p.isObjectProperty) || []
                            return objectProps.length > 0 && (
                              <div className="accordion-section">
                                <button 
                                  className={`accordion-header ${expandedSections.objectProps ? 'expanded' : ''}`}
                                  onClick={() => toggleSection('objectProps')}
                                >
                                  <span>Object Properties</span>
                                  <span className="accordion-count">{objectProps.length}</span>
                                  <span className="accordion-icon">{expandedSections.objectProps ? '−' : '+'}</span>
                                </button>
                                {expandedSections.objectProps && (
                                  <div className="accordion-content">
                                    <div className="properties-list">
                                      {objectProps.map((prop, i) => (
                                        <div key={i} className={`property-item ${prop.hasCompetingClaims || prop.hasProvenance ? 'has-details' : ''}`}>
                                          <div 
                                            className="property-header"
                                            onClick={() => (prop.hasCompetingClaims || prop.hasProvenance) && togglePropExpand(prop.predicate)}
                                          >
                                            <span className="prop-name" title={prop.predicate}>{getLocalName(prop.predicate)}</span>
                                            <span className="prop-indicators">
                                              {prop.hasProvenance && <span className="indicator prov-indicator" title="Has provenance">P</span>}
                                              {prop.hasCompetingClaims && <span className="indicator claims-indicator" title="Competing claims">C</span>}
                                            </span>
                                            {(prop.hasCompetingClaims || prop.hasProvenance) && (
                                              <span className="prop-expand">{expandedProps[prop.predicate] ? '−' : '+'}</span>
                                            )}
                                          </div>
                                          <div className="property-values">
                                            {prop.values.map((val, j) => (
                                              <div key={j} className="prop-value-row">
                                                <span className="prop-value" title={val.value}>
                                                  {getLocalName(val.value)}
                                                </span>
                                                {isURI(val.value) && !isNodeInGraph(val.value) && (
                                                  <button 
                                                    className="expand-btn"
                                                    onClick={(e) => { e.stopPropagation(); expandNode(val.value); }}
                                                    title="Add to graph"
                                                  >+</button>
                                                )}
                                                {isURI(val.value) && isNodeInGraph(val.value) && (
                                                  <span className="in-graph-indicator" title="In graph">●</span>
                                                )}
                                              </div>
                                            ))}
                                          </div>
                                          {expandedProps[prop.predicate] && (
                                            <div className="property-details">
                                              {prop.values.map((val, j) => (
                                                val.sources.length > 0 && (
                                                  <div key={j} className="value-provenance">
                                                    <div className="val-header">{formatValue(val.value)}</div>
                                                    {val.sources.map((src, k) => (
                                                      <div key={k} className="source-info">
                                                        {src.source && <div className="src-row"><span className="src-label">Source:</span> <span className="src-val">{getLocalName(src.source)}</span></div>}
                                                        {src.confidence && <div className="src-row"><span className="src-label">Confidence:</span> <span className="src-val">{formatValue(src.confidence)}</span></div>}
                                                        {src.timestamp && <div className="src-row"><span className="src-label">Time:</span> <span className="src-val">{new Date(src.timestamp).toLocaleString()}</span></div>}
                                                        {src.metadata?.map((m, l) => (
                                                          <div key={l} className="src-row"><span className="src-label">{getLocalName(m.predicate)}:</span> <span className="src-val">{formatValue(m.value)}</span></div>
                                                        ))}
                                                      </div>
                                                    ))}
                                                  </div>
                                                )
                                              ))}
                                            </div>
                                          )}
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            )
                          })()}

                          {/* Data Properties Accordion */}
                          {(() => {
                            const dataProps = nodeProperties?.filter(p => !p.isObjectProperty) || []
                            return dataProps.length > 0 && (
                              <div className="accordion-section">
                                <button 
                                  className={`accordion-header ${expandedSections.dataProps ? 'expanded' : ''}`}
                                  onClick={() => toggleSection('dataProps')}
                                >
                                  <span>Data Properties</span>
                                  <span className="accordion-count">{dataProps.length}</span>
                                  <span className="accordion-icon">{expandedSections.dataProps ? '−' : '+'}</span>
                                </button>
                                {expandedSections.dataProps && (
                                  <div className="accordion-content">
                                    <div className="properties-list">
                                      {dataProps.map((prop, i) => (
                                        <div key={i} className={`property-item ${prop.hasCompetingClaims || prop.hasProvenance ? 'has-details' : ''}`}>
                                          <div 
                                            className="property-header"
                                            onClick={() => (prop.hasCompetingClaims || prop.hasProvenance) && togglePropExpand(prop.predicate)}
                                          >
                                            <span className="prop-name" title={prop.predicate}>{getLocalName(prop.predicate)}</span>
                                            <span className="prop-indicators">
                                              {prop.hasProvenance && <span className="indicator prov-indicator" title="Has provenance">P</span>}
                                              {prop.hasCompetingClaims && <span className="indicator claims-indicator" title="Competing claims">C</span>}
                                            </span>
                                            {(prop.hasCompetingClaims || prop.hasProvenance) && (
                                              <span className="prop-expand">{expandedProps[prop.predicate] ? '−' : '+'}</span>
                                            )}
                                          </div>
                                          <div className="property-values">
                                            {prop.values.map((val, j) => (
                                              <div key={j} className="prop-value-row">
                                                <span className="prop-value literal-value" title={val.value}>
                                                  {formatValue(val.value)}
                                                </span>
                                              </div>
                                            ))}
                                          </div>
                                          {expandedProps[prop.predicate] && (
                                            <div className="property-details">
                                              {prop.values.map((val, j) => (
                                                val.sources.length > 0 && (
                                                  <div key={j} className="value-provenance">
                                                    <div className="val-header">"{formatValue(val.value)}"</div>
                                                    {val.sources.map((src, k) => (
                                                      <div key={k} className="source-info">
                                                        {src.source && <div className="src-row"><span className="src-label">Source:</span> <span className="src-val">{getLocalName(src.source)}</span></div>}
                                                        {src.confidence && <div className="src-row"><span className="src-label">Confidence:</span> <span className="src-val">{formatValue(src.confidence)}</span></div>}
                                                        {src.timestamp && <div className="src-row"><span className="src-label">Time:</span> <span className="src-val">{new Date(src.timestamp).toLocaleString()}</span></div>}
                                                        {src.metadata?.map((m, l) => (
                                                          <div key={l} className="src-row"><span className="src-label">{getLocalName(m.predicate)}:</span> <span className="src-val">{formatValue(m.value)}</span></div>
                                                        ))}
                                                      </div>
                                                    ))}
                                                  </div>
                                                )
                                              ))}
                                            </div>
                                          )}
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            )
                          })()}

                          {/* RDF-Star Annotations Accordion */}
                          <div className="accordion-section">
                            <button 
                              className={`accordion-header ${expandedSections.annotations ? 'expanded' : ''}`}
                              onClick={() => toggleSection('annotations')}
                            >
                              <span>RDF-Star Annotations</span>
                              <span className="accordion-count">{nodeAnnotations?.length || 0}</span>
                              <span className="accordion-icon">{expandedSections.annotations ? '−' : '+'}</span>
                            </button>
                            {expandedSections.annotations && (
                              <div className="accordion-content">
                                {nodeAnnotations && nodeAnnotations.length > 0 ? (
                                  <div className="annotations-list">
                                    {nodeAnnotations.map((ann, i) => (
                                      <div key={i} className="annotation-item">
                                        <div className="annotation-triple">
                                          &lt;&lt; {getLocalName(ann.innerS)} {getLocalName(ann.innerP)} {getLocalName(ann.innerO)} &gt;&gt;
                                        </div>
                                        <div className="annotation-meta">
                                          <span className="ann-pred">{getLocalName(ann.annPred)}</span>
                                          <span className="ann-val">{formatValue(ann.annVal)}</span>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                ) : (
                                  <p className="no-data">No RDF-Star annotations</p>
                                )}
                              </div>
                            )}
                          </div>

                          {(!nodeProperties || nodeProperties.length === 0) && (
                            <p className="no-data">No properties found</p>
                          )}
                        </>
                      ) : selectedEdge ? (
                        <>
                          <div className="detail-uri">Edge: {getLocalName(selectedEdge.predicate)}</div>
                          <div className="detail-full-uri">{selectedEdge.predicate}</div>
                          <div className="edge-detail">
                            <span className="edge-label">Source</span>
                            <span className="edge-value">{getLocalName(selectedEdge.source?.id || selectedEdge.source)}</span>
                          </div>
                          <div className="edge-detail">
                            <span className="edge-label">Target</span>
                            <span className="edge-value">{getLocalName(selectedEdge.target?.id || selectedEdge.target)}</span>
                          </div>
                        </>
                      ) : (
                        <div className="empty-tab-content">
                          <p>Click a node or edge to see details</p>
                        </div>
                      )}
                    </div>
                  )}
                  
                  {/* Node Styling Tab (Graph view only) */}
                  {sidePanel === 'styling' && viewMode === 'graph' && (
                    <div className="graph-styling-tab">
                      <div className="styling-section">
                        <h4>Node Types</h4>
                        {availableTypes.length > 0 ? (
                          <div className="type-styles-list">
                            {availableTypes.map((typeUri, i) => {
                              // Get unique data properties from nodes of this type
                              const availableProps = [...new Set(
                                Object.entries(graphStyles.nodeData || {})
                                  .filter(([nodeId]) => graphStyles.nodeTypes[nodeId]?.includes(typeUri))
                                  .flatMap(([_, data]) => Object.keys(data))
                              )]
                              
                              return (
                                <div key={typeUri} className="type-style-row">
                                  <div className="type-style-header">
                                    <input 
                                      type="color" 
                                      className="type-color-input"
                                      value={graphStyles.nodeColors[typeUri] || DEFAULT_TYPE_COLORS[i % DEFAULT_TYPE_COLORS.length]}
                                      onChange={(e) => setGraphStyles(prev => ({
                                        ...prev,
                                        nodeColors: { ...prev.nodeColors, [typeUri]: e.target.value }
                                      }))}
                                      title="Change color"
                                    />
                                    <span className="type-name" title={typeUri}>{getLocalName(typeUri)}</span>
                                  </div>
                                  <div className="type-label-select">
                                    <span className="label-text">Label:</span>
                                    <select
                                      value={graphStyles.labelProperties[typeUri] || ''}
                                      onChange={(e) => setGraphStyles(prev => ({
                                        ...prev,
                                        labelProperties: { ...prev.labelProperties, [typeUri]: e.target.value }
                                      }))}
                                    >
                                      <option value="">URI (default)</option>
                                      {availableProps.map(prop => (
                                        <option key={prop} value={prop}>{getLocalName(prop)}</option>
                                      ))}
                                    </select>
                                  </div>
                                </div>
                              )
                            })}
                          </div>
                        ) : (
                          <p className="no-data">
                            No types found. Run a query that includes rdf:type triples to enable styling.
                          </p>
                        )}
                      </div>
                      
                      <div className="styling-actions">
                        <button 
                          className="btn secondary small"
                          onClick={() => setGraphStyles(prev => ({
                            ...prev,
                            nodeColors: {},
                            labelProperties: {},
                          }))}
                        >
                          Reset All Styles
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
        </>
        )}
      </main>
    </div>
  )
}

export default App
