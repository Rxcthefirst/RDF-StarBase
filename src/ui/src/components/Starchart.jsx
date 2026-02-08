// ============================================================================
// Starchart — Visual RML/R2RML Mapping Editor with Smart Recommendations
// ============================================================================

import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import cytoscape from 'cytoscape'
import {
  DatabaseIcon, TableIcon, FileIcon, PlusIcon, TrashIcon,
  PlayIcon, DownloadIcon, UploadIcon, RefreshIcon, SearchIcon,
  ChevronDownIcon, ChevronRightIcon, LinkIcon, SettingsIcon,
  CheckIcon, AlertIcon, CloseIcon, GripIcon, NetworkIcon
} from './Icons'
import './Starchart.css'

// ============================================================================
// Sample Ontology (will be loaded from API in production)
// ============================================================================

const SAMPLE_ONTOLOGY = {
  classes: [
    { uri: 'foaf:Person', label: 'Person', description: 'A person', 
      properties: ['foaf:name', 'foaf:givenName', 'foaf:familyName', 'foaf:mbox', 'foaf:phone', 'foaf:age', 'foaf:knows', 'foaf:homepage'] },
    { uri: 'foaf:Organization', label: 'Organization', description: 'An organization',
      properties: ['foaf:name', 'foaf:homepage', 'schema:address', 'schema:telephone'] },
    { uri: 'schema:Product', label: 'Product', description: 'A product or service',
      properties: ['schema:name', 'schema:description', 'schema:price', 'schema:sku', 'schema:manufacturer', 'schema:category'] },
    { uri: 'schema:Event', label: 'Event', description: 'An event',
      properties: ['schema:name', 'schema:description', 'schema:startDate', 'schema:endDate', 'schema:location', 'schema:organizer'] },
    { uri: 'schema:Place', label: 'Place', description: 'A location',
      properties: ['schema:name', 'schema:address', 'schema:geo', 'schema:telephone'] },
    { uri: 'schema:PostalAddress', label: 'PostalAddress', description: 'A mailing address',
      properties: ['schema:streetAddress', 'schema:addressLocality', 'schema:addressRegion', 'schema:postalCode', 'schema:addressCountry'] },
  ],
  properties: [
    // Person properties
    { uri: 'foaf:name', label: 'name', aliases: ['full name', 'fullname', 'display name'], range: 'xsd:string', description: 'Full name' },
    { uri: 'foaf:givenName', label: 'givenName', aliases: ['first name', 'firstname', 'given', 'first_name'], range: 'xsd:string', description: 'First/given name' },
    { uri: 'foaf:familyName', label: 'familyName', aliases: ['last name', 'lastname', 'surname', 'family', 'last_name'], range: 'xsd:string', description: 'Last/family name' },
    { uri: 'foaf:mbox', label: 'mbox', aliases: ['email', 'e-mail', 'mail', 'email address', 'e_mail'], range: 'xsd:anyURI', description: 'Email address' },
    { uri: 'foaf:phone', label: 'phone', aliases: ['telephone', 'tel', 'phone number', 'mobile', 'cell'], range: 'xsd:string', description: 'Phone number' },
    { uri: 'foaf:age', label: 'age', aliases: ['years old'], range: 'xsd:integer', description: 'Age in years' },
    { uri: 'foaf:knows', label: 'knows', aliases: ['friend', 'contact', 'connection'], range: 'foaf:Person', description: 'A person known by this person' },
    { uri: 'foaf:homepage', label: 'homepage', aliases: ['website', 'url', 'web', 'site'], range: 'xsd:anyURI', description: 'Homepage URL' },
    // Product properties  
    { uri: 'schema:name', label: 'name', aliases: ['title', 'product name', 'item name', 'product_name'], range: 'xsd:string', description: 'Name or title' },
    { uri: 'schema:description', label: 'description', aliases: ['desc', 'summary', 'details', 'about'], range: 'xsd:string', description: 'Description text' },
    { uri: 'schema:price', label: 'price', aliases: ['cost', 'amount', 'value', 'msrp'], range: 'xsd:decimal', description: 'Price amount' },
    { uri: 'schema:sku', label: 'sku', aliases: ['product id', 'item number', 'part number', 'upc', 'ean', 'product_id'], range: 'xsd:string', description: 'Stock keeping unit' },
    { uri: 'schema:manufacturer', label: 'manufacturer', aliases: ['brand', 'maker', 'vendor', 'company'], range: 'foaf:Organization', description: 'Manufacturer' },
    { uri: 'schema:category', label: 'category', aliases: ['type', 'classification', 'department'], range: 'xsd:string', description: 'Category or type' },
    // Event properties
    { uri: 'schema:startDate', label: 'startDate', aliases: ['start', 'begin', 'from', 'start time', 'begins'], range: 'xsd:dateTime', description: 'Start date/time' },
    { uri: 'schema:endDate', label: 'endDate', aliases: ['end', 'finish', 'to', 'end time', 'ends'], range: 'xsd:dateTime', description: 'End date/time' },
    { uri: 'schema:location', label: 'location', aliases: ['place', 'venue', 'where', 'address'], range: 'schema:Place', description: 'Location' },
    { uri: 'schema:organizer', label: 'organizer', aliases: ['host', 'organized by', 'coordinator'], range: 'foaf:Person', description: 'Event organizer' },
    // Address properties
    { uri: 'schema:streetAddress', label: 'streetAddress', aliases: ['street', 'address line', 'address1', 'addr'], range: 'xsd:string', description: 'Street address' },
    { uri: 'schema:addressLocality', label: 'addressLocality', aliases: ['city', 'town', 'locality'], range: 'xsd:string', description: 'City/town' },
    { uri: 'schema:addressRegion', label: 'addressRegion', aliases: ['state', 'province', 'region', 'county'], range: 'xsd:string', description: 'State/province' },
    { uri: 'schema:postalCode', label: 'postalCode', aliases: ['zip', 'zipcode', 'zip code', 'postcode'], range: 'xsd:string', description: 'Postal/ZIP code' },
    { uri: 'schema:addressCountry', label: 'addressCountry', aliases: ['country', 'nation'], range: 'xsd:string', description: 'Country' },
    { uri: 'schema:address', label: 'address', aliases: ['full address', 'mailing address'], range: 'schema:PostalAddress', description: 'Full address' },
    { uri: 'schema:telephone', label: 'telephone', aliases: ['phone', 'tel', 'phone number'], range: 'xsd:string', description: 'Telephone number' },
    { uri: 'schema:geo', label: 'geo', aliases: ['coordinates', 'lat/long', 'gps'], range: 'schema:GeoCoordinates', description: 'Geographic coordinates' },
  ],
  prefixes: {
    'foaf': 'http://xmlns.com/foaf/0.1/',
    'schema': 'http://schema.org/',
    'xsd': 'http://www.w3.org/2001/XMLSchema#',
    'rr': 'http://www.w3.org/ns/r2rml#',
    'rml': 'http://semweb.mmlab.be/ns/rml#',
    'skos': 'http://www.w3.org/2004/02/skos/core#',
  }
}

// ============================================================================
// String Similarity (Simulated Embeddings)
// ============================================================================

/**
 * Calculate similarity between two strings (0-1)
 * Uses a combination of techniques to simulate embedding similarity
 */
function calculateSimilarity(str1, str2) {
  const s1 = str1.toLowerCase().replace(/[_-]/g, ' ').trim()
  const s2 = str2.toLowerCase().replace(/[_-]/g, ' ').trim()
  
  // Exact match
  if (s1 === s2) return 1.0
  
  // Contains match
  if (s1.includes(s2) || s2.includes(s1)) return 0.85
  
  // Word overlap (Jaccard-like)
  const words1 = new Set(s1.split(/\s+/))
  const words2 = new Set(s2.split(/\s+/))
  const intersection = [...words1].filter(w => words2.has(w))
  const union = new Set([...words1, ...words2])
  const jaccard = intersection.length / union.size
  
  // Levenshtein-based similarity
  const maxLen = Math.max(s1.length, s2.length)
  const editDist = levenshteinDistance(s1, s2)
  const levenshteinSim = 1 - (editDist / maxLen)
  
  // Combine scores
  return Math.max(jaccard * 0.7 + levenshteinSim * 0.3, levenshteinSim * 0.5)
}

function levenshteinDistance(s1, s2) {
  const m = s1.length, n = s2.length
  const dp = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0))
  
  for (let i = 0; i <= m; i++) dp[i][0] = i
  for (let j = 0; j <= n; j++) dp[0][j] = j
  
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (s1[i-1] === s2[j-1]) {
        dp[i][j] = dp[i-1][j-1]
      } else {
        dp[i][j] = 1 + Math.min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
      }
    }
  }
  return dp[m][n]
}

/**
 * Get property recommendations for a column header
 */
function getRecommendations(columnHeader, ontology, topK = 5) {
  const scores = []
  
  for (const prop of ontology.properties) {
    // Check similarity against label
    let maxScore = calculateSimilarity(columnHeader, prop.label)
    
    // Check against aliases
    for (const alias of (prop.aliases || [])) {
      const aliasScore = calculateSimilarity(columnHeader, alias)
      maxScore = Math.max(maxScore, aliasScore)
    }
    
    // Check against URI local name
    const localName = prop.uri.split(':')[1]
    const uriScore = calculateSimilarity(columnHeader, localName)
    maxScore = Math.max(maxScore, uriScore)
    
    scores.push({
      property: prop,
      score: maxScore,
      confidence: maxScore >= 0.8 ? 'high' : maxScore >= 0.5 ? 'medium' : 'low'
    })
  }
  
  return scores
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
}

// ============================================================================
// CSV Parser
// ============================================================================

function parseCSV(text) {
  const lines = text.trim().split('\n')
  if (lines.length === 0) return { headers: [], rows: [] }
  
  // Simple CSV parsing (handles basic cases)
  const parseRow = (line) => {
    const result = []
    let current = ''
    let inQuotes = false
    
    for (let i = 0; i < line.length; i++) {
      const char = line[i]
      if (char === '"') {
        inQuotes = !inQuotes
      } else if (char === ',' && !inQuotes) {
        result.push(current.trim())
        current = ''
      } else {
        current += char
      }
    }
    result.push(current.trim())
    return result
  }
  
  const headers = parseRow(lines[0])
  const rows = lines.slice(1, 11).map(parseRow) // First 10 data rows
  
  return { headers, rows, totalRows: lines.length - 1 }
}

// ============================================================================
// Step 1: File Upload Component
// ============================================================================

function FileUploadStep({ onFileLoaded }) {
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleFile = useCallback((file) => {
    if (!file) return
    
    if (!file.name.endsWith('.csv')) {
      setError('Please upload a CSV file')
      return
    }
    
    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const parsed = parseCSV(e.target.result)
        if (parsed.headers.length === 0) {
          setError('CSV file appears to be empty')
          return
        }
        onFileLoaded({
          name: file.name,
          ...parsed
        })
      } catch (err) {
        setError('Failed to parse CSV: ' + err.message)
      }
    }
    reader.onerror = () => setError('Failed to read file')
    reader.readAsText(file)
  }, [onFileLoaded])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }, [handleFile])

  return (
    <div className="upload-step">
      <div 
        className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Upload CSV file"
        onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current?.click()}
      >
        <UploadIcon size={48} />
        <h3>Upload Your Data</h3>
        <p>Drag and drop a CSV file here, or click to browse</p>
        <p className="hint">We'll analyze the columns and help you map them to your ontology</p>
        
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={(e) => handleFile(e.target.files[0])}
          style={{ display: 'none' }}
        />
      </div>
      
      {error && (
        <div className="error-message" role="alert">
          <AlertIcon size={16} />
          {error}
          <button className="btn-icon" onClick={() => setError(null)}>
            <CloseIcon size={14} />
          </button>
        </div>
      )}
      
      <div className="sample-files">
        <p>Or try a sample:</p>
        <button 
          className="btn btn-sm btn-secondary"
          onClick={() => onFileLoaded({
            name: 'sample_people.csv',
            headers: ['id', 'first_name', 'last_name', 'email', 'phone', 'city', 'country'],
            rows: [
              ['1', 'John', 'Doe', 'john.doe@example.com', '+1-555-1234', 'New York', 'USA'],
              ['2', 'Jane', 'Smith', 'jane.smith@example.com', '+1-555-5678', 'Los Angeles', 'USA'],
              ['3', 'Bob', 'Johnson', 'bob.j@example.com', '+1-555-9012', 'Chicago', 'USA'],
              ['4', 'Alice', 'Williams', 'alice.w@example.com', '+44-20-1234', 'London', 'UK'],
              ['5', 'Charlie', 'Brown', 'charlie.b@example.com', '+1-555-3456', 'Boston', 'USA'],
            ],
            totalRows: 5
          })}
        >
          People Data
        </button>
        <button 
          className="btn btn-sm btn-secondary"
          onClick={() => onFileLoaded({
            name: 'sample_products.csv',
            headers: ['sku', 'product_name', 'description', 'price', 'category', 'manufacturer'],
            rows: [
              ['SKU001', 'Wireless Mouse', 'Ergonomic wireless mouse', '29.99', 'Electronics', 'TechCorp'],
              ['SKU002', 'USB-C Hub', '7-port USB-C hub', '49.99', 'Electronics', 'ConnectPro'],
              ['SKU003', 'Mechanical Keyboard', 'RGB mechanical keyboard', '89.99', 'Electronics', 'KeyMaster'],
              ['SKU004', 'Monitor Stand', 'Adjustable monitor stand', '39.99', 'Office', 'DeskWorks'],
              ['SKU005', 'Webcam HD', '1080p webcam with mic', '59.99', 'Electronics', 'VisionTech'],
            ],
            totalRows: 5
          })}
        >
          Product Catalog
        </button>
      </div>
    </div>
  )
}

// ============================================================================
// Step 2: Data Preview Component
// ============================================================================

function DataPreviewStep({ file, onContinue, onBack }) {
  return (
    <div className="preview-step">
      <div className="preview-header">
        <div className="file-info">
          <FileIcon size={20} />
          <span className="file-name">{file.name}</span>
          <span className="file-stats">{file.headers.length} columns · {file.totalRows} rows</span>
        </div>
        <div className="preview-actions">
          <button className="btn btn-secondary" onClick={onBack}>
            Change File
          </button>
          <button className="btn btn-primary" onClick={onContinue}>
            Start Mapping
            <ChevronRightIcon size={16} />
          </button>
        </div>
      </div>
      
      <div className="preview-table-container">
        <table className="preview-table">
          <thead>
            <tr>
              {file.headers.map((header, i) => (
                <th key={i}>
                  <span className="header-name">{header}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {file.rows.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j}>{cell || <span className="empty-cell">(empty)</span>}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      <p className="preview-hint">
        Showing first {file.rows.length} of {file.totalRows} rows
      </p>
    </div>
  )
}

// ============================================================================
// Cytoscape Ontology Graph Component
// ============================================================================

function OntologyGraph({ 
  ontology, 
  highlightedProperties = [], 
  selectedProperty = null,
  onSelectProperty 
}) {
  const containerRef = useRef(null)
  const cyRef = useRef(null)

  // Build Cytoscape elements (without selection state - that's handled separately)
  const elements = useMemo(() => {
    const nodes = []
    const edges = []
    
    // Add class nodes
    ontology.classes.forEach(cls => {
      nodes.push({
        data: { 
          id: cls.uri, 
          label: cls.label,
          type: 'class',
          description: cls.description
        },
        classes: 'class-node'
      })
    })
    
    // Add property nodes only for highlighted ones
    const propSet = new Set(highlightedProperties.map(h => h.property.uri))
    ontology.properties.forEach(prop => {
      if (propSet.has(prop.uri)) {
        const rec = highlightedProperties.find(h => h.property.uri === prop.uri)
        const confidence = rec?.confidence || 'low'
        
        nodes.push({
          data: { 
            id: prop.uri, 
            label: prop.label,
            type: 'property',
            confidence,
            description: prop.description
          },
          classes: `property-node ${confidence}`
        })
      }
    })
    
    // Add edges from classes to their properties
    let edgeId = 0
    ontology.classes.forEach(cls => {
      cls.properties.forEach(propUri => {
        if (propSet.has(propUri)) {
          edges.push({
            data: {
              id: `edge-${edgeId++}`,
              source: cls.uri,
              target: propUri
            }
          })
        }
      })
    })
    
    return [...nodes, ...edges]
  }, [ontology, highlightedProperties])

  // Update selection styling without re-rendering the whole graph
  useEffect(() => {
    if (!cyRef.current) return
    
    // Remove 'selected' class from all property nodes
    cyRef.current.nodes('.property-node').removeClass('selected')
    
    // Add 'selected' class to the selected node
    if (selectedProperty) {
      const node = cyRef.current.getElementById(selectedProperty)
      if (node.length) {
        node.addClass('selected')
      }
    }
  }, [selectedProperty])

  // Initialize Cytoscape
  useEffect(() => {
    if (!containerRef.current) return
    
    // Destroy previous instance
    if (cyRef.current) {
      cyRef.current.destroy()
    }
    
    // Skip if no elements
    if (elements.length === 0) {
      return
    }
    
    const cy = cytoscape({
      container: containerRef.current,
      elements: elements,
      style: [
        // Class nodes - diamond shape
        {
          selector: 'node.class-node',
          style: {
            'shape': 'diamond',
            'width': 50,
            'height': 50,
            'background-color': '#3b82f6',
            'border-color': '#60a5fa',
            'border-width': 3,
            'label': 'data(label)',
            'text-valign': 'bottom',
            'text-margin-y': 8,
            'color': '#f8fafc',
            'font-size': '12px',
            'text-outline-color': '#0f172a',
            'text-outline-width': 2
          }
        },
        // Property nodes - circles
        {
          selector: 'node.property-node',
          style: {
            'shape': 'ellipse',
            'width': 30,
            'height': 30,
            'background-color': '#334155',
            'border-width': 3,
            'label': 'data(label)',
            'text-valign': 'top',
            'text-margin-y': -8,
            'color': '#f8fafc',
            'font-size': '11px',
            'text-outline-color': '#0f172a',
            'text-outline-width': 2
          }
        },
        // Confidence colors
        {
          selector: 'node.property-node.high',
          style: {
            'border-color': '#22c55e'
          }
        },
        {
          selector: 'node.property-node.medium',
          style: {
            'border-color': '#f59e0b'
          }
        },
        {
          selector: 'node.property-node.low',
          style: {
            'border-color': '#6b7280'
          }
        },
        // Selected state
        {
          selector: 'node.property-node.selected',
          style: {
            'width': 40,
            'height': 40,
            'border-width': 3,
            'background-color': 'data(confidence)' === 'high' ? '#22c55e' : 
                                'data(confidence)' === 'medium' ? '#f59e0b' : '#6b7280',
            'color': '#3b82f6'
          }
        },
        {
          selector: 'node.property-node.selected.high',
          style: { 'background-color': '#22c55e' }
        },
        {
          selector: 'node.property-node.selected.medium',
          style: { 'background-color': '#f59e0b' }
        },
        {
          selector: 'node.property-node.selected.low',
          style: { 'background-color': '#6b7280' }
        },
        // Edges
        {
          selector: 'edge',
          style: {
            'width': 1.5,
            'line-color': '#475569',
            'target-arrow-color': '#475569',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'arrow-scale': 0.8
          }
        }
      ],
      layout: {
        name: 'cose',
        animate: false,
        nodeRepulsion: 8000,
        idealEdgeLength: 100,
        nodeOverlap: 20,
        padding: 50,
        fit: true
      },
      minZoom: 0.3,
      maxZoom: 3,
      wheelSensitivity: 0.3
    })
    
    // Handle node clicks
    cy.on('tap', 'node.property-node', (evt) => {
      const node = evt.target
      if (onSelectProperty) {
        onSelectProperty(node.id())
      }
    })
    
    // Fit to viewport after layout settles
    cy.once('layoutstop', () => {
      cy.fit(50)
    })
    
    cyRef.current = cy
    
    return () => {
      if (cyRef.current) {
        cyRef.current.destroy()
        cyRef.current = null
      }
    }
  }, [elements, onSelectProperty])
  
  const showEmptyMessage = elements.length === 0

  return (
    <div className="ontology-graph-container">
      <div ref={containerRef} className="ontology-graph-canvas" />
      {showEmptyMessage && (
        <div className="empty-graph-message">
          Select a column to see ontology recommendations
        </div>
      )}
      <div className="graph-legend">
        <div className="legend-item">
          <span className="legend-diamond">◆</span> Class
        </div>
        <div className="legend-item">
          <span className="legend-circle high">●</span> High match
        </div>
        <div className="legend-item">
          <span className="legend-circle medium">●</span> Medium match
        </div>
        <div className="legend-item">
          <span className="legend-circle low">●</span> Low match
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// Step 3: Column Mapping Wizard
// ============================================================================

function MappingWizardStep({ file, ontology, onComplete, onBack }) {
  const [currentColumnIndex, setCurrentColumnIndex] = useState(0)
  const [mappings, setMappings] = useState({})
  const [selectedProperty, setSelectedProperty] = useState(null)
  
  const currentColumn = file.headers[currentColumnIndex]
  const recommendations = useMemo(() => 
    getRecommendations(currentColumn, ontology, 8),
    [currentColumn, ontology]
  )
  
  // Sample values for current column
  const sampleValues = useMemo(() => 
    file.rows.map(row => row[currentColumnIndex]).filter(Boolean).slice(0, 5),
    [file.rows, currentColumnIndex]
  )
  
  const handleSelectProperty = useCallback((propertyUri) => {
    setSelectedProperty(propertyUri)
  }, [])
  
  const handleConfirmMapping = useCallback(() => {
    if (selectedProperty) {
      setMappings(prev => ({
        ...prev,
        [currentColumn]: selectedProperty
      }))
    }
    
    if (currentColumnIndex < file.headers.length - 1) {
      setCurrentColumnIndex(prev => prev + 1)
      setSelectedProperty(null)
    } else {
      // All columns mapped
      onComplete({
        ...mappings,
        [currentColumn]: selectedProperty
      })
    }
  }, [selectedProperty, currentColumn, currentColumnIndex, file.headers.length, mappings, onComplete])
  
  const handleSkip = useCallback(() => {
    if (currentColumnIndex < file.headers.length - 1) {
      setCurrentColumnIndex(prev => prev + 1)
      setSelectedProperty(null)
    } else {
      onComplete(mappings)
    }
  }, [currentColumnIndex, file.headers.length, mappings, onComplete])
  
  const handlePrevious = useCallback(() => {
    if (currentColumnIndex > 0) {
      setCurrentColumnIndex(prev => prev - 1)
      setSelectedProperty(mappings[file.headers[currentColumnIndex - 1]] || null)
    }
  }, [currentColumnIndex, mappings, file.headers])

  const progress = ((currentColumnIndex + 1) / file.headers.length) * 100

  return (
    <div className="mapping-wizard">
      {/* Progress bar */}
      <div className="wizard-progress">
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
        <span className="progress-text">
          Column {currentColumnIndex + 1} of {file.headers.length}
        </span>
      </div>
      
      <div className="wizard-content">
        {/* Left: Column info + recommendations */}
        <div className="wizard-left">
          <div className="current-column">
            <h3>Map Column</h3>
            <div className="column-name-display">
              <TableIcon size={20} />
              <span>{currentColumn}</span>
            </div>
            
            <div className="sample-values">
              <h4>Sample Values</h4>
              <ul>
                {sampleValues.map((val, i) => (
                  <li key={i}>{val}</li>
                ))}
              </ul>
            </div>
          </div>
          
          <div className="recommendations">
            <h4>
              <NetworkIcon size={16} />
              Recommended Properties
            </h4>
            <p className="rec-hint">Click to select, or choose from the graph</p>
            
            <ul className="recommendation-list">
              {recommendations.map((rec, i) => (
                <li 
                  key={rec.property.uri}
                  className={`recommendation-item ${rec.confidence} ${selectedProperty === rec.property.uri ? 'selected' : ''}`}
                  onClick={() => handleSelectProperty(rec.property.uri)}
                >
                  <div className="rec-main">
                    <span className="rec-label">{rec.property.label}</span>
                    <span className="rec-uri">{rec.property.uri}</span>
                  </div>
                  <div className="rec-meta">
                    <span className={`rec-score ${rec.confidence}`}>
                      {Math.round(rec.score * 100)}%
                    </span>
                    {selectedProperty === rec.property.uri && (
                      <CheckIcon size={16} className="rec-check" />
                    )}
                  </div>
                </li>
              ))}
            </ul>
            
            <button 
              className="btn btn-sm btn-secondary skip-btn"
              onClick={handleSkip}
            >
              Skip this column
            </button>
          </div>
        </div>
        
        {/* Right: Ontology graph */}
        <div className="wizard-right">
          <OntologyGraph
            ontology={ontology}
            highlightedProperties={recommendations}
            selectedProperty={selectedProperty}
            onSelectProperty={handleSelectProperty}
          />
        </div>
      </div>
      
      {/* Navigation */}
      <div className="wizard-nav">
        <button 
          className="btn btn-secondary"
          onClick={currentColumnIndex === 0 ? onBack : handlePrevious}
        >
          {currentColumnIndex === 0 ? 'Back to Preview' : 'Previous Column'}
        </button>
        
        <button 
          className="btn btn-primary"
          onClick={handleConfirmMapping}
          disabled={!selectedProperty}
        >
          {currentColumnIndex < file.headers.length - 1 ? (
            <>Confirm & Next <ChevronRightIcon size={16} /></>
          ) : (
            <>Finish Mapping <CheckIcon size={16} /></>
          )}
        </button>
      </div>
    </div>
  )
}

// ============================================================================
// Step 4: Review & Export
// ============================================================================

function ReviewStep({ file, mappings, ontology, onBack, onRestart }) {
  const [exportFormat, setExportFormat] = useState('rml')
  const [isConverting, setIsConverting] = useState(false)
  const [conversionResult, setConversionResult] = useState(null)
  const [outputFormat, setOutputFormat] = useState('ttl')
  
  const mappedColumns = Object.entries(mappings).filter(([_, prop]) => prop)
  
  // Generate RML output
  const rmlOutput = useMemo(() => {
    const prefixLines = Object.entries(ontology.prefixes)
      .map(([prefix, uri]) => `@prefix ${prefix}: <${uri}> .`)
      .join('\n')
    
    const predicateObjects = mappedColumns.map(([col, propUri]) => {
      return `    rr:predicateObjectMap [
        rr:predicate ${propUri} ;
        rr:objectMap [ rml:reference "${col}" ]
    ]`
    }).join(' ;\n')
    
    return `${prefixLines}
@prefix rr: <http://www.w3.org/ns/r2rml#> .
@prefix rml: <http://semweb.mmlab.be/ns/rml#> .
@prefix ql: <http://semweb.mmlab.be/ns/ql#> .

<#${file.name.replace('.csv', '')}Map> a rr:TriplesMap ;
    rml:logicalSource [
        rml:source "${file.name}" ;
        rml:referenceFormulation ql:CSV
    ] ;
${predicateObjects} .`
  }, [file, mappedColumns, ontology])

  // Generate YARRRML output
  const yarrrmlOutput = useMemo(() => {
    const prefixes = Object.entries(ontology.prefixes)
      .map(([prefix, uri]) => `    ${prefix}: "${uri}"`)
      .join('\n')
    
    const mappingEntries = mappedColumns.map(([col, propUri]) => {
      return `        - [${propUri}, $(${col})]`
    }).join('\n')
    
    const mapName = file.name.replace('.csv', '').replace(/[^a-zA-Z0-9]/g, '_')
    
    return `prefixes:
${prefixes}
    rr: "http://www.w3.org/ns/r2rml#"
    rml: "http://semweb.mmlab.be/ns/rml#"

mappings:
  ${mapName}:
    sources:
      - [${file.name}~csv]
    s: ex:resource/$(id)
    po:
${mappingEntries}`
  }, [file, mappedColumns, ontology])

  const currentOutput = exportFormat === 'rml' ? rmlOutput : yarrrmlOutput
  const fileExtension = exportFormat === 'rml' ? '.rml.ttl' : '.yarrrml.yml'

  const handleDownload = useCallback(() => {
    const blob = new Blob([currentOutput], { 
      type: exportFormat === 'rml' ? 'text/turtle' : 'text/yaml' 
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${file.name.replace('.csv', '')}_mapping${fileExtension}`
    a.click()
    URL.revokeObjectURL(url)
  }, [currentOutput, file.name, fileExtension, exportFormat])

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(currentOutput)
  }, [currentOutput])

  // Run ETL transformation
  const handleRunTransformation = useCallback(async () => {
    setIsConverting(true)
    setConversionResult(null)
    
    try {
      // Build mapping config from our mappings
      const mappingConfig = {
        sources: [{
          type: 'csv',
          file: file.name,
        }],
        mappings: mappedColumns.map(([col, propUri]) => ({
          source_column: col,
          predicate: propUri,
        })),
        prefixes: ontology.prefixes,
      }
      
      // Create form data
      const formData = new FormData()
      
      // Re-create the file blob from the parsed data
      const csvContent = [
        file.headers.join(','),
        ...file.preview.map(row => file.headers.map(h => row[h] || '').join(','))
      ].join('\n')
      const dataBlob = new Blob([csvContent], { type: 'text/csv' })
      formData.append('data_file', dataBlob, file.name)
      formData.append('mapping', JSON.stringify(mappingConfig))
      formData.append('output_format', outputFormat)
      
      const response = await fetch('/etl/convert', {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Conversion failed')
      }
      
      const result = await response.json()
      setConversionResult(result)
    } catch (err) {
      setConversionResult({ error: err.message })
    } finally {
      setIsConverting(false)
    }
  }, [file, mappedColumns, ontology.prefixes, outputFormat])

  const handleDownloadRdf = useCallback(() => {
    if (!conversionResult?.rdf_content) return
    
    const mimeTypes = {
      ttl: 'text/turtle',
      nt: 'application/n-triples',
      jsonld: 'application/ld+json',
      xml: 'application/rdf+xml',
    }
    
    const blob = new Blob([conversionResult.rdf_content], { 
      type: mimeTypes[outputFormat] || 'text/plain' 
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${file.name.replace('.csv', '')}.${outputFormat}`
    a.click()
    URL.revokeObjectURL(url)
  }, [conversionResult, outputFormat, file.name])

  return (
    <div className="review-step">
      <div className="review-header">
        <h2>Mapping Complete!</h2>
        <p>
          {mappedColumns.length} of {file.headers.length} columns mapped
        </p>
      </div>
      
      <div className="review-content">
        {/* Mapping summary */}
        <div className="mapping-summary">
          <h3>Column Mappings</h3>
          <table className="summary-table">
            <thead>
              <tr>
                <th>CSV Column</th>
                <th>→</th>
                <th>Ontology Property</th>
              </tr>
            </thead>
            <tbody>
              {file.headers.map(col => {
                const prop = mappings[col]
                const propDef = prop ? ontology.properties.find(p => p.uri === prop) : null
                return (
                  <tr key={col} className={prop ? '' : 'unmapped'}>
                    <td>{col}</td>
                    <td>→</td>
                    <td>
                      {propDef ? (
                        <span className="mapped-property">
                          {propDef.label}
                          <span className="prop-uri">{prop}</span>
                        </span>
                      ) : (
                        <span className="unmapped-label">(not mapped)</span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        
        {/* Mapping output section */}
        <div className="rml-output">
          <div className="rml-header">
            <h3>Generated Mapping</h3>
            <div className="format-selector">
              <button 
                className={`btn btn-sm ${exportFormat === 'rml' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setExportFormat('rml')}
              >
                RML
              </button>
              <button 
                className={`btn btn-sm ${exportFormat === 'yarrrml' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setExportFormat('yarrrml')}
              >
                YARRRML
              </button>
            </div>
            <div className="rml-actions">
              <button className="btn btn-sm btn-secondary" onClick={handleCopy}>
                Copy
              </button>
              <button className="btn btn-sm btn-primary" onClick={handleDownload}>
                <DownloadIcon size={14} /> Download
              </button>
            </div>
          </div>
          <pre className="rml-code">{currentOutput}</pre>
        </div>

        {/* ETL Transformation Section */}
        <div className="etl-section">
          <div className="etl-header">
            <h3>Run Transformation</h3>
            <p className="etl-description">
              Convert your CSV data to RDF using the mapping above
            </p>
          </div>
          
          <div className="etl-controls">
            <div className="output-format-select">
              <label>Output Format:</label>
              <select 
                value={outputFormat} 
                onChange={e => setOutputFormat(e.target.value)}
                disabled={isConverting}
              >
                <option value="ttl">Turtle (.ttl)</option>
                <option value="nt">N-Triples (.nt)</option>
                <option value="jsonld">JSON-LD (.jsonld)</option>
                <option value="xml">RDF/XML (.xml)</option>
              </select>
            </div>
            
            <button 
              className="btn btn-success btn-run-etl"
              onClick={handleRunTransformation}
              disabled={isConverting || mappedColumns.length === 0}
            >
              {isConverting ? (
                <>
                  <RefreshIcon size={14} className="spin" /> Converting...
                </>
              ) : (
                <>
                  <PlayIcon size={14} /> Run Transformation
                </>
              )}
            </button>
          </div>
          
          {/* Conversion Result */}
          {conversionResult && (
            <div className={`etl-result ${conversionResult.error ? 'error' : 'success'}`}>
              {conversionResult.error ? (
                <>
                  <AlertIcon size={16} />
                  <span>Error: {conversionResult.error}</span>
                </>
              ) : (
                <>
                  <div className="result-header">
                    <CheckIcon size={16} />
                    <span>Generated {conversionResult.triple_count} triples</span>
                    <button className="btn btn-sm btn-primary" onClick={handleDownloadRdf}>
                      <DownloadIcon size={14} /> Download RDF
                    </button>
                  </div>
                  <pre className="rdf-preview">
                    {conversionResult.rdf_content?.slice(0, 2000)}
                    {conversionResult.rdf_content?.length > 2000 && '\n\n... (truncated)'}
                  </pre>
                </>
              )}
            </div>
          )}
        </div>
      </div>
      
      <div className="review-actions">
        <button className="btn btn-secondary" onClick={onBack}>
          Edit Mappings
        </button>
        <button className="btn btn-primary" onClick={onRestart}>
          Map Another File
        </button>
      </div>
    </div>
  )
}

// ============================================================================
// Main Starchart Component
// ============================================================================

export function Starchart({ theme = 'dark' }) {
  const [step, setStep] = useState('upload') // upload | preview | mapping | review
  const [file, setFile] = useState(null)
  const [mappings, setMappings] = useState({})
  const [ontology] = useState(SAMPLE_ONTOLOGY)

  const handleFileLoaded = useCallback((fileData) => {
    setFile(fileData)
    setStep('preview')
  }, [])

  const handleStartMapping = useCallback(() => {
    setStep('mapping')
  }, [])

  const handleMappingComplete = useCallback((finalMappings) => {
    setMappings(finalMappings)
    setStep('review')
  }, [])

  const handleRestart = useCallback(() => {
    setFile(null)
    setMappings({})
    setStep('upload')
  }, [])

  return (
    <div className="starchart" data-theme={theme}>
      {/* Header */}
      <div className="starchart-header">
        <div className="header-title">
          <NetworkIcon size={24} />
          <h1>Starchart</h1>
          <span className="subtitle">Visual RML Mapper</span>
        </div>
        
        {step !== 'upload' && (
          <div className="header-steps">
            <span className={`step ${step === 'upload' ? 'active' : 'done'}`}>1. Upload</span>
            <span className={`step ${step === 'preview' ? 'active' : step === 'mapping' || step === 'review' ? 'done' : ''}`}>2. Preview</span>
            <span className={`step ${step === 'mapping' ? 'active' : step === 'review' ? 'done' : ''}`}>3. Map</span>
            <span className={`step ${step === 'review' ? 'active' : ''}`}>4. Export</span>
          </div>
        )}
      </div>
      
      {/* Content */}
      <div className="starchart-content">
        {step === 'upload' && (
          <FileUploadStep onFileLoaded={handleFileLoaded} />
        )}
        
        {step === 'preview' && file && (
          <DataPreviewStep 
            file={file} 
            onContinue={handleStartMapping}
            onBack={handleRestart}
          />
        )}
        
        {step === 'mapping' && file && (
          <MappingWizardStep
            file={file}
            ontology={ontology}
            onComplete={handleMappingComplete}
            onBack={() => setStep('preview')}
          />
        )}
        
        {step === 'review' && file && (
          <ReviewStep
            file={file}
            mappings={mappings}
            ontology={ontology}
            onBack={() => setStep('mapping')}
            onRestart={handleRestart}
          />
        )}
      </div>
    </div>
  )
}

export default Starchart
