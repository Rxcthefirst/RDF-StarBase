// ============================================================================
// Starchart — Visual RML/R2RML Mapping Editor with Smart Recommendations
// ============================================================================

import { useState, useCallback, useRef, useEffect, useMemo, useLayoutEffect } from 'react'
import cytoscape from 'cytoscape'
import cola from 'cytoscape-cola'
import {
  DatabaseIcon, TableIcon, FileIcon, PlusIcon, TrashIcon,
  PlayIcon, DownloadIcon, UploadIcon, RefreshIcon, SearchIcon,
  ChevronRightIcon, LinkIcon, SettingsIcon,
  CheckIcon, AlertIcon, CloseIcon, GripIcon, NetworkIcon
} from './Icons'
import './Starchart.css'

// Register the cola layout extension
try { cytoscape.use(cola) } catch (e) { /* already registered */ }

// ============================================================================
// Sample Data Generators (randomized for variety)
// ============================================================================

const FIRST_NAMES = [
  'Emma', 'Liam', 'Olivia', 'Noah', 'Ava', 'William', 'Sophia', 'James',
  'Isabella', 'Oliver', 'Mia', 'Benjamin', 'Charlotte', 'Elijah', 'Amelia',
  'Lucas', 'Harper', 'Mason', 'Evelyn', 'Logan', 'Abigail', 'Alexander',
  'Emily', 'Ethan', 'Elizabeth', 'Jacob', 'Sofia', 'Michael', 'Avery', 'Daniel'
]

const LAST_NAMES = [
  'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
  'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
  'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
  'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson'
]

const CITIES = [
  { city: 'New York', state: 'NY', country: 'USA' },
  { city: 'Los Angeles', state: 'CA', country: 'USA' },
  { city: 'Chicago', state: 'IL', country: 'USA' },
  { city: 'Houston', state: 'TX', country: 'USA' },
  { city: 'Phoenix', state: 'AZ', country: 'USA' },
  { city: 'London', state: '', country: 'UK' },
  { city: 'Paris', state: '', country: 'France' },
  { city: 'Tokyo', state: '', country: 'Japan' },
  { city: 'Sydney', state: 'NSW', country: 'Australia' },
  { city: 'Berlin', state: '', country: 'Germany' },
  { city: 'Toronto', state: 'ON', country: 'Canada' },
  { city: 'Mumbai', state: 'MH', country: 'India' },
  { city: 'São Paulo', state: 'SP', country: 'Brazil' },
  { city: 'Singapore', state: '', country: 'Singapore' },
  { city: 'Seoul', state: '', country: 'South Korea' }
]

const PRODUCT_ADJECTIVES = [
  'Premium', 'Deluxe', 'Professional', 'Essential', 'Ultimate', 'Classic',
  'Modern', 'Wireless', 'Portable', 'Smart', 'Ultra', 'Pro', 'Max', 'Elite'
]

const PRODUCT_NOUNS = [
  'Mouse', 'Keyboard', 'Monitor', 'Headphones', 'Speaker', 'Hub', 'Charger',
  'Stand', 'Webcam', 'Microphone', 'Cable', 'Adapter', 'Case', 'Light',
  'Controller', 'Sensor', 'Tracker', 'Watch', 'Band', 'Earbuds'
]

const CATEGORIES = [
  'Electronics', 'Computers', 'Audio', 'Video', 'Accessories', 'Mobile',
  'Gaming', 'Office', 'Smart Home', 'Wearables', 'Photography', 'Networking'
]

const BRANDS = [
  'TechCorp', 'ProGear', 'QualityFirst', 'SmartLife', 'EliteGoods', 'ConnectPro',
  'VisionTech', 'SoundMax', 'PowerUp', 'DataLink', 'NetWorks', 'DigiCore'
]

const ORDER_STATUSES = ['Pending', 'Processing', 'Shipped', 'Delivered', 'Cancelled']
const PAYMENT_METHODS = ['Credit Card', 'PayPal', 'Bank Transfer', 'Apple Pay', 'Google Pay']

// Helper: random element from array
const randomFrom = arr => arr[Math.floor(Math.random() * arr.length)]

// Helper: random integer in range
const randomInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min

// Helper: random price
const randomPrice = (min, max) => (Math.random() * (max - min) + min).toFixed(2)

// Helper: random date in past N days
const randomDate = (daysBack = 365) => {
  const d = new Date()
  d.setDate(d.getDate() - randomInt(0, daysBack))
  return d.toISOString().split('T')[0]
}

// Generate 20 random customers
function generateSamplePeople() {
  const rows = []
  for (let i = 1; i <= 20; i++) {
    const first = randomFrom(FIRST_NAMES)
    const last = randomFrom(LAST_NAMES)
    const loc = randomFrom(CITIES)
    const email = `${first.toLowerCase()}.${last.toLowerCase()}@example.com`
    const phone = `+1-${randomInt(200, 999)}-${randomInt(100, 999)}-${randomInt(1000, 9999)}`
    rows.push([
      `CUST${String(i).padStart(5, '0')}`,
      first,
      last,
      email,
      phone,
      loc.city,
      loc.state,
      loc.country,
      randomDate(730),  // registration date
      String(randomInt(0, 50000))  // loyalty points
    ])
  }
  return {
    name: 'sample_customers.csv',
    headers: ['customer_id', 'first_name', 'last_name', 'email_address', 'phone_number', 
              'city', 'state_province', 'country', 'registered_at', 'loyalty_points'],
    rows,
    totalRows: 20
  }
}

// Generate 20 random products
function generateSampleProducts() {
  const rows = []
  for (let i = 1; i <= 20; i++) {
    const adj = randomFrom(PRODUCT_ADJECTIVES)
    const noun = randomFrom(PRODUCT_NOUNS)
    const listPrice = randomPrice(19.99, 299.99)
    const salePrice = (parseFloat(listPrice) * (1 - Math.random() * 0.2)).toFixed(2)
    rows.push([
      `PROD${String(i).padStart(6, '0')}`,
      `SKU-${Math.random().toString(36).substring(2, 8).toUpperCase()}`,
      `${adj} ${noun}`,
      `High-quality ${adj.toLowerCase()} ${noun.toLowerCase()} for everyday use`,
      listPrice,
      salePrice,
      randomFrom(CATEGORIES),
      randomFrom(BRANDS),
      String(randomInt(0, 500)),
      randomDate(180)
    ])
  }
  return {
    name: 'sample_products.csv',
    headers: ['product_id', 'sku', 'product_name', 'description', 'list_price', 
              'sale_price', 'category', 'brand', 'stock_quantity', 'created_at'],
    rows,
    totalRows: 20
  }
}

// Generate 20 random orders
function generateSampleOrders() {
  const rows = []
  for (let i = 1; i <= 20; i++) {
    const subtotal = randomPrice(25, 500)
    const tax = (parseFloat(subtotal) * 0.08).toFixed(2)
    const shipping = randomPrice(0, 15)
    const total = (parseFloat(subtotal) + parseFloat(tax) + parseFloat(shipping)).toFixed(2)
    rows.push([
      `ORD${String(i).padStart(8, '0')}`,
      `CUST${String(randomInt(1, 100)).padStart(5, '0')}`,
      randomDate(90),
      randomFrom(ORDER_STATUSES),
      subtotal,
      tax,
      shipping,
      total,
      'USD',
      randomFrom(PAYMENT_METHODS),
      String(randomInt(1, 10))
    ])
  }
  return {
    name: 'sample_orders.csv',
    headers: ['order_id', 'customer_id', 'order_date', 'order_status', 'subtotal',
              'tax_amount', 'shipping_cost', 'total_amount', 'currency', 'payment_method', 'items_count'],
    rows,
    totalRows: 20
  }
}

// ============================================================================
// Empty Ontology (loaded from repository)
// ============================================================================

const EMPTY_ONTOLOGY = {
  classes: [],
  properties: [],
  prefixes: {}
}

// ============================================================================
// String Similarity (Local Fallback)
// ============================================================================

/**
 * Calculate similarity between two strings (0-1)
 * Uses a combination of techniques for local fallback when API unavailable
 */
function calculateSimilarity(str1, str2) {
  // Handle null/undefined inputs
  if (!str1 || !str2) return 0
  
  const s1 = String(str1).toLowerCase().replace(/[_-]/g, ' ').trim()
  const s2 = String(str2).toLowerCase().replace(/[_-]/g, ' ').trim()
  
  // Empty string check
  if (!s1 || !s2) return 0
  
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
 * Get property recommendations locally (fallback)
 */
function getLocalRecommendations(columnHeader, ontology, topK = 5) {
  // Handle missing inputs
  if (!columnHeader || !ontology || !ontology.properties) {
    return []
  }
  
  const scores = []
  
  for (const prop of ontology.properties) {
    if (!prop || !prop.uri) continue
    
    // Check similarity against label
    let maxScore = calculateSimilarity(columnHeader, prop.label || '')
    let matchType = 'label'
    
    // Check against aliases
    for (const alias of (prop.aliases || [])) {
      const aliasScore = calculateSimilarity(columnHeader, alias)
      if (aliasScore > maxScore) {
        maxScore = aliasScore
        matchType = 'alias'
      }
    }
    
    // Check against URI local name
    const localName = prop.uri.includes(':') 
      ? prop.uri.split(':').pop() 
      : prop.uri.split('/').pop() || prop.uri
    const uriScore = calculateSimilarity(columnHeader, localName)
    if (uriScore > maxScore) {
      maxScore = uriScore
      matchType = 'uri'
    }
    
    scores.push({
      property: prop,
      score: maxScore,
      confidence: maxScore >= 0.8 ? 'high' : maxScore >= 0.5 ? 'medium' : 'low',
      matchType,
    })
  }
  
  return scores
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
}

/**
 * Fetch recommendations from the embedding API
 */
async function fetchRecommendations(columnHeader, ontology, topK = 8) {
  try {
    const response = await fetch('/api/etl/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        column_header: columnHeader,
        properties: ontology.properties.map(p => ({
          uri: p.uri,
          label: p.label,
          aliases: p.aliases || [],
          description: p.description || '',
          domain: p.domain,
          range: p.range,
        })),
        top_k: topK,
      }),
    })
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`)
    }
    
    const data = await response.json()
    
    // Convert API response to component format
    return {
      recommendations: data.recommendations.map(rec => ({
        property: ontology.properties.find(p => p.uri === rec.uri) || { uri: rec.uri, label: rec.label },
        score: rec.score,
        confidence: rec.confidence,
        matchType: rec.match_type,
      })),
      usingEmbeddings: data.using_embeddings,
    }
  } catch (error) {
    console.warn('Failed to fetch recommendations from API, using local fallback:', error)
    return null
  }
}

/**
 * Custom hook for property recommendations with API enhancement
 * Shows local recommendations immediately, then enhances with embeddings
 */
function useRecommendations(columnHeader, ontology, topK = 8) {
  const [recommendations, setRecommendations] = useState([])
  const [usingEmbeddings, setUsingEmbeddings] = useState(false)
  const [loading, setLoading] = useState(false)
  
  useEffect(() => {
    // Skip if no column header or ontology
    if (!columnHeader || !ontology || !ontology.properties || ontology.properties.length === 0) {
      setRecommendations([])
      setUsingEmbeddings(false)
      setLoading(false)
      return
    }
    
    let cancelled = false
    const controller = new AbortController()
    
    // Immediate local recommendations - no waiting
    const localRecs = getLocalRecommendations(columnHeader, ontology, topK)
    setRecommendations(localRecs)
    setUsingEmbeddings(false)
    setLoading(false)
    
    // Try to enhance with embeddings in background (with timeout)
    async function enhanceWithEmbeddings() {
      // Short delay to let local results show first
      await new Promise(r => setTimeout(r, 100))
      
      if (cancelled) return
      
      try {
        // Add timeout to the fetch
        const timeoutId = setTimeout(() => controller.abort(), 3000) // 3 second timeout
        
        const response = await fetch('/api/etl/recommend', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            column_header: columnHeader,
            properties: ontology.properties.map(p => ({
              uri: p.uri,
              label: p.label || '',
              aliases: p.aliases || [],
              description: p.description || '',
              domain: p.domain,
              range: p.range,
            })),
            top_k: topK,
          }),
          signal: controller.signal,
        })
        
        clearTimeout(timeoutId)
        
        if (!response.ok || cancelled) return
        
        const data = await response.json()
        
        if (cancelled) return
        
        // Only update if we got better results
        if (data.using_embeddings && data.recommendations?.length > 0) {
          const enhancedRecs = data.recommendations.map(rec => ({
            property: ontology.properties.find(p => p.uri === rec.uri) || { uri: rec.uri, label: rec.label },
            score: rec.score,
            confidence: rec.confidence,
            matchType: rec.match_type,
          }))
          setRecommendations(enhancedRecs)
          setUsingEmbeddings(true)
        }
      } catch (error) {
        // Silently ignore - we already have local recommendations
        if (error.name !== 'AbortError') {
          console.debug('Embedding enhancement skipped:', error.message)
        }
      }
    }
    
    enhanceWithEmbeddings()
    
    return () => { 
      cancelled = true
      controller.abort()
    }
  }, [columnHeader, ontology, topK])
  
  return { recommendations, usingEmbeddings, loading }
}

// Keep the old function name as an alias for backwards compatibility
const getRecommendations = getLocalRecommendations

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
          onClick={() => onFileLoaded(generateSamplePeople())}
        >
          Customers (20 rows)
        </button>
        <button 
          className="btn btn-sm btn-secondary"
          onClick={() => onFileLoaded(generateSampleProducts())}
        >
          Products (20 rows)
        </button>
        <button 
          className="btn btn-sm btn-secondary"
          onClick={() => onFileLoaded(generateSampleOrders())}
        >
          Orders (20 rows)
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
// Cytoscape Ontology Graph Component — Classes as Nodes, Properties as Edges
// ============================================================================

/**
 * Build graph elements from ontology data with optional 2-hop neighborhood filtering
 * Following the original Starchart pattern:
 * - Classes are nodes (ellipse shape)
 * - Object properties are edges connecting domain → range
 * - Data properties are NOT visualized (they don't connect classes)
 */
function buildOntologyElements(ontology, highlightedProperties = [], options = {}) {
  const { 
    maxClasses = 100, 
    maxProperties = 200, 
    allowDataProperties = false,
    focusClass = null,  // Center class for neighborhood filtering
    hops = 2            // Number of hops from focus class
  } = options
  
  const elements = []
  const classSet = new Set()
  const highlightedUris = new Set(highlightedProperties.map(h => h.property?.uri || h.uri))
  
  // Helper: extract short label from URI
  const getLabel = (uri, prefLabel, label) => {
    if (prefLabel) return prefLabel
    if (label) return label
    if (!uri) return ''
    if (uri.includes('#')) return uri.split('#').pop()
    if (uri.includes('/')) return uri.split('/').pop()
    return uri
  }
  
  // Helper: check if range is a datatype (xsd:, XMLSchema)
  const isDatatypeRange = (range) => {
    if (!range) return true
    return /xsd:|XMLSchema|string|integer|decimal|date|boolean|float|double/i.test(range)
  }
  
  // Build adjacency for neighborhood calculation
  const adjacency = new Map() // classUri -> Set of connected classUris
  const propertyList = ontology.properties || []
  
  for (const prop of propertyList) {
    const domain = prop.domain
    const range = prop.range
    if (!domain || !range) continue
    if (isDatatypeRange(range) && !allowDataProperties) continue
    
    if (!adjacency.has(domain)) adjacency.set(domain, new Set())
    if (!adjacency.has(range)) adjacency.set(range, new Set())
    adjacency.get(domain).add(range)
    adjacency.get(range).add(domain)
  }
  
  // Calculate N-hop neighborhood from focus class
  let neighborhoodClasses = null
  let classDistances = new Map()
  
  if (focusClass && adjacency.has(focusClass)) {
    neighborhoodClasses = new Set([focusClass])
    classDistances.set(focusClass, 0)
    let frontier = new Set([focusClass])
    
    for (let hop = 1; hop <= hops; hop++) {
      const nextFrontier = new Set()
      for (const cls of frontier) {
        const neighbors = adjacency.get(cls) || new Set()
        for (const neighbor of neighbors) {
          if (!neighborhoodClasses.has(neighbor)) {
            neighborhoodClasses.add(neighbor)
            classDistances.set(neighbor, hop)
            nextFrontier.add(neighbor)
          }
        }
      }
      frontier = nextFrontier
    }
  }
  
  // Build class nodes (filtered by neighborhood if focus class set)
  const classSubset = (ontology.classes || []).slice(0, maxClasses)
  for (const cls of classSubset) {
    const id = cls.uri
    
    // Skip if not in neighborhood (when filtering)
    if (neighborhoodClasses && !neighborhoodClasses.has(id)) continue
    
    const label = getLabel(cls.uri, null, cls.label)
    const distance = classDistances.get(id) ?? -1
    const isFocus = id === focusClass
    
    elements.push({
      data: { 
        id, 
        label: label || id, 
        type: 'class',
        description: cls.description,
        distance,
        isFocus
      },
      classes: `class-node ${isFocus ? 'focus-node' : ''} ${distance === 1 ? 'neighbor-1' : ''} ${distance === 2 ? 'neighbor-2' : ''}`
    })
    classSet.add(id)
  }
  
  // Build edges from object properties (domain → range)
  let edgeCount = 0
  for (const prop of propertyList) {
    if (edgeCount >= maxProperties) break
    
    const domain = prop.domain
    const range = prop.range
    
    // Skip if no domain or range
    if (!domain || !range) continue
    
    // Skip datatype properties unless explicitly allowed
    if (isDatatypeRange(range) && !allowDataProperties) continue
    
    // When filtering by neighborhood, skip edges not in neighborhood
    if (neighborhoodClasses) {
      if (!neighborhoodClasses.has(domain) || !neighborhoodClasses.has(range)) continue
    }
    
    // Create stub nodes for classes not in ontology (but in neighborhood)
    if (!classSet.has(domain)) {
      const distance = classDistances.get(domain) ?? -1
      elements.push({
        data: { 
          id: domain, 
          label: getLabel(domain, null, null), 
          type: 'class',
          stub: true,
          distance
        },
        classes: 'class-node stub-node'
      })
      classSet.add(domain)
    }
    
    if (!classSet.has(range)) {
      const distance = classDistances.get(range) ?? -1
      elements.push({
        data: { 
          id: range, 
          label: getLabel(range, null, null), 
          type: 'class',
          stub: true,
          distance
        },
        classes: 'class-node stub-node'
      })
      classSet.add(range)
    }
    
    // Create edge for property
    const isHighlighted = highlightedUris.has(prop.uri)
    const rec = highlightedProperties.find(h => (h.property?.uri || h.uri) === prop.uri)
    const confidence = rec?.confidence || 'none'
    
    elements.push({
      data: {
        id: `${prop.uri}-edge`,
        source: domain,
        target: range,
        label: getLabel(prop.uri, null, prop.label),
        propertyUri: prop.uri,
        type: 'property',
        confidence: isHighlighted ? confidence : 'none',
        highlighted: isHighlighted
      },
      classes: `property-edge ${isHighlighted ? `highlighted ${confidence}` : ''}`
    })
    edgeCount++
  }
  
  return elements
}

// ============================================================================
// Class Recommendation & Viewpoint Graph Builder
// ============================================================================

/**
 * Recommend entity classes for a file based on file name, column names, and sample values.
 * Returns ranked list of classes with scores and matching reasons.
 */
function recommendClassesForFile(file, ontology) {
  const recommendations = []
  const columnNames = file.headers.map(h => h.toLowerCase())
  
  // Helper: extract local name from URI
  const getLocalName = (uri) => {
    if (!uri) return ''
    if (uri.includes('#')) return uri.split('#').pop()
    if (uri.includes('/')) return uri.split('/').pop()
    return uri
  }
  
  // Helper: normalize text for matching
  const normalize = (s) => (s || '').toLowerCase().replace(/[_-]/g, '').replace(/s$/, '') // Also remove trailing 's' for plurals
  
  // Extract potential entity name from file name (e.g., "reviews.csv" -> "review")
  const fileBaseName = (file.name || '').replace(/\.[^.]+$/, '') // Remove extension
  const normalizedFileName = normalize(fileBaseName)
  
  // Score each class
  for (const cls of ontology.classes || []) {
    const className = normalize(getLocalName(cls.uri))
    const classLabel = normalize(cls.label)
    
    // Get class aliases for matching (altLabels from ontology)
    const classAliases = (cls.aliases || []).map(a => normalize(a))
    
    let score = 0
    const matchingColumns = []
    const matchingProperties = []
    let matchReason = ''
    
    // === FILE NAME MATCHING (strongest signal) ===
    if (normalizedFileName) {
      // Direct file name match
      if (normalizedFileName === className || normalizedFileName === classLabel) {
        score += 0.6
        matchReason = 'file name'
      }
      // File name contains class name or vice versa
      else if (normalizedFileName.includes(className) || className.includes(normalizedFileName)) {
        score += 0.5
        matchReason = 'file name'
      }
      // File name matches an alias
      else if (classAliases.some(alias => normalizedFileName === alias || normalizedFileName.includes(alias))) {
        score += 0.45
        matchReason = 'file name (alias)'
      }
    }
    
    // === COLUMN NAME MATCHING ===
    for (const col of columnNames) {
      const normCol = normalize(col)
      
      // Column contains class name (e.g., "review_id", "review_text")
      if (normCol.includes(className) || normCol.includes(classLabel)) {
        score += 0.25
        if (!matchingColumns.includes(col)) matchingColumns.push(col)
      }
      
      // Check for ID columns: {class}id, {class}_id, id (for first column named just "id")
      if (normCol === `${className}id` || normCol === 'id' || normCol === 'identifier') {
        score += 0.2
        if (!matchingColumns.includes(col)) matchingColumns.push(col)
      }
      
      // Check against class aliases
      for (const alias of classAliases) {
        if (normCol.includes(alias)) {
          score += 0.15
          if (!matchingColumns.includes(col)) matchingColumns.push(col)
        }
      }
    }
    
    // === PROPERTY MATCHING ===
    // Check how many data properties of this class match column names
    const classProperties = getClassDataProperties(cls.uri, ontology)
    for (const prop of classProperties) {
      const propName = normalize(getLocalName(prop.uri))
      const propLabel = normalize(prop.label)
      const propAliases = (prop.aliases || []).map(a => normalize(a))
      
      for (const col of columnNames) {
        const normCol = normalize(col)
        
        // Direct match
        if (normCol === propName || normCol === propLabel) {
          score += 0.15
          matchingProperties.push({ col, property: prop.label || getLocalName(prop.uri) })
        }
        // Partial match
        else if (normCol.includes(propName) || propName.includes(normCol)) {
          score += 0.1
          matchingProperties.push({ col, property: prop.label || getLocalName(prop.uri) })
        }
        // Alias match
        else if (propAliases.some(alias => normCol === alias || normCol.includes(alias))) {
          score += 0.1
          matchingProperties.push({ col, property: prop.label || getLocalName(prop.uri) })
        }
      }
    }
    
    // Bonus for having multiple property matches
    if (matchingProperties.length >= 3) score += 0.15
    if (matchingProperties.length >= 5) score += 0.1
    
    if (score > 0) {
      recommendations.push({
        class: cls,
        score: Math.min(1.0, score),
        confidence: score > 0.5 ? 'high' : score > 0.25 ? 'medium' : 'low',
        matchingColumns,
        matchingProperties,
        matchReason,
        dataPropertyCount: classProperties.length
      })
    }
  }
  
  // Sort by score descending
  recommendations.sort((a, b) => b.score - a.score)
  return recommendations.slice(0, 5)
}

/**
 * Get data properties for a specific class (properties where this class is the domain
 * and the range is a datatype like xsd:string, xsd:decimal, etc.)
 */
function getClassDataProperties(classUri, ontology) {
  const isDatatypeRange = (range) => {
    if (!range) return false
    return /xsd:|XMLSchema|string|integer|decimal|date|boolean|float|double|dateTime|time/i.test(range)
  }
  
  const dataProps = []
  for (const prop of ontology.properties || []) {
    if (prop.domain === classUri && isDatatypeRange(prop.range)) {
      dataProps.push(prop)
    }
  }
  return dataProps
}

/**
 * Get object properties where this class is the domain (outgoing relationships)
 */
function getClassObjectProperties(classUri, ontology) {
  const isDatatypeRange = (range) => {
    if (!range) return false
    return /xsd:|XMLSchema|string|integer|decimal|date|boolean|float|double/i.test(range)
  }
  
  const objProps = []
  for (const prop of ontology.properties || []) {
    if (prop.domain === classUri && prop.range && !isDatatypeRange(prop.range)) {
      objProps.push(prop)
    }
  }
  return objProps
}

/**
 * Build a focused viewpoint graph centered on a class.
 * Shows:
 * - The center class (selected entity)
 * - Its data properties as attribute nodes
 * - Its object properties as edges to connected classes (1-hop)
 * - Connected classes
 */
function buildViewpointGraph(centerClassUri, ontology, options = {}) {
  const { showDataProperties = true, highlightedColumns = [] } = options
  const elements = []
  const addedNodes = new Set()
  const addedEdges = new Set()
  
  // Helper: extract short label from URI  
  const getLabel = (uri, label) => {
    if (label) return label
    if (!uri) return ''
    if (uri.includes('#')) return uri.split('#').pop()
    if (uri.includes('/')) return uri.split('/').pop()
    return uri
  }
  
  // Helper: check if range is datatype
  const isDatatypeRange = (range) => {
    if (!range) return true
    return /xsd:|XMLSchema|string|integer|decimal|date|boolean|float|double/i.test(range)
  }
  
  // Find center class
  const centerClass = (ontology.classes || []).find(c => c.uri === centerClassUri)
  if (!centerClass) return elements
  
  // Add center class node
  elements.push({
    data: {
      id: centerClassUri,
      label: getLabel(centerClassUri, centerClass.label),
      type: 'class',
      isCenter: true
    },
    classes: 'class-node center-class'
  })
  addedNodes.add(centerClassUri)
  
  // Add data properties as attribute nodes (if enabled)
  if (showDataProperties) {
    const dataProps = getClassDataProperties(centerClassUri, ontology)
    for (const prop of dataProps) {
      const propLabel = getLabel(prop.uri, prop.label)
      const isHighlighted = highlightedColumns.some(col => 
        propLabel.toLowerCase().includes(col.toLowerCase()) ||
        col.toLowerCase().includes(propLabel.toLowerCase())
      )
      
      elements.push({
        data: {
          id: prop.uri,
          label: propLabel,
          type: 'dataProperty',
          range: prop.range,
          highlighted: isHighlighted
        },
        classes: `data-property-node ${isHighlighted ? 'highlighted' : ''}`
      })
      addedNodes.add(prop.uri)
      
      // Edge from class to data property
      elements.push({
        data: {
          id: `${centerClassUri}-${prop.uri}`,
          source: centerClassUri,
          target: prop.uri,
          label: '',
          type: 'hasProperty'
        },
        classes: 'data-property-edge'
      })
    }
  }
  
  // Add object properties and connected classes (1-hop neighborhood)
  for (const prop of ontology.properties || []) {
    const domain = prop.domain
    const range = prop.range
    
    if (!domain || !range || isDatatypeRange(range)) continue
    
    // Outgoing: center → other
    if (domain === centerClassUri) {
      // Add range class if not added
      if (!addedNodes.has(range)) {
        const rangeClass = (ontology.classes || []).find(c => c.uri === range)
        elements.push({
          data: {
            id: range,
            label: getLabel(range, rangeClass?.label),
            type: 'class',
            isNeighbor: true
          },
          classes: 'class-node neighbor-class'
        })
        addedNodes.add(range)
      }
      
      // Add edge
      const edgeId = `${prop.uri}-edge`
      if (!addedEdges.has(edgeId)) {
        elements.push({
          data: {
            id: edgeId,
            source: domain,
            target: range,
            label: getLabel(prop.uri, prop.label),
            propertyUri: prop.uri,
            type: 'objectProperty'
          },
          classes: 'object-property-edge outgoing'
        })
        addedEdges.add(edgeId)
      }
    }
    
    // Incoming: other → center
    if (range === centerClassUri && domain !== centerClassUri) {
      // Add domain class if not added
      if (!addedNodes.has(domain)) {
        const domainClass = (ontology.classes || []).find(c => c.uri === domain)
        elements.push({
          data: {
            id: domain,
            label: getLabel(domain, domainClass?.label),
            type: 'class',
            isNeighbor: true
          },
          classes: 'class-node neighbor-class'
        })
        addedNodes.add(domain)
      }
      
      // Add incoming edge
      const edgeId = `${prop.uri}-edge-in`
      if (!addedEdges.has(edgeId)) {
        elements.push({
          data: {
            id: edgeId,
            source: domain,
            target: range,
            label: getLabel(prop.uri, prop.label),
            propertyUri: prop.uri,
            type: 'objectProperty'
          },
          classes: 'object-property-edge incoming'
        })
        addedEdges.add(edgeId)
      }
    }
  }
  
  return elements
}

// ============================================================================
// Viewpoint Graph Component — Focused view centered on a class
// ============================================================================

function ViewpointGraph({ 
  centerClass, 
  ontology,
  highlightedColumns = [],
  selectedProperty = null,
  onSelectProperty,
  onSelectClass
}) {
  const containerRef = useRef(null)
  const cyRef = useRef(null)
  const [loading, setLoading] = useState(false)
  
  // Build focused elements
  const elements = useMemo(() => {
    if (!centerClass || !ontology) return []
    return buildViewpointGraph(centerClass, ontology, {
      showDataProperties: true,
      highlightedColumns
    })
  }, [centerClass, ontology, highlightedColumns])
  
  // Initialize cytoscape
  useLayoutEffect(() => {
    let cancelled = false
    const el = containerRef.current
    if (!el) return
    
    setLoading(true)
    
    // Small delay for container to be sized
    const initTimer = setTimeout(() => {
      if (cancelled || !el || cyRef.current) return
      
      const cy = cytoscape({
        container: el,
        elements: elements,
        style: [
          // Center class
          {
            selector: 'node.center-class',
            style: {
              'shape': 'ellipse',
              'width': 80,
              'height': 55,
              'background-color': '#2e7d32',
              'border-color': '#66bb6a',
              'border-width': 3,
              'label': 'data(label)',
              'text-valign': 'center',
              'text-halign': 'center',
              'color': '#ffffff',
              'font-size': '11px',
              'font-weight': '600',
              'text-wrap': 'wrap',
              'text-max-width': '75px',
              'cursor': 'pointer'
            }
          },
          // Neighbor classes
          {
            selector: 'node.neighbor-class',
            style: {
              'shape': 'ellipse',
              'width': 60,
              'height': 40,
              'background-color': '#1565c0',
              'border-color': '#64b5f6',
              'border-width': 2,
              'label': 'data(label)',
              'text-valign': 'center',
              'text-halign': 'center',
              'color': '#ffffff',
              'font-size': '10px',
              'text-wrap': 'wrap',
              'text-max-width': '55px',
              'cursor': 'pointer'
            }
          },
          // Data property nodes (attributes)
          {
            selector: 'node.data-property-node',
            style: {
              'shape': 'round-rectangle',
              'width': 'label',
              'height': 24,
              'padding': '8px',
              'background-color': '#37474f',
              'border-color': '#546e7a',
              'border-width': 1,
              'label': 'data(label)',
              'text-valign': 'center',
              'text-halign': 'center',
              'color': '#eceff1',
              'font-size': '9px',
              'cursor': 'pointer'
            }
          },
          {
            selector: 'node.data-property-node.highlighted',
            style: {
              'background-color': '#1b5e20',
              'border-color': '#4caf50',
              'border-width': 2,
              'color': '#ffffff'
            }
          },
          {
            selector: 'node.data-property-node.selected',
            style: {
              'background-color': '#0d47a1',
              'border-color': '#42a5f5',
              'border-width': 2
            }
          },
          // Data property edges (dotted, subtle)
          {
            selector: 'edge.data-property-edge',
            style: {
              'width': 1,
              'line-color': '#546e7a',
              'line-style': 'dotted',
              'target-arrow-shape': 'none',
              'curve-style': 'bezier',
              'opacity': 0.6
            }
          },
          // Object property edges
          {
            selector: 'edge.object-property-edge',
            style: {
              'width': 2,
              'line-color': '#78909c',
              'target-arrow-color': '#78909c',
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier',
              'label': 'data(label)',
              'font-size': '8px',
              'color': '#b0bec5',
              'text-background-color': '#1a1a2e',
              'text-background-opacity': 0.9,
              'text-background-shape': 'round-rectangle',
              'text-background-padding': '2px',
              'arrow-scale': 0.8
            }
          },
          {
            selector: 'edge.object-property-edge.outgoing',
            style: {
              'line-color': '#4caf50',
              'target-arrow-color': '#4caf50'
            }
          },
          {
            selector: 'edge.object-property-edge.incoming',
            style: {
              'line-color': '#2196f3',
              'target-arrow-color': '#2196f3'
            }
          }
        ],
        layout: {
          name: 'cola',
          maxSimulationTime: 1500,
          fit: true,
          padding: 30,
          nodeSpacing: 25,
          edgeLength: 120
        },
        minZoom: 0.3,
        maxZoom: 2.5,
        wheelSensitivity: 0.3
      })
      
      cyRef.current = cy
      
      // Click handlers
      cy.on('tap', 'node.data-property-node', (e) => {
        const node = e.target
        const propertyUri = node.id()
        if (onSelectProperty) {
          cy.nodes('.data-property-node').removeClass('selected')
          node.addClass('selected')
          onSelectProperty(propertyUri)
        }
      })
      
      // Class node clicks - both neighbor and center can be clicked
      cy.on('tap', 'node.neighbor-class, node.center-class', (e) => {
        const node = e.target
        const classUri = node.id()
        if (onSelectClass) {
          onSelectClass(classUri)
        }
      })
      
      cy.on('ready', () => setLoading(false))
    }, 50)
    
    return () => {
      cancelled = true
      clearTimeout(initTimer)
      if (cyRef.current) {
        try { cyRef.current.destroy() } catch (e) {}
        cyRef.current = null
      }
    }
  }, []) // Run only once on mount
  
  // Update elements when data changes
  useEffect(() => {
    if (!cyRef.current) return
    
    const cy = cyRef.current
    cy.batch(() => {
      cy.elements().remove()
      cy.add(elements)
    })
    cy.layout({
      name: 'cola',
      maxSimulationTime: 1000,
      fit: true,
      padding: 30
    }).run()
  }, [elements])
  
  // Update selected property
  useEffect(() => {
    if (!cyRef.current) return
    
    cyRef.current.nodes('.data-property-node').removeClass('selected')
    if (selectedProperty) {
      const node = cyRef.current.getElementById(selectedProperty)
      if (node.length) {
        node.addClass('selected')
      }
    }
  }, [selectedProperty])
  
  const showEmptyMessage = elements.length === 0
  
  return (
    <div className="viewpoint-graph-container">
      <div ref={containerRef} className="viewpoint-graph-canvas" />
      {loading && (
        <div className="graph-loading">
          Building viewpoint...
        </div>
      )}
      {showEmptyMessage && !loading && (
        <div className="empty-graph-message">
          Select an entity class to see its viewpoint
        </div>
      )}
      <div className="graph-legend viewpoint-legend">
        <div className="legend-item">
          <span className="legend-circle focus">●</span> Entity class
        </div>
        <div className="legend-item">
          <span className="legend-circle neighbor">●</span> Connected class
        </div>
        <div className="legend-item">
          <span className="legend-rect attr">▬</span> Data property
        </div>
      </div>
    </div>
  )
}

function OntologyGraph({ 
  ontology, 
  highlightedProperties = [], 
  selectedProperty = null,
  onSelectProperty,
  selectedClass = null
}) {
  const containerRef = useRef(null)
  const cyRef = useRef(null)
  const [loading, setLoading] = useState(false)
  const [neighborhoodHops, setNeighborhoodHops] = useState(2)
  const initStarted = useRef(false)

  // Build elements with 2-hop neighborhood from selected class
  const elements = useMemo(() => {
    return buildOntologyElements(ontology, highlightedProperties, {
      maxClasses: 100,
      maxProperties: 200,
      allowDataProperties: false,
      focusClass: selectedClass || null,
      hops: neighborhoodHops
    })
  }, [ontology, highlightedProperties, selectedClass, neighborhoodHops])

  // Initialize Cytoscape with cola layout
  useLayoutEffect(() => {
    if (!containerRef.current) return
    if (initStarted.current) return
    initStarted.current = true
    
    setLoading(true)
    
    // Wait for container to have size
    const el = containerRef.current
    const initGraph = () => {
      if (el.clientWidth < 10 || el.clientHeight < 10) {
        setTimeout(initGraph, 50)
        return
      }
      
      // Destroy previous instance
      if (cyRef.current) {
        try { cyRef.current.destroy() } catch {}
      }
      
      // Skip if no elements
      if (elements.length === 0) {
        setLoading(false)
        return
      }
      
      const cy = cytoscape({
        container: el,
        elements: elements,
        style: [
          // Class nodes - ellipse shape (cleaner than rectangles)
          {
            selector: 'node.class-node',
            style: {
              'shape': 'ellipse',
              'width': 60,
              'height': 40,
              'background-color': '#1976d2',
              'border-color': '#42a5f5',
              'border-width': 2,
              'label': 'data(label)',
              'text-valign': 'center',
              'text-halign': 'center',
              'color': '#ffffff',
              'font-size': '10px',
              'font-weight': '500',
              'text-wrap': 'wrap',
              'text-max-width': '55px'
            }
          },
          // Focus class (center of neighborhood)
          {
            selector: 'node.focus-node',
            style: {
              'background-color': '#2e7d32',
              'border-color': '#66bb6a',
              'border-width': 3,
              'width': 70,
              'height': 50,
              'font-size': '11px',
              'font-weight': '600'
            }
          },
          // 1-hop neighbors
          {
            selector: 'node.neighbor-1',
            style: {
              'background-color': '#1565c0',
              'border-color': '#64b5f6'
            }
          },
          // 2-hop neighbors (more faded)
          {
            selector: 'node.neighbor-2',
            style: {
              'background-color': '#455a64',
              'border-color': '#78909c',
              'opacity': 0.8
            }
          },
          // Stub nodes (classes not in ontology)
          {
            selector: 'node.stub-node',
            style: {
              'background-color': '#546e7a',
              'border-color': '#78909c',
              'opacity': 0.6,
              'border-style': 'dashed'
            }
          },
          // Selected class
          {
            selector: 'node.class-node.selected-class',
            style: {
              'background-color': '#1565c0',
              'border-color': '#90caf9',
              'border-width': 4
            }
          },
          // Property edges - default
          {
            selector: 'edge.property-edge',
            style: {
              'width': 1.5,
              'line-color': '#616161',
              'target-arrow-color': '#616161',
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier',
              'label': 'data(label)',
              'font-size': '8px',
              'color': '#bdbdbd',
              'text-background-color': '#1a1a2e',
              'text-background-opacity': 0.85,
              'text-background-shape': 'round-rectangle',
              'text-background-padding': '2px',
              'text-wrap': 'wrap',
              'text-max-width': '90px',
              'arrow-scale': 0.7
            }
          },
          // Highlighted edges (recommendations)
          {
            selector: 'edge.highlighted',
            style: {
              'width': 2.5,
              'line-style': 'solid'
            }
          },
          {
            selector: 'edge.highlighted.high',
            style: {
              'line-color': '#4caf50',
              'target-arrow-color': '#4caf50',
              'color': '#81c784'
            }
          },
          {
            selector: 'edge.highlighted.medium',
            style: {
              'line-color': '#ff9800',
              'target-arrow-color': '#ff9800',
              'color': '#ffb74d'
            }
          },
          {
            selector: 'edge.highlighted.low',
            style: {
              'line-color': '#9e9e9e',
              'target-arrow-color': '#9e9e9e',
              'color': '#bdbdbd'
            }
          },
          // Selected edge
          {
            selector: 'edge.selected',
            style: {
              'width': 4,
              'line-color': '#3b82f6',
              'target-arrow-color': '#3b82f6',
              'color': '#60a5fa',
              'z-index': 100
            }
          }
        ],
        layout: {
          name: 'cola',
          maxSimulationTime: 800,
          fit: true,
          padding: 40,
          nodeSpacing: 30,
          edgeLengthVal: 100,
          animate: false
        },
        minZoom: 0.2,
        maxZoom: 3,
        wheelSensitivity: 0.3
      })
      
      // Handle edge clicks - select property
      cy.on('tap', 'edge.property-edge', (evt) => {
        const edge = evt.target
        const propertyUri = edge.data('propertyUri')
        if (onSelectProperty && propertyUri) {
          onSelectProperty(propertyUri)
        }
      })
      
      // Handle class node clicks - select class
      cy.on('tap', 'node.center-class, node.neighbor-class', (evt) => {
        const node = evt.target
        const classUri = node.id()
        if (onSelectClass && classUri) {
          onSelectClass(classUri)
        }
      })
      
      // Layout complete
      cy.once('layoutstop', () => {
        cy.fit(50)
        setLoading(false)
      })
      
      cyRef.current = cy
      
      // Expose for debugging
      if (typeof window !== 'undefined') {
        window.cyOntology = cy
      }
    }
    
    initGraph()
    
    return () => {
      if (cyRef.current) {
        try { cyRef.current.destroy() } catch {}
        cyRef.current = null
      }
      initStarted.current = false
    }
  }, []) // Only init once

  // Update elements when data changes
  useEffect(() => {
    if (!cyRef.current) return
    
    try {
      const cy = cyRef.current
      cy.batch(() => {
        cy.elements().remove()
        cy.add(elements)
      })
      cy.layout({ 
        name: 'cola', 
        maxSimulationTime: 500, 
        fit: true, 
        padding: 40,
        animate: false
      }).run()
    } catch (e) {
      console.error('Failed to update ontology graph:', e)
    }
  }, [elements])

  // Update selection styling
  useEffect(() => {
    if (!cyRef.current) return
    
    // Remove selection from all edges
    cyRef.current.edges().removeClass('selected')
    
    // Add selection to selected property edge
    if (selectedProperty) {
      const edge = cyRef.current.getElementById(`${selectedProperty}-edge`)
      if (edge.length) {
        edge.addClass('selected')
      }
    }
  }, [selectedProperty])

  // Update selected class styling
  useEffect(() => {
    if (!cyRef.current) return
    
    cyRef.current.nodes().removeClass('selected-class')
    
    if (selectedClass) {
      const node = cyRef.current.getElementById(selectedClass)
      if (node.length) {
        node.addClass('selected-class')
      }
    }
  }, [selectedClass])

  const showEmptyMessage = elements.length === 0

  return (
    <div className="ontology-graph-container">
      <div ref={containerRef} className="ontology-graph-canvas" />
      {loading && (
        <div className="graph-loading">
          Building ontology graph...
        </div>
      )}
      {showEmptyMessage && !loading && (
        <div className="empty-graph-message">
          {ontology.classes.length === 0 
            ? 'No ontology loaded. Select a repository with an ontology.'
            : 'No object properties found to visualize.'}
        </div>
      )}
      <div className="graph-controls">
        <label className="hops-control">
          <span>Depth:</span>
          <select 
            value={neighborhoodHops} 
            onChange={(e) => setNeighborhoodHops(Number(e.target.value))}
            className="hops-select"
          >
            <option value={1}>1 hop</option>
            <option value={2}>2 hops</option>
            <option value={3}>3 hops</option>
            <option value={0}>All</option>
          </select>
        </label>
      </div>
      <div className="graph-legend">
        <div className="legend-item">
          <span className="legend-circle focus">●</span> Target class
        </div>
        <div className="legend-item">
          <span className="legend-circle neighbor">●</span> Related class
        </div>
        <div className="legend-item">
          <span className="legend-arrow high">→</span> High match
        </div>
        <div className="legend-item">
          <span className="legend-arrow medium">→</span> Medium match
        </div>
        <div className="legend-item">
          <span className="legend-arrow low">→</span> Low match
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// Step 3: Column Mapping Wizard (Two-Phase: Class Selection → Column Mapping)
// ============================================================================

/**
 * Auto-detect which column is likely the subject identifier for this entity class
 */
function detectSubjectColumn(file, entityClassUri, ontology) {
  const headers = file.headers
  
  // Extract class name from URI
  let className = ''
  if (entityClassUri) {
    if (entityClassUri.includes('#')) className = entityClassUri.split('#').pop()
    else if (entityClassUri.includes('/')) className = entityClassUri.split('/').pop()
    className = className.toLowerCase()
  }
  
  // Priority 1: Column named "{class}_id" or "{class}id"
  for (const col of headers) {
    const normCol = col.toLowerCase().replace(/[_-]/g, '')
    if (normCol === `${className}id` || normCol === `${className.replace(/s$/, '')}id`) {
      return col
    }
  }
  
  // Priority 2: Column named just "id" or "identifier"
  for (const col of headers) {
    const normCol = col.toLowerCase().replace(/[_-]/g, '')
    if (normCol === 'id' || normCol === 'identifier') {
      return col
    }
  }
  
  // Priority 3: First column ending in "_id"
  for (const col of headers) {
    if (col.toLowerCase().endsWith('_id') || col.toLowerCase().endsWith('id')) {
      return col
    }
  }
  
  // Fallback: first column
  return headers[0]
}

function MappingWizardStep({ file, ontology, onComplete, onBack }) {
  // Phase: 'class-selection' or 'column-mapping'
  const [phase, setPhase] = useState('class-selection')
  
  // Class selection phase
  const [selectedEntityClass, setSelectedEntityClass] = useState(null)
  const [classRecommendations, setClassRecommendations] = useState([])
  const [classSearchQuery, setClassSearchQuery] = useState('')
  
  // Subject identifier column (detected or user-selected)
  const [subjectColumn, setSubjectColumn] = useState(null)
  
  // Column mapping phase - skip the subject column
  const [currentColumnIndex, setCurrentColumnIndex] = useState(0)
  const [mappings, setMappings] = useState({}) // { col: { predicate, isIri, iriPattern } }
  const [selectedProperty, setSelectedProperty] = useState(null)
  
  // Get columns to map (exclude subject column)
  const columnsToMap = useMemo(() => {
    return file.headers.filter(h => h !== subjectColumn)
  }, [file.headers, subjectColumn])
  
  const currentColumn = columnsToMap[currentColumnIndex]
  
  // Object property configuration (for IRI references vs literals)
  const [isObjectProperty, setIsObjectProperty] = useState(false)
  const [iriPattern, setIriPattern] = useState('')
  
  // Auto-suggest IRI pattern based on the property's RANGE (target class), not the property name
  const suggestIriPattern = useCallback((propUri, rangeUri) => {
    // Use the range (target class) to build the IRI pattern
    // e.g., property "placedBy" with range "Customer" → "http://example.org/customer/$(customer_id)"
    
    let entityType = ''
    
    if (rangeUri) {
      // Extract local name from range URI
      let rangeName = rangeUri
      if (rangeUri.includes(':')) {
        rangeName = rangeUri.split(':').pop()
      } else if (rangeUri.includes('#')) {
        rangeName = rangeUri.split('#').pop()
      } else if (rangeUri.includes('/')) {
        rangeName = rangeUri.split('/').pop()
      }
      entityType = rangeName.toLowerCase()
    } else {
      // Fallback: derive from property name (less ideal)
      let localName = propUri
      if (propUri.includes(':')) {
        localName = propUri.split(':').pop()
      } else if (propUri.includes('#')) {
        localName = propUri.split('#').pop()
      } else if (propUri.includes('/')) {
        localName = propUri.split('/').pop()
      }
      entityType = localName.toLowerCase()
      // Remove common prefixes when deriving from property name
      if (entityType.startsWith('has')) {
        entityType = entityType.slice(3)
      }
      if (entityType.startsWith('is')) {
        entityType = entityType.slice(2)
      }
    }
    
    return `http://example.org/${entityType}/$(${currentColumn})`
  }, [currentColumn])
  
  // Use API-based recommendations with local fallback (for column mapping phase)
  const { recommendations, usingEmbeddings, loading: recommendationsLoading } = useRecommendations(
    phase === 'column-mapping' ? currentColumn : null, ontology, 8
  )
  
  // Compute class recommendations on mount or when file/ontology changes
  useEffect(() => {
    if (ontology && ontology.classes && ontology.classes.length > 0) {
      const recs = recommendClassesForFile(file, ontology)
      setClassRecommendations(recs)
      // Auto-select the top recommendation if confidence is high
      if (recs.length > 0 && recs[0].confidence === 'high') {
        const topClassUri = recs[0].class.uri
        setSelectedEntityClass(topClassUri)
        // Auto-detect subject column for this class
        setSubjectColumn(detectSubjectColumn(file, topClassUri, ontology))
      }
    }
  }, [file, ontology])
  
  // When entity class changes, re-detect subject column
  useEffect(() => {
    if (selectedEntityClass) {
      const detected = detectSubjectColumn(file, selectedEntityClass, ontology)
      setSubjectColumn(detected)
    }
  }, [selectedEntityClass, file, ontology])
  
  // Get data properties for the selected entity class
  const classDataProperties = useMemo(() => {
    if (!selectedEntityClass || !ontology) return []
    return getClassDataProperties(selectedEntityClass, ontology)
  }, [selectedEntityClass, ontology])
  
  // Get object properties for the selected entity class (outgoing relationships)
  const classObjectProperties = useMemo(() => {
    if (!selectedEntityClass || !ontology) return []
    return getClassObjectProperties(selectedEntityClass, ontology)
  }, [selectedEntityClass, ontology])
  
  // Filter classes by search query
  const filteredClasses = useMemo(() => {
    if (!ontology?.classes) return []
    if (!classSearchQuery.trim()) return ontology.classes
    
    const query = classSearchQuery.toLowerCase()
    return ontology.classes.filter(cls => {
      const label = getLabel(cls.uri, cls.label).toLowerCase()
      const uri = cls.uri.toLowerCase()
      return label.includes(query) || uri.includes(query)
    })
  }, [ontology, classSearchQuery])
  
  // Sample values for current column (use index within columnsToMap)
  const sampleValues = useMemo(() => {
    if (!currentColumn) return []
    const colIndex = file.headers.indexOf(currentColumn)
    if (colIndex < 0) return []
    return file.rows.map(row => row[colIndex]).filter(Boolean).slice(0, 5)
  }, [file, currentColumn])
  
  const handleSelectProperty = useCallback((propertyUri) => {
    setSelectedProperty(propertyUri)
    // Auto-detect if this looks like an object property based on ontology range
    const prop = ontology.properties.find(p => p.uri === propertyUri)
    if (prop && prop.range && !prop.range.startsWith('xsd:')) {
      // Range is not XSD datatype - likely object property
      setIsObjectProperty(true)
      // Use the range (target class) for IRI pattern, not the property name
      setIriPattern(suggestIriPattern(propertyUri, prop.range))
    } else {
      setIsObjectProperty(false)
      setIriPattern('')
    }
  }, [ontology, suggestIriPattern])
  
  const handleConfirmMapping = useCallback(() => {
    if (selectedProperty) {
      const mappingConfig = {
        predicate: selectedProperty,
        isIri: isObjectProperty,
        iriPattern: isObjectProperty ? iriPattern : null,
      }
      setMappings(prev => ({
        ...prev,
        [currentColumn]: mappingConfig
      }))
    }
    
    if (currentColumnIndex < columnsToMap.length - 1) {
      setCurrentColumnIndex(prev => prev + 1)
      setSelectedProperty(null)
      setIsObjectProperty(false)
      setIriPattern('')
    } else {
      // All columns mapped - include subject column info in result
      const finalMapping = selectedProperty ? {
        predicate: selectedProperty,
        isIri: isObjectProperty,
        iriPattern: isObjectProperty ? iriPattern : null,
      } : null
      onComplete({
        subjectColumn,
        entityClass: selectedEntityClass,
        columnMappings: {
          ...mappings,
          ...(finalMapping && { [currentColumn]: finalMapping })
        }
      })
    }
  }, [selectedProperty, isObjectProperty, iriPattern, currentColumn, currentColumnIndex, columnsToMap.length, mappings, subjectColumn, selectedEntityClass, onComplete])
  
  const handleSkip = useCallback(() => {
    if (currentColumnIndex < columnsToMap.length - 1) {
      setCurrentColumnIndex(prev => prev + 1)
      setSelectedProperty(null)
      setIsObjectProperty(false)
      setIriPattern('')
    } else {
      onComplete({
        subjectColumn,
        entityClass: selectedEntityClass,
        columnMappings: mappings
      })
    }
  }, [currentColumnIndex, columnsToMap.length, mappings, subjectColumn, selectedEntityClass, onComplete])
  
  const handlePrevious = useCallback(() => {
    if (currentColumnIndex > 0) {
      const prevCol = columnsToMap[currentColumnIndex - 1]
      const prevMapping = mappings[prevCol]
      setCurrentColumnIndex(prev => prev - 1)
      if (prevMapping) {
        setSelectedProperty(prevMapping.predicate || null)
        setIsObjectProperty(prevMapping.isIri || false)
        setIriPattern(prevMapping.iriPattern || '')
      } else {
        setSelectedProperty(null)
        setIsObjectProperty(false)
        setIriPattern('')
      }
    }
  }, [currentColumnIndex, mappings, columnsToMap])

  // Class selection handlers
  const handleConfirmClass = useCallback(() => {
    if (selectedEntityClass) {
      setPhase('column-mapping')
    }
  }, [selectedEntityClass])
  
  const handleBackToClassSelection = useCallback(() => {
    setPhase('class-selection')
    setCurrentColumnIndex(0)
    setSelectedProperty(null)
  }, [])

  // Helper to get label from URI
  const getLabel = (uri, label) => {
    if (label) return label
    if (!uri) return ''
    if (uri.includes('#')) return uri.split('#').pop()
    if (uri.includes('/')) return uri.split('/').pop()
    return uri
  }

  const progress = columnsToMap.length > 0 ? ((currentColumnIndex + 1) / columnsToMap.length) * 100 : 0

  // ======== CLASS SELECTION PHASE ========
  if (phase === 'class-selection') {
    return (
      <div className="mapping-wizard class-selection-phase">
        <div className="wizard-header">
          <h2>Select Entity Class</h2>
          <p className="wizard-subtitle">
            What type of entities does this data represent?
          </p>
        </div>
        
        <div className="wizard-content">
          {/* Left: File summary + Class recommendations */}
          <div className="wizard-left">
            <div className="file-summary">
              <h3>
                <FileIcon size={18} />
                {file.name}
              </h3>
              <p>{file.headers.length} columns · {file.totalRows} rows</p>
              
              <div className="column-list">
                <h4>Columns</h4>
                <ul>
                  {file.headers.slice(0, 8).map((col, i) => (
                    <li key={i}>{col}</li>
                  ))}
                  {file.headers.length > 8 && (
                    <li className="more">+{file.headers.length - 8} more</li>
                  )}
                </ul>
              </div>
            </div>
            
            <div className="class-recommendations">
              <h4>
                <NetworkIcon size={16} />
                Recommended Classes
              </h4>
              
              {classRecommendations.length === 0 ? (
                <p className="no-recommendations">
                  No strong class matches found. Select a class from the graph or search below.
                </p>
              ) : (
                <ul className="class-recommendation-list">
                  {classRecommendations.map((rec, i) => (
                    <li 
                      key={rec.class.uri}
                      className={`class-rec-item ${rec.confidence} ${selectedEntityClass === rec.class.uri ? 'selected' : ''}`}
                      onClick={() => setSelectedEntityClass(rec.class.uri)}
                    >
                      <div className="rec-main">
                        <span className="rec-label">{getLabel(rec.class.uri, rec.class.label)}</span>
                      </div>
                      <div className="rec-details">
                        <span className={`rec-score ${rec.confidence}`}>
                          {Math.round(rec.score * 100)}%
                        </span>
                        <span className="rec-props">
                          {rec.dataPropertyCount} properties
                        </span>
                        {rec.matchingColumns.length > 0 && (
                          <span className="rec-matches" title={`Matches: ${rec.matchingColumns.join(', ')}`}>
                            {rec.matchingColumns.length} column matches
                          </span>
                        )}
                        {selectedEntityClass === rec.class.uri && (
                          <CheckIcon size={16} className="rec-check" />
                        )}
                      </div>
                    </li>
                  ))}
                </ul>
              )}
              
              {/* Class search */}
              <div className="class-search">
                <h5>
                  <SearchIcon size={14} />
                  Search Classes
                </h5>
                <input 
                  type="text"
                  className="class-search-input"
                  placeholder="Type to search classes..."
                  value={classSearchQuery}
                  onChange={(e) => setClassSearchQuery(e.target.value)}
                />
                {classSearchQuery && filteredClasses.length > 0 && (
                  <ul className="class-search-results">
                    {filteredClasses.slice(0, 10).map(cls => (
                      <li 
                        key={cls.uri}
                        className={`search-result-item ${selectedEntityClass === cls.uri ? 'selected' : ''}`}
                        onClick={() => {
                          setSelectedEntityClass(cls.uri)
                          setClassSearchQuery('')
                        }}
                      >
                        <span className="result-label">{getLabel(cls.uri, cls.label)}</span>
                        {selectedEntityClass === cls.uri && <CheckIcon size={14} />}
                      </li>
                    ))}
                    {filteredClasses.length > 10 && (
                      <li className="more-results">+{filteredClasses.length - 10} more classes</li>
                    )}
                  </ul>
                )}
                {classSearchQuery && filteredClasses.length === 0 && (
                  <p className="no-search-results">No classes match "{classSearchQuery}"</p>
                )}
              </div>
            </div>
          </div>
          
          {/* Right: Viewpoint graph */}
          <div className="wizard-right">
            {selectedEntityClass ? (
              <ViewpointGraph
                centerClass={selectedEntityClass}
                ontology={ontology}
                highlightedColumns={file.headers}
                onSelectClass={(classUri) => setSelectedEntityClass(classUri)}
              />
            ) : (
              <div className="ontology-overview">
                <h4>
                  <NetworkIcon size={20} />
                  Available Classes
                </h4>
                <p className="overview-hint">Click a class to see its properties and connections</p>
                <div className="class-grid">
                  {(ontology.classes || []).slice(0, 20).map(cls => {
                    const dataProps = getClassDataProperties(cls.uri, ontology)
                    const objProps = getClassObjectProperties(cls.uri, ontology)
                    return (
                      <button
                        key={cls.uri}
                        className="class-card"
                        onClick={() => setSelectedEntityClass(cls.uri)}
                      >
                        <span className="class-card-label">{getLabel(cls.uri, cls.label)}</span>
                        <span className="class-card-meta">
                          {dataProps.length} attrs · {objProps.length} rels
                        </span>
                      </button>
                    )
                  })}
                </div>
                {ontology.classes.length > 20 && (
                  <p className="more-classes">+{ontology.classes.length - 20} more classes (use search)</p>
                )}
              </div>
            )}
          </div>
        </div>
        
        {/* Subject column indicator */}
        {selectedEntityClass && subjectColumn && (
          <div className="subject-column-indicator">
            <span className="indicator-label">Subject Identifier:</span>
            <span className="indicator-value">{subjectColumn}</span>
            <span className="indicator-note">
              (This column will be used as the entity identifier, not mapped to a property)
            </span>
          </div>
        )}
        
        <div className="wizard-nav">
          <button className="btn btn-secondary" onClick={onBack}>
            Back to Preview
          </button>
          <button 
            className="btn btn-primary"
            onClick={handleConfirmClass}
            disabled={!selectedEntityClass}
          >
            Continue to Column Mapping
            <ChevronRightIcon size={16} />
          </button>
        </div>
      </div>
    )
  }

  // ======== COLUMN MAPPING PHASE ========
  return (
    <div className="mapping-wizard column-mapping-phase">
      {/* Phase indicator */}
      <div className="phase-indicator">
        <span className="phase-item completed" onClick={handleBackToClassSelection}>
          <CheckIcon size={14} />
          {getLabel(selectedEntityClass, null)}
        </span>
        <ChevronRightIcon size={14} className="phase-arrow" />
        <span className="phase-item active">
          Map Columns ({currentColumnIndex + 1}/{columnsToMap.length})
        </span>
      </div>
      
      {/* Progress bar */}
      <div className="wizard-progress">
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
        <span className="progress-text">
          Column {currentColumnIndex + 1} of {columnsToMap.length}
          {subjectColumn && <span className="subject-note"> ({subjectColumn} = subject ID)</span>}
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
              Properties of {getLabel(selectedEntityClass, null)}
            </h4>
            <p className="rec-hint">
              Select a data property to map this column
            </p>
            
            {/* Class data properties - prioritized */}
            {classDataProperties.length > 0 && (
              <ul className="recommendation-list class-properties">
                {classDataProperties.map((prop) => {
                  const propLabel = getLabel(prop.uri, prop.label)
                  const isMatch = propLabel.toLowerCase().includes(currentColumn.toLowerCase()) ||
                                  currentColumn.toLowerCase().includes(propLabel.toLowerCase())
                  return (
                    <li 
                      key={prop.uri}
                      className={`recommendation-item ${isMatch ? 'high' : 'medium'} ${selectedProperty === prop.uri ? 'selected' : ''}`}
                      onClick={() => handleSelectProperty(prop.uri)}
                    >
                      <div className="rec-main">
                        <span className="rec-label">{propLabel}</span>
                        <span className="rec-uri">{prop.uri}</span>
                      </div>
                      <div className="rec-meta">
                        {isMatch && <span className="rec-match-type">name match</span>}
                        {selectedProperty === prop.uri && (
                          <CheckIcon size={16} className="rec-check" />
                        )}
                      </div>
                    </li>
                  )
                })}
              </ul>
            )}
            
            {classDataProperties.length === 0 && (
              <p className="no-properties">
                No data properties defined for this class.
              </p>
            )}
            
            {/* Object properties - relationships to other classes */}
            {classObjectProperties.length > 0 && (
              <>
                <h5 className="object-properties-label">
                  <LinkIcon size={14} />
                  Relationships (Foreign Keys)
                </h5>
                <ul className="recommendation-list object-properties">
                  {classObjectProperties.map((prop) => {
                    const propLabel = getLabel(prop.uri, prop.label)
                    const rangeLabel = getLabel(prop.range, null)
                    const isMatch = propLabel.toLowerCase().includes(currentColumn.toLowerCase()) ||
                                    currentColumn.toLowerCase().includes(propLabel.toLowerCase()) ||
                                    rangeLabel.toLowerCase().includes(currentColumn.replace(/_?id$/i, '').toLowerCase())
                    return (
                      <li 
                        key={prop.uri}
                        className={`recommendation-item object-prop ${isMatch ? 'high' : 'low'} ${selectedProperty === prop.uri ? 'selected' : ''}`}
                        onClick={() => handleSelectProperty(prop.uri)}
                      >
                        <div className="rec-main">
                          <span className="rec-label">{propLabel}</span>
                          <span className="rec-arrow">→</span>
                          <span className="rec-range">{rangeLabel}</span>
                        </div>
                        <div className="rec-meta">
                          {isMatch && <span className="rec-match-type">FK match</span>}
                          {selectedProperty === prop.uri && (
                            <CheckIcon size={16} className="rec-check" />
                          )}
                        </div>
                      </li>
                    )
                  })}
                </ul>
              </>
            )}
            
            {/* General recommendations as fallback */}
            {recommendations.length > 0 && (
              <>
                <h5 className="other-recommendations-label">Other Suggestions</h5>
                <ul className="recommendation-list other-recommendations">
                  {recommendations.slice(0, 4).map((rec, i) => (
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
                        {rec.matchType && rec.matchType !== 'label' && (
                          <span className="rec-match-type" title={`Matched via ${rec.matchType}`}>
                            {rec.matchType}
                          </span>
                        )}
                        {selectedProperty === rec.property.uri && (
                          <CheckIcon size={16} className="rec-check" />
                        )}
                      </div>
                    </li>
                  ))}
                </ul>
              </>
            )}
            
            {recommendationsLoading && (
              <div className="recommendations-loading">
                <RefreshIcon size={20} className="spin" />
                <span>Loading suggestions...</span>
              </div>
            )}
            
            {/* Object Property Configuration - shows when a property is selected */}
            {selectedProperty && (
              <div className="object-property-config">
                <h4>Property Type</h4>
                <label className="property-type-toggle">
                  <input
                    type="checkbox"
                    checked={isObjectProperty}
                    onChange={(e) => {
                      setIsObjectProperty(e.target.checked)
                      if (e.target.checked && !iriPattern) {
                        // Find property range for IRI pattern suggestion
                        const prop = ontology.properties.find(p => p.uri === selectedProperty)
                        setIriPattern(suggestIriPattern(selectedProperty, prop?.range))
                      }
                    }}
                  />
                  <span className="toggle-label">
                    Object Property (IRI reference)
                  </span>
                </label>
                <p className="type-hint">
                  {isObjectProperty 
                    ? 'Values will be IRIs linking to other resources'
                    : 'Values will be literal data (strings, numbers, etc.)'}
                </p>
                
                {isObjectProperty && (
                  <div className="iri-pattern-config">
                    <label>IRI Pattern:</label>
                    <input
                      type="text"
                      value={iriPattern}
                      onChange={(e) => setIriPattern(e.target.value)}
                      placeholder={`http://example.org/entity/$(${currentColumn})`}
                      className="iri-pattern-input"
                    />
                    <p className="pattern-hint">
                      Use <code>$({currentColumn})</code> to reference column values
                    </p>
                    {sampleValues[0] && iriPattern && (
                      <div className="pattern-preview">
                        <span>Preview: </span>
                        <code>{iriPattern.replace(`$(${currentColumn})`, sampleValues[0])}</code>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
            
            <button 
              className="btn btn-sm btn-secondary skip-btn"
              onClick={handleSkip}
            >
              Skip this column
            </button>
          </div>
        </div>
        
        {/* Right: Viewpoint graph centered on selected class */}
        <div className="wizard-right">
          <ViewpointGraph
            centerClass={selectedEntityClass}
            ontology={ontology}
            highlightedColumns={[currentColumn]}
            selectedProperty={selectedProperty}
            onSelectProperty={handleSelectProperty}
            onSelectClass={(uri) => {
              // Clicking a neighbor class navigates to it
              setSelectedEntityClass(uri)
            }}
          />
        </div>
      </div>
      
      {/* Navigation */}
      <div className="wizard-nav">
        <button 
          className="btn btn-secondary"
          onClick={currentColumnIndex === 0 ? handleBackToClassSelection : handlePrevious}
        >
          {currentColumnIndex === 0 ? 'Change Entity Class' : 'Previous Column'}
        </button>
        
        <button 
          className="btn btn-primary"
          onClick={handleConfirmMapping}
          disabled={!selectedProperty}
        >
          {currentColumnIndex < columnsToMap.length - 1 ? (
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
  
  // Job-based processing for large files
  const [currentJobId, setCurrentJobId] = useState(null)
  const [jobProgress, setJobProgress] = useState(null)
  const pollingRef = useRef(null)
  
  // Large file threshold (5000 rows = use async job)
  const LARGE_FILE_THRESHOLD = 5000
  
  // Subject column selector - auto-detect ID column
  const [subjectColumn, setSubjectColumn] = useState(() => {
    // Try to auto-detect an ID column
    const idPatterns = ['id', 'sku', 'uuid', 'identifier', 'key', 'code', 'number', 'no']
    for (const header of file.headers) {
      const lower = header.toLowerCase().replace(/[_-]/g, '')
      if (idPatterns.some(p => lower === p || lower.endsWith(p) || lower.startsWith(p + '_'))) {
        return header
      }
    }
    // Fall back to first column
    return file.headers[0]
  })
  
  // Class type for instances
  const [selectedClass, setSelectedClass] = useState('')
  const [customClass, setCustomClass] = useState('') // For manual class entry
  
  // Auto-select first class when ontology loads
  useEffect(() => {
    if (ontology.classes && ontology.classes.length > 0 && !selectedClass) {
      setSelectedClass(ontology.classes[0].uri)
    }
  }, [ontology.classes])
  
  // Metadata options for RDF-Star generation
  const [enableMetadata, setEnableMetadata] = useState(true)
  const [metadataOptions, setMetadataOptions] = useState({
    source: '',
    confidence: '1.0',
    generateTimestamp: true,
  })
  const [exportMode, setExportMode] = useState('rdfstar') // 'plain' | 'rdfstar' | 'load'
  const [selectedRepo, setSelectedRepo] = useState('')
  const [repositories, setRepositories] = useState([])
  const [loadingRepos, setLoadingRepos] = useState(false)
  
  const mappedColumns = Object.entries(mappings).filter(([_, mapping]) => mapping && mapping.predicate)
  
  // Multi-column subject template support
  const [subjectColumns, setSubjectColumns] = useState([])
  const [subjectTemplate, setSubjectTemplate] = useState('')
  
  // Generate subject template from selected columns
  useEffect(() => {
    if (subjectColumns.length > 0) {
      const colRefs = subjectColumns.map(col => `$(${col})`).join('_')
      setSubjectTemplate(`http://example.org/resource/${colRefs}`)
    } else if (subjectColumn) {
      setSubjectTemplate(`http://example.org/resource/$(${subjectColumn})`)
    }
  }, [subjectColumns, subjectColumn])
  
  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
      }
    }
  }, [])
  
  // Poll for job status when a job is running
  useEffect(() => {
    if (!currentJobId) return
    
    const pollJob = async () => {
      try {
        const response = await fetch(`/api/etl/jobs/${currentJobId}`)
        if (!response.ok) {
          throw new Error('Failed to get job status')
        }
        
        const status = await response.json()
        setJobProgress(status)
        
        // Job completed or failed
        if (status.status === 'completed') {
          clearInterval(pollingRef.current)
          pollingRef.current = null
          
          // Fetch the result
          const resultResponse = await fetch(`/api/etl/jobs/${currentJobId}/result`)
          if (resultResponse.ok) {
            const result = await resultResponse.json()
            setConversionResult(result)
          } else {
            const error = await resultResponse.json()
            setConversionResult({ error: error.detail || 'Failed to get result' })
          }
          
          setIsConverting(false)
          setCurrentJobId(null)
          setJobProgress(null)
        } else if (status.status === 'failed') {
          clearInterval(pollingRef.current)
          pollingRef.current = null
          setConversionResult({ error: status.error || 'Job failed' })
          setIsConverting(false)
          setCurrentJobId(null)
          setJobProgress(null)
        }
      } catch (err) {
        console.error('Error polling job:', err)
      }
    }
    
    // Start polling
    pollJob() // Immediate first poll
    pollingRef.current = setInterval(pollJob, 1000) // Poll every second
    
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
      }
    }
  }, [currentJobId])
  
  // Load repositories for the "load to repo" option
  useEffect(() => {
    async function loadRepos() {
      setLoadingRepos(true)
      try {
        const response = await fetch('/api/repositories')
        if (response.ok) {
          const data = await response.json()
          setRepositories(data.repositories || [])
          if (data.repositories?.length > 0 && !selectedRepo) {
            setSelectedRepo(data.repositories[0].name)
          }
        }
      } catch (err) {
        console.error('Failed to load repositories:', err)
      } finally {
        setLoadingRepos(false)
      }
    }
    if (exportMode === 'load') {
      loadRepos()
    }
  }, [exportMode])
  
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
    s: ex:resource/$(${subjectColumn})
    po:
${mappingEntries}`
  }, [file, mappedColumns, ontology, subjectColumn])

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

  // Helper to build CSV content from file data
  const buildCsvContent = useCallback(() => {
    return [
      file.headers.join(','),
      ...file.rows.map(row => row.map(cell => {
        // Escape cells that contain commas or quotes
        if (typeof cell === 'string' && (cell.includes(',') || cell.includes('"'))) {
          return `"${cell.replace(/"/g, '""')}"`
        }
        return cell || ''
      }).join(','))
    ].join('\n')
  }, [file])

  // Run ETL transformation (sync for small files, async job for large files)
  const handleRunTransformation = useCallback(async () => {
    setIsConverting(true)
    setConversionResult(null)
    setJobProgress(null)
    
    try {
      // Build mapping config from our mappings with object property support
      const mappingConfig = {
        sources: [{
          type: 'csv',
          file: file.name,
        }],
        subject_template: subjectTemplate || `http://example.org/resource/$(${subjectColumn})`,
        mappings: mappedColumns.map(([col, mapping]) => ({
          source_column: col,
          predicate: mapping.predicate,
          is_iri: mapping.isIri || false,
          template: mapping.iriPattern || null,
        })),
        prefixes: ontology.prefixes,
        // Add class type if selected (plus owl:NamedIndividual is added by default)
        class_type: selectedClass || null,
        add_owl_individual: true,
      }
      
      // Create form data
      const formData = new FormData()
      
      // Re-create the file blob from the parsed data
      const csvContent = buildCsvContent()
      const dataBlob = new Blob([csvContent], { type: 'text/csv' })
      formData.append('data_file', dataBlob, file.name)
      formData.append('mapping', JSON.stringify(mappingConfig))
      formData.append('output_format', outputFormat)
      
      // Add metadata options for RDF-Star generation
      const metadataConfig = {
        enabled: exportMode === 'rdfstar' || exportMode === 'load',
        source: metadataOptions.source || `starchart:${file.name}`,
        confidence: parseFloat(metadataOptions.confidence) || 1.0,
        generate_timestamp: metadataOptions.generateTimestamp,
      }
      formData.append('metadata', JSON.stringify(metadataConfig))
      formData.append('export_mode', exportMode)
      
      // If loading to repository, include the repo name
      if (exportMode === 'load' && selectedRepo) {
        formData.append('repository', selectedRepo)
      }
      
      // For large files, use async job API with polling
      const isLargeFile = file.totalRows > LARGE_FILE_THRESHOLD
      
      if (isLargeFile) {
        // Use async job API
        const response = await fetch('/api/etl/jobs', {
          method: 'POST',
          body: formData,
        })
        
        if (!response.ok) {
          const error = await response.json()
          throw new Error(error.detail || 'Failed to create job')
        }
        
        const job = await response.json()
        setCurrentJobId(job.job_id)
        // Polling will be handled by the useEffect hook
        // isConverting stays true until job completes
      } else {
        // Use sync API for small files
        const response = await fetch('/api/etl/convert', {
          method: 'POST',
          body: formData,
        })
        
        if (!response.ok) {
          const error = await response.json()
          throw new Error(error.detail || 'Conversion failed')
        }
        
        const result = await response.json()
        setConversionResult(result)
        setIsConverting(false)
      }
    } catch (err) {
      setConversionResult({ error: err.message })
      setIsConverting(false)
    }
  }, [file, mappedColumns, ontology.prefixes, outputFormat, exportMode, metadataOptions, selectedRepo, subjectColumn, subjectTemplate, buildCsvContent, selectedClass])

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
      
      {/* Sub-step tabs for cleaner organization */}
      <div className="review-tabs">
        <button 
          className={`review-tab ${!conversionResult ? 'active' : ''}`}
          onClick={() => setConversionResult(null)}
        >
          <span className="tab-num">1</span>
          <span className="tab-label">Configure</span>
        </button>
        <button 
          className={`review-tab ${conversionResult ? 'active' : ''}`}
          disabled={!conversionResult && !isConverting}
        >
          <span className="tab-num">2</span>
          <span className="tab-label">Results</span>
        </button>
      </div>
      
      <div className="review-content">
        {/* Configuration Tab */}
        {!conversionResult && (
          <>
            {/* Subject column selector with multi-column support */}
            <div className="subject-column-selector">
              <h3>Subject Identifier</h3>
              <p className="hint">Select column(s) that uniquely identify each row (used for generating IRIs)</p>
              
              {/* Single column mode */}
              <div className="subject-mode-toggle">
                <label>
                  <input
                    type="radio"
                    name="subjectMode"
                    checked={subjectColumns.length === 0}
                    onChange={() => setSubjectColumns([])}
                    disabled={isConverting}
                  />
                  Single column
                </label>
                <label>
                  <input
                    type="radio"
                    name="subjectMode"
                    checked={subjectColumns.length > 0}
                    onChange={() => setSubjectColumns([subjectColumn])}
                    disabled={isConverting}
                  />
                  Multi-column composite key
                </label>
              </div>
              
              {subjectColumns.length === 0 ? (
                <div className="subject-select-row">
                  <label>ID Column:</label>
                  <select 
                    value={subjectColumn} 
                    onChange={e => setSubjectColumn(e.target.value)}
                    disabled={isConverting}
                  >
                    {file.headers.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
              ) : (
                <div className="multi-column-selector">
                  <label>Select columns (in order):</label>
                  <div className="column-checkboxes">
                    {file.headers.map(col => (
                      <label key={col} className="column-checkbox">
                        <input
                          type="checkbox"
                          checked={subjectColumns.includes(col)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSubjectColumns(prev => [...prev, col])
                            } else {
                              setSubjectColumns(prev => prev.filter(c => c !== col))
                            }
                          }}
                          disabled={isConverting}
                        />
                        {col}
                      </label>
                    ))}
                  </div>
                  {subjectColumns.length > 0 && (
                    <div className="selected-columns-preview">
                      <span>Order: </span>
                      {subjectColumns.map((col, i) => (
                        <span key={col} className="column-tag">
                          {col}
                          <button 
                            type="button"
                            className="remove-column"
                            onClick={() => setSubjectColumns(prev => prev.filter(c => c !== col))}
                          >×</button>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}
              
              <div className="iri-preview">
                → <code>&lt;{subjectTemplate || `http://example.org/resource/$(${subjectColumn})`}&gt;</code>
              </div>
            </div>
            
            {/* Class type selector */}
            <div className="class-type-selector">
              <h3>Instance Type</h3>
              <p className="hint">Select the rdf:type class for generated instances (owl:NamedIndividual is added automatically)</p>
              <div className="subject-select-row">
                <label>Class:</label>
                <select 
                  value={selectedClass} 
                  onChange={e => setSelectedClass(e.target.value)}
                  disabled={isConverting}
                >
                  <option value="">(none - only owl:NamedIndividual)</option>
                  {(ontology.classes || []).map(cls => (
                    <option key={cls.uri} value={cls.uri}>
                      {cls.label || cls.uri}
                    </option>
                  ))}
                </select>
                {selectedClass && (
                  <span className="type-preview">
                    → <code>a {selectedClass.includes(':') ? selectedClass : `<${selectedClass}>`}</code>
                  </span>
                )}
              </div>
            </div>
            
            {/* Collapsible mapping summary */}
            <details className="mapping-summary-details" open>
              <summary><h3>Column Mappings</h3></summary>
              <table className="summary-table">
                <thead>
                  <tr>
                    <th>CSV Column</th>
                    <th>→</th>
                    <th>Ontology Property</th>
                    <th>Type</th>
                  </tr>
                </thead>
                <tbody>
                  {file.headers.map(col => {
                    const mapping = mappings[col]
                    const propUri = mapping?.predicate
                    const propDef = propUri ? ontology.properties.find(p => p.uri === propUri) : null
                    return (
                      <tr key={col} className={mapping ? '' : 'unmapped'}>
                        <td>{col}</td>
                        <td>→</td>
                        <td>
                          {propDef ? (
                            <span className="mapped-property">
                              {propDef.label}
                              <span className="prop-uri">{propUri}</span>
                            </span>
                          ) : propUri ? (
                            <span className="mapped-property">
                              <span className="prop-uri">{propUri}</span>
                            </span>
                          ) : (
                            <span className="unmapped-label">(not mapped)</span>
                          )}
                        </td>
                        <td>
                          {mapping?.isIri ? (
                            <span className="property-type-badge iri" title={mapping.iriPattern || 'IRI reference'}>
                              IRI
                            </span>
                          ) : mapping ? (
                            <span className="property-type-badge literal">Literal</span>
                          ) : null}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </details>
            
            {/* Collapsible RML/YARRRML output */}
            <details className="rml-output-details">
              <summary>
                <h3>Generated Mapping</h3>
                <div className="format-selector">
                  <button 
                    className={`btn btn-xs ${exportFormat === 'rml' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={(e) => { e.stopPropagation(); setExportFormat('rml') }}
                  >
                    RML
                  </button>
                  <button 
                    className={`btn btn-xs ${exportFormat === 'yarrrml' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={(e) => { e.stopPropagation(); setExportFormat('yarrrml') }}
                  >
                    YARRRML
                  </button>
                </div>
              </summary>
              <div className="rml-content">
                <div className="rml-actions">
                  <button className="btn btn-sm btn-secondary" onClick={handleCopy}>
                    Copy
                  </button>
                  <button className="btn btn-sm btn-primary" onClick={handleDownload}>
                    <DownloadIcon size={14} /> Download
                  </button>
                </div>
                <pre className="rml-code">{currentOutput}</pre>
              </div>
            </details>

            {/* ETL Transformation Section */}
            <div className="etl-section">
              <div className="etl-header">
                <h3>Run Transformation</h3>
                <p className="etl-description">
                  Convert your CSV data to RDF using the mapping above
                </p>
              </div>
            </div>
          
            {/* Export Mode Selector */}
            <div className="export-mode-selector">
            <label>Export Mode:</label>
            <div className="mode-buttons">
              <button 
                className={`mode-btn ${exportMode === 'plain' ? 'active' : ''}`}
                onClick={() => setExportMode('plain')}
                disabled={isConverting}
              >
                <span className="mode-icon">📄</span>
                <span className="mode-label">Plain RDF</span>
                <span className="mode-desc">Standard triples only</span>
              </button>
              <button 
                className={`mode-btn ${exportMode === 'rdfstar' ? 'active' : ''}`}
                onClick={() => setExportMode('rdfstar')}
                disabled={isConverting}
              >
                <span className="mode-icon">⭐</span>
                <span className="mode-label">RDF-Star</span>
                <span className="mode-desc">With provenance metadata</span>
              </button>
              <button 
                className={`mode-btn ${exportMode === 'load' ? 'active' : ''}`}
                onClick={() => setExportMode('load')}
                disabled={isConverting}
              >
                <span className="mode-icon">📥</span>
                <span className="mode-label">Load to Repo</span>
                <span className="mode-desc">Direct import with metadata</span>
              </button>
            </div>
          </div>
          
          {/* Metadata Options (shown for RDF-Star and Load modes) */}
          {(exportMode === 'rdfstar' || exportMode === 'load') && (
            <div className="metadata-options">
              <h4>Provenance Metadata (RDF-Star)</h4>
              <p className="metadata-hint">
                These values will be attached to each generated triple using RDF-Star syntax
              </p>
              
              <div className="metadata-fields">
                <div className="meta-field">
                  <label>
                    <span className="field-name">prov:wasDerivedFrom</span>
                    <span className="field-hint">Data source identifier</span>
                  </label>
                  <input
                    type="text"
                    placeholder={`e.g., file:${file.name} or https://example.org/source`}
                    value={metadataOptions.source}
                    onChange={e => setMetadataOptions(prev => ({ ...prev, source: e.target.value }))}
                    disabled={isConverting}
                  />
                </div>
                
                <div className="meta-field">
                  <label>
                    <span className="field-name">prov:value (confidence)</span>
                    <span className="field-hint">Confidence score (0.0 - 1.0)</span>
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    value={metadataOptions.confidence}
                    onChange={e => setMetadataOptions(prev => ({ ...prev, confidence: e.target.value }))}
                    disabled={isConverting}
                  />
                </div>
                
                <div className="meta-field checkbox-field">
                  <label>
                    <input
                      type="checkbox"
                      checked={metadataOptions.generateTimestamp}
                      onChange={e => setMetadataOptions(prev => ({ ...prev, generateTimestamp: e.target.checked }))}
                      disabled={isConverting}
                    />
                    <span className="field-name">prov:generatedAtTime</span>
                    <span className="field-hint">Auto-generate timestamp</span>
                  </label>
                </div>
              </div>
              
              <div className="metadata-preview">
                <span className="preview-label">Preview:</span>
                <code className="preview-code">
                  {'<<:subject :predicate "value">> prov:wasDerivedFrom <'}
                  {metadataOptions.source || `starchart:${file.name}`}
                  {'> ; prov:value '}
                  {metadataOptions.confidence || '1.0'}
                  {metadataOptions.generateTimestamp && ' ; prov:generatedAtTime "2024-..."^^xsd:dateTime'}
                  {' .'}
                </code>
              </div>
            </div>
          )}
          
          {/* Repository Selector (shown for Load mode) */}
          {exportMode === 'load' && (
            <div className="repo-selector">
              <label>Target Repository:</label>
              {loadingRepos ? (
                <span className="loading">Loading repositories...</span>
              ) : repositories.length > 0 ? (
                <select
                  value={selectedRepo}
                  onChange={e => setSelectedRepo(e.target.value)}
                  disabled={isConverting}
                >
                  {repositories.map(repo => (
                    <option key={repo.name} value={repo.name}>
                      {repo.name} ({repo.triple_count?.toLocaleString() || 0} triples)
                    </option>
                  ))}
                </select>
              ) : (
                <span className="no-repos">No repositories found. Create one first.</span>
              )}
            </div>
          )}
          
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
              disabled={isConverting || mappedColumns.length === 0 || (exportMode === 'load' && !selectedRepo)}
            >
              {isConverting ? (
                <>
                  <RefreshIcon size={14} className="spin" /> 
                  {jobProgress ? `Processing... ${jobProgress.progress_percent}%` : 'Converting...'}
                </>
              ) : exportMode === 'load' ? (
                <>
                  <PlayIcon size={14} /> Transform & Load
                </>
              ) : (
                <>
                  <PlayIcon size={14} /> Run Transformation
                </>
              )}
            </button>
          </div>
          
          {/* Job Progress Display */}
          {jobProgress && (
            <div className={`job-progress ${jobProgress.status || ''}`}>
              <div className="progress-header">
                <span className="progress-phase">
                  {jobProgress.status === 'completed' ? '✓ Complete' :
                   jobProgress.status === 'failed' ? '✗ Failed' :
                   jobProgress.current_phase || 'Processing...'}
                </span>
                <span className="progress-stats">
                  {jobProgress.processed_rows?.toLocaleString() || 0} / {jobProgress.total_rows?.toLocaleString() || 0} rows
                </span>
              </div>
              <div className="progress-bar-container">
                <div 
                  className="progress-bar" 
                  style={{ width: `${jobProgress.progress_percent || 0}%` }}
                />
              </div>
              <div className="progress-details">
                <span>{(jobProgress.triple_count || 0).toLocaleString()} triples generated</span>
                {(jobProgress.annotation_count || 0) > 0 && (
                  <span> • {jobProgress.annotation_count.toLocaleString()} annotations</span>
                )}
                {jobProgress.error && (
                  <span className="progress-error"> • {jobProgress.error}</span>
                )}
              </div>
            </div>
          )}
          </>
        )}
        
        {/* Results Tab - shown when conversion is complete */}
        {conversionResult && (
          <div className="results-tab-content">
            <div className={`etl-result ${conversionResult.error ? 'error' : 'success'}`}>
              {conversionResult.error ? (
                <>
                  <AlertIcon size={16} />
                  <span>Error: {conversionResult.error}</span>
                  <button 
                    className="btn btn-sm btn-secondary"
                    onClick={() => setConversionResult(null)}
                  >
                    Back to Configuration
                  </button>
                </>
              ) : (
                <>
                  <div className="result-header">
                    <CheckIcon size={16} />
                    <span>
                      {exportMode === 'load' 
                        ? (conversionResult.loaded 
                            ? `Successfully loaded ${conversionResult.triple_count} triples to ${selectedRepo}`
                            : `Generated ${conversionResult.triple_count} triples (load failed - see warnings)`)
                        : `Generated ${conversionResult.triple_count} triples`
                      }
                      {conversionResult.annotation_count > 0 && (
                        <span className="annotation-count">
                          {' '}(+ {conversionResult.annotation_count} RDF-Star annotations)
                        </span>
                      )}
                    </span>
                  </div>
                  
                  {/* Show warnings if any */}
                  {conversionResult.warnings?.length > 0 && (
                    <div className="result-warnings">
                      <AlertIcon size={14} />
                      <span>Warnings:</span>
                      <ul>
                        {conversionResult.warnings.map((w, i) => (
                          <li key={i}>{w}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  <div className="result-actions">
                    {exportMode !== 'load' && (
                      <button className="btn btn-primary" onClick={handleDownloadRdf}>
                        <DownloadIcon size={14} /> Download RDF
                      </button>
                    )}
                    {exportMode === 'load' && !conversionResult.loaded && (
                      <button className="btn btn-primary" onClick={handleDownloadRdf}>
                        <DownloadIcon size={14} /> Download RDF Instead
                      </button>
                    )}
                    <button 
                      className="btn btn-secondary"
                      onClick={() => setConversionResult(null)}
                    >
                      Back to Configuration
                    </button>
                  </div>
                  <pre className="rdf-preview">
                    {conversionResult.rdf_content?.slice(0, 4000)}
                    {conversionResult.rdf_content?.length > 4000 && '\n\n... (truncated)'}
                  </pre>
                </>
              )}
            </div>
          </div>
        )}
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
// Ontology Loading Hook - Fetches from current repository
// ============================================================================

function useOntologyLoader(currentRepo) {
  const [ontology, setOntology] = useState(EMPTY_ONTOLOGY)
  const [ontologyName, setOntologyName] = useState('No repository selected')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  // Fetch ontology from repository when it changes
  useEffect(() => {
    if (!currentRepo) {
      setOntology(EMPTY_ONTOLOGY)
      setOntologyName('No repository selected')
      setError(null)
      return
    }

    const controller = new AbortController()
    
    async function loadFromRepo() {
      setIsLoading(true)
      setError(null)
      try {
        // Add timeout for slow repositories
        const timeoutId = setTimeout(() => controller.abort(), 10000) // 10 second timeout
        
        const response = await fetch(`/api/repositories/${currentRepo}/ontology`, {
          signal: controller.signal,
        })
        
        clearTimeout(timeoutId)
        
        if (!response.ok) {
          throw new Error(`Failed to load ontology: ${response.statusText}`)
        }
        const data = await response.json()
        
        if (data.class_count === 0 && data.property_count === 0) {
          setOntology(EMPTY_ONTOLOGY)
          setOntologyName(`${currentRepo} (no ontology found)`)
          setError('No RDFS/OWL classes or properties found in this repository. Import an ontology first.')
        } else {
          setOntology({
            classes: data.classes || [],
            properties: data.properties || [],
            prefixes: data.prefixes || {},
          })
          setOntologyName(`${currentRepo} (${data.class_count} classes, ${data.property_count} properties)`)
        }
      } catch (err) {
        if (err.name === 'AbortError') {
          setError('Ontology loading timed out. The repository may be too large.')
        } else {
          setError(err.message)
        }
        setOntology(EMPTY_ONTOLOGY)
        setOntologyName(`${currentRepo} (error)`)
        console.error('Failed to load ontology from repository:', err)
      } finally {
        setIsLoading(false)
      }
    }

    loadFromRepo()
    
    return () => controller.abort()
  }, [currentRepo])

  // Manual refresh
  const refresh = useCallback(async () => {
    if (!currentRepo) return
    
    setIsLoading(true)
    setError(null)
    try {
      const response = await fetch(`/api/repositories/${currentRepo}/ontology`)
      if (!response.ok) {
        throw new Error(`Failed to load ontology: ${response.statusText}`)
      }
      const data = await response.json()
      
      if (data.class_count === 0 && data.property_count === 0) {
        setOntology(EMPTY_ONTOLOGY)
        setOntologyName(`${currentRepo} (no ontology found)`)
        setError('No RDFS/OWL classes or properties found in this repository.')
      } else {
        setOntology({
          classes: data.classes || [],
          properties: data.properties || [],
          prefixes: data.prefixes || {},
        })
        setOntologyName(`${currentRepo} (${data.class_count} classes, ${data.property_count} properties)`)
      }
    } catch (err) {
      setError(err.message)
      console.error('Failed to refresh ontology:', err)
    } finally {
      setIsLoading(false)
    }
  }, [currentRepo])

  return {
    ontology,
    ontologyName,
    isLoading,
    error,
    refresh,
  }
}

// ============================================================================
// Ontology Status Component - Shows what's loaded from repository
// ============================================================================

function OntologyStatus({ 
  currentRepo,
  ontologyName, 
  isLoading, 
  error,
  onRefresh 
}) {
  return (
    <div className="ontology-selector">
      <div className="ontology-current">
        <span className="ontology-label">Ontology:</span>
        <span className="ontology-name">{ontologyName}</span>
        {isLoading ? (
          <span className="ontology-loading">⏳</span>
        ) : (
          <button 
            className="ontology-refresh-btn" 
            onClick={onRefresh}
            title="Refresh ontology from repository"
          >
            <RefreshIcon size={14} />
          </button>
        )}
      </div>
      
      {error && <div className="ontology-error">{error}</div>}
      
      {!currentRepo && (
        <div className="ontology-hint">
          Select a repository to load its ontology
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Main Starchart Component
// ============================================================================

export function Starchart({ theme = 'dark', currentRepo }) {
  const [step, setStep] = useState('upload') // upload | preview | mapping | review
  const [file, setFile] = useState(null)
  const [mappings, setMappings] = useState({})
  
  // Use the ontology loader hook - fetches from current repository
  const {
    ontology,
    ontologyName,
    isLoading: ontologyLoading,
    error: ontologyError,
    refresh: refreshOntology,
  } = useOntologyLoader(currentRepo)

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
        
        {/* Ontology Status - shows what's loaded from repository */}
        <OntologyStatus
          currentRepo={currentRepo}
          ontologyName={ontologyName}
          isLoading={ontologyLoading}
          error={ontologyError}
          onRefresh={refreshOntology}
        />
        
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
