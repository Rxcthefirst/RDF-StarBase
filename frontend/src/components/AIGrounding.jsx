import { useState, useEffect } from 'react'
import {
  ZapIcon, SearchIcon, CheckCircleIcon, InfoIcon, CopyIcon,
  ChevronDownIcon, PlayIcon, BookIcon, CodeIcon, GlobeIcon
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
// API Endpoint Card
// ============================================================================
function EndpointCard({ endpoint, method, description, children, isOpen, onToggle }) {
  const methodColors = {
    GET: 'var(--success)',
    POST: 'var(--accent-color)',
    DELETE: 'var(--error)',
  }

  return (
    <div className={`endpoint-card ${isOpen ? 'open' : ''}`}>
      <div className="endpoint-header" onClick={onToggle}>
        <span className="endpoint-method" style={{ background: methodColors[method] }}>
          {method}
        </span>
        <code className="endpoint-path">{endpoint}</code>
        <span className="endpoint-desc">{description}</span>
        <ChevronDownIcon size={16} className={`chevron ${isOpen ? 'open' : ''}`} />
      </div>
      {isOpen && (
        <div className="endpoint-content">
          {children}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Code Example with Copy
// ============================================================================
function CodeExample({ title, language, code }) {
  const [copied, setCopied] = useState(false)

  const copyCode = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="code-example">
      <div className="code-header">
        <span className="code-title">{title}</span>
        <span className="code-lang">{language}</span>
        <button className="copy-btn" onClick={copyCode}>
          {copied ? <CheckCircleIcon size={14} /> : <CopyIcon size={14} />}
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <pre className="code-block"><code>{code}</code></pre>
    </div>
  )
}

// ============================================================================
// Interactive Query Demo
// ============================================================================
function QueryDemo({ currentRepo }) {
  const [subject, setSubject] = useState('')
  const [predicate, setPredicate] = useState('')
  const [minConfidence, setMinConfidence] = useState('medium')
  const [limit, setLimit] = useState(10)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const executeQuery = async () => {
    if (!currentRepo) {
      setError('No repository selected')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const payload = {
        subject: subject || null,
        predicate: predicate || null,
        min_confidence: minConfidence,
        limit: limit,
        include_inferred: true,
      }
      const data = await fetchJson(`/repositories/${currentRepo}/ai/query`, {
        method: 'POST',
        body: JSON.stringify(payload)
      })
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="demo-panel">
      <h4><PlayIcon size={16} /> Try It: Query Facts</h4>
      
      <div className="demo-form">
        <div className="form-row">
          <label>
            Subject (IRI)
            <input 
              type="text" 
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              placeholder="http://example.org/entity (optional)"
            />
          </label>
          <label>
            Predicate (IRI)
            <input 
              type="text" 
              value={predicate}
              onChange={(e) => setPredicate(e.target.value)}
              placeholder="http://schema.org/name (optional)"
            />
          </label>
        </div>
        <div className="form-row">
          <label>
            Minimum Confidence
            <select value={minConfidence} onChange={(e) => setMinConfidence(e.target.value)}>
              <option value="high">High (≥0.9)</option>
              <option value="medium">Medium (≥0.7)</option>
              <option value="low">Low (≥0.5)</option>
              <option value="any">Any (≥0.0)</option>
            </select>
          </label>
          <label>
            Limit
            <input 
              type="number" 
              value={limit}
              onChange={(e) => setLimit(parseInt(e.target.value) || 10)}
              min={1}
              max={100}
            />
          </label>
          <button className="btn primary" onClick={executeQuery} disabled={loading}>
            <SearchIcon size={16} />
            {loading ? 'Querying...' : 'Query Facts'}
          </button>
        </div>
      </div>

      {error && <div className="demo-error">{error}</div>}
      
      {result && (
        <div className="demo-result">
          <div className="result-meta">
            <span>Found <strong>{result.filtered_count}</strong> of {result.total_count} facts</span>
            <span>Confidence threshold: {result.confidence_threshold}</span>
            <span>Sources: {result.sources_used?.join(', ') || 'none'}</span>
          </div>
          
          {result.facts?.length > 0 ? (
            <div className="facts-list">
              {result.facts.map((fact, i) => (
                <div key={i} className="fact-card">
                  <div className="fact-triple">
                    <span className="fact-subject" title={fact.subject}>{getLocalName(fact.subject)}</span>
                    <span className="fact-predicate" title={fact.predicate}>{getLocalName(fact.predicate)}</span>
                    <span className="fact-object">{String(fact.object)}</span>
                  </div>
                  <div className="fact-citation">
                    <span className="citation-source">📚 {fact.citation.source}</span>
                    <span className="citation-confidence">🎯 {(fact.citation.confidence * 100).toFixed(0)}%</span>
                    <span className="citation-hash">#{fact.citation.fact_hash}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="no-results">No facts found matching your criteria</p>
          )}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Interactive Verify Demo
// ============================================================================
function VerifyDemo({ currentRepo }) {
  const [subject, setSubject] = useState('')
  const [predicate, setPredicate] = useState('')
  const [expectedObject, setExpectedObject] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const verifyClaim = async () => {
    if (!subject || !predicate) {
      setError('Subject and predicate are required')
      return
    }
    if (!currentRepo) {
      setError('No repository selected')
      return
    }
    
    setLoading(true)
    setError(null)
    try {
      const payload = {
        subject,
        predicate,
        expected_object: expectedObject || null,
        min_confidence: 'medium',
      }
      const data = await fetchJson(`/repositories/${currentRepo}/ai/verify`, {
        method: 'POST',
        body: JSON.stringify(payload)
      })
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="demo-panel">
      <h4><CheckCircleIcon size={16} /> Try It: Verify Claim</h4>
      
      <div className="demo-form">
        <div className="form-row">
          <label>
            Subject (IRI) *
            <input 
              type="text" 
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              placeholder="http://example.org/customer/123"
              required
            />
          </label>
          <label>
            Predicate (IRI) *
            <input 
              type="text" 
              value={predicate}
              onChange={(e) => setPredicate(e.target.value)}
              placeholder="http://schema.org/name"
              required
            />
          </label>
        </div>
        <div className="form-row">
          <label>
            Expected Value (optional)
            <input 
              type="text" 
              value={expectedObject}
              onChange={(e) => setExpectedObject(e.target.value)}
              placeholder="Alice Johnson"
            />
          </label>
          <button className="btn primary" onClick={verifyClaim} disabled={loading}>
            <CheckCircleIcon size={16} />
            {loading ? 'Verifying...' : 'Verify Claim'}
          </button>
        </div>
      </div>

      {error && <div className="demo-error">{error}</div>}
      
      {result && (
        <div className="demo-result">
          <div className={`verification-status ${result.claim_supported ? 'supported' : 'not-supported'}`}>
            <span className="status-icon">{result.claim_supported ? '✅' : '❌'}</span>
            <span className="status-text">
              {result.claim_supported ? 'Claim Supported' : 'Claim Not Supported'}
            </span>
            {result.confidence && (
              <span className="status-confidence">
                Confidence: {(result.confidence * 100).toFixed(0)}%
              </span>
            )}
            {result.has_conflicts && (
              <span className="status-conflicts">⚠️ Conflicts detected</span>
            )}
          </div>
          
          <div className="recommendation-box">
            <strong>AI Recommendation:</strong>
            <p>{result.recommendation}</p>
          </div>
          
          {result.supporting_facts?.length > 0 && (
            <div className="evidence-section">
              <h5>Supporting Evidence ({result.supporting_facts.length})</h5>
              {result.supporting_facts.slice(0, 3).map((fact, i) => (
                <div key={i} className="evidence-item supporting">
                  <span>{getLocalName(fact.predicate)}: {String(fact.object)}</span>
                  <span className="evidence-source">from {fact.citation.source}</span>
                </div>
              ))}
            </div>
          )}
          
          {result.contradicting_facts?.length > 0 && (
            <div className="evidence-section">
              <h5>Contradicting Evidence ({result.contradicting_facts.length})</h5>
              {result.contradicting_facts.slice(0, 3).map((fact, i) => (
                <div key={i} className="evidence-item contradicting">
                  <span>{getLocalName(fact.predicate)}: {String(fact.object)}</span>
                  <span className="evidence-source">from {fact.citation.source}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Interactive Context Demo
// ============================================================================
function ContextDemo({ currentRepo }) {
  const [entityIri, setEntityIri] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const getContext = async () => {
    if (!entityIri) {
      setError('Entity IRI is required')
      return
    }
    if (!currentRepo) {
      setError('No repository selected')
      return
    }
    
    setLoading(true)
    setError(null)
    try {
      const encoded = encodeURIComponent(entityIri)
      const data = await fetchJson(`/repositories/${currentRepo}/ai/context/${encoded}?min_confidence=low&limit=50`)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="demo-panel">
      <h4><GlobeIcon size={16} /> Try It: Entity Context</h4>
      
      <div className="demo-form">
        <div className="form-row full">
          <label>
            Entity IRI *
            <input 
              type="text" 
              value={entityIri}
              onChange={(e) => setEntityIri(e.target.value)}
              placeholder="http://example.org/customer/123"
              required
            />
          </label>
          <button className="btn primary" onClick={getContext} disabled={loading}>
            <SearchIcon size={16} />
            {loading ? 'Loading...' : 'Get Context'}
          </button>
        </div>
      </div>

      {error && <div className="demo-error">{error}</div>}
      
      {result && (
        <div className="demo-result">
          <div className="context-summary">
            <div className="summary-item">
              <span className="summary-label">Entity</span>
              <span className="summary-value">{getLocalName(result.entity)}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Facts</span>
              <span className="summary-value">{result.facts?.length || 0}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Sources</span>
              <span className="summary-value">{result.sources?.join(', ') || 'none'}</span>
            </div>
          </div>
          
          {result.confidence_summary && (
            <div className="confidence-breakdown">
              <span>🟢 High: {result.confidence_summary.high_confidence_count}</span>
              <span>🟡 Medium: {result.confidence_summary.medium_confidence_count}</span>
              <span>🔴 Low: {result.confidence_summary.low_confidence_count}</span>
            </div>
          )}
          
          {result.facts?.length > 0 && (
            <div className="facts-list compact">
              {result.facts.slice(0, 10).map((fact, i) => (
                <div key={i} className="fact-row">
                  <span className="fact-pred">{getLocalName(fact.predicate)}</span>
                  <span className="fact-val">{String(fact.object)}</span>
                  <span className="fact-conf">{(fact.citation.confidence * 100).toFixed(0)}%</span>
                </div>
              ))}
              {result.facts.length > 10 && (
                <div className="more-facts">...and {result.facts.length - 10} more</div>
              )}
            </div>
          )}
          
          {result.related_entities?.length > 0 && (
            <div className="related-entities">
              <h5>Related Entities</h5>
              <div className="entity-chips">
                {result.related_entities.slice(0, 8).map((entity, i) => (
                  <span 
                    key={i} 
                    className="entity-chip"
                    onClick={() => {
                      setEntityIri(entity)
                      getContext()
                    }}
                    title={entity}
                  >
                    {getLocalName(entity)}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Helper to get local name from URI
function getLocalName(uri) {
  if (!uri) return uri
  if (typeof uri !== 'string') return String(uri)
  if (uri.includes('#')) return uri.split('#').pop()
  return uri.split('/').pop()
}

// ============================================================================
// Main AI Grounding Component
// ============================================================================
export default function AIGrounding({ currentRepo }) {
  const [activeSection, setActiveSection] = useState('overview')
  const [openEndpoints, setOpenEndpoints] = useState({})
  const [health, setHealth] = useState(null)

  useEffect(() => {
    if (currentRepo) {
      fetchJson(`/repositories/${currentRepo}/ai/health`)
        .then(setHealth)
        .catch(() => setHealth({ status: 'unknown' }))
    } else {
      setHealth({ status: 'no-repo' })
    }
  }, [currentRepo])

  const toggleEndpoint = (key) => {
    setOpenEndpoints(prev => ({ ...prev, [key]: !prev[key] }))
  }

  const sections = [
    { id: 'overview', label: 'Overview', icon: InfoIcon },
    { id: 'query', label: 'Query Facts', icon: SearchIcon },
    { id: 'verify', label: 'Verify Claims', icon: CheckCircleIcon },
    { id: 'context', label: 'Entity Context', icon: GlobeIcon },
    { id: 'integration', label: 'Integration', icon: CodeIcon },
  ]

  // Show message if no repository selected
  if (!currentRepo) {
    return (
      <div className="ai-grounding">
        <div className="no-repo-message">
          <ZapIcon size={48} />
          <h2>No Repository Selected</h2>
          <p>Select a repository from the header dropdown to use the AI Grounding API.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="ai-grounding">
      {/* Sidebar Navigation */}
      <aside className="ai-sidebar">
        <div className="sidebar-header">
          <h2><ZapIcon size={18} /> AI Grounding API</h2>
          {health && (
            <span className={`health-badge ${health.status}`}>
              {health.status === 'healthy' ? '● Online' : '○ Offline'}
            </span>
          )}
        </div>
        
        <nav className="ai-nav">
          {sections.map(section => (
            <button
              key={section.id}
              className={`nav-item ${activeSection === section.id ? 'active' : ''}`}
              onClick={() => setActiveSection(section.id)}
            >
              <section.icon size={16} />
              {section.label}
            </button>
          ))}
        </nav>
        
        <div className="sidebar-footer">
          <a href="/docs#/AI%20Grounding" target="_blank" rel="noopener noreferrer" className="docs-link">
            <BookIcon size={14} />
            OpenAPI Docs
          </a>
        </div>
      </aside>

      {/* Main Content */}
      <main className="ai-main">
        {/* Overview Section */}
        {activeSection === 'overview' && (
          <div className="ai-section">
            <h2>AI Grounding API</h2>
            <p className="section-intro">
              Build trustworthy AI applications with provenance-backed knowledge. 
              The AI Grounding API is designed for LLM tool calls and RAG pipelines,
              providing structured fact retrieval with citations.
            </p>
            
            <div className="feature-grid">
              <div className="feature-card">
                <div className="feature-icon"><SearchIcon size={24} /></div>
                <h3>Query Facts</h3>
                <p>Retrieve facts with full provenance chains for grounding AI responses.</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon"><CheckCircleIcon size={24} /></div>
                <h3>Verify Claims</h3>
                <p>Check if a claim is supported by the knowledge base before stating it.</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon"><GlobeIcon size={24} /></div>
                <h3>Entity Context</h3>
                <p>Get everything known about an entity in a single call.</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon"><ZapIcon size={24} /></div>
                <h3>Materialize</h3>
                <p>Run RDFS/OWL reasoning to derive new facts automatically.</p>
              </div>
            </div>
            
            <div className="api-overview">
              <h3>Endpoints</h3>
              
              <EndpointCard
                endpoint="/ai/query"
                method="POST"
                description="Query facts for AI grounding"
                isOpen={openEndpoints['query']}
                onToggle={() => toggleEndpoint('query')}
              >
                <p>Retrieve facts with provenance for RAG pipelines. Filter by subject, predicate, confidence, and freshness.</p>
                <CodeExample 
                  title="Request"
                  language="json"
                  code={`{
  "subject": "http://example.org/customer/123",
  "predicate": null,
  "min_confidence": "medium",
  "max_age_days": 30,
  "limit": 100
}`}
                />
              </EndpointCard>
              
              <EndpointCard
                endpoint="/ai/verify"
                method="POST"
                description="Verify a claim against the knowledge base"
                isOpen={openEndpoints['verify']}
                onToggle={() => toggleEndpoint('verify')}
              >
                <p>Check if a statement is supported, with evidence for and against. Use before making claims in AI responses.</p>
                <CodeExample 
                  title="Request"
                  language="json"
                  code={`{
  "subject": "http://example.org/customer/123",
  "predicate": "http://schema.org/name",
  "expected_object": "Alice Johnson",
  "min_confidence": "medium"
}`}
                />
              </EndpointCard>
              
              <EndpointCard
                endpoint="/ai/context/{iri}"
                method="GET"
                description="Get full context for an entity"
                isOpen={openEndpoints['context']}
                onToggle={() => toggleEndpoint('context')}
              >
                <p>Retrieve all facts about an entity, including related entities and confidence summaries.</p>
              </EndpointCard>
              
              <EndpointCard
                endpoint="/ai/materialize"
                method="POST"
                description="Run reasoning engine to derive new facts"
                isOpen={openEndpoints['materialize']}
                onToggle={() => toggleEndpoint('materialize')}
              >
                <p>Execute RDFS/OWL 2 RL forward-chaining inference, persisting derived facts with provenance.</p>
              </EndpointCard>
            </div>
          </div>
        )}

        {/* Query Section */}
        {activeSection === 'query' && (
          <div className="ai-section">
            <h2>Query Facts</h2>
            <p className="section-intro">
              Use <code>POST /ai/query</code> to retrieve facts with full provenance for grounding AI responses.
              Each fact includes source, confidence, timestamp, and a unique citation hash.
            </p>
            
            <QueryDemo currentRepo={currentRepo} />
            
            <h3>Integration Examples</h3>
            <CodeExample 
              title="Python Client"
              language="python"
              code={`import requests

response = requests.post("http://localhost:8000/ai/query", json={
    "subject": "http://example.org/customer/123",
    "min_confidence": "medium",
    "limit": 50
})

facts = response.json()["facts"]
for fact in facts:
    print(f"{fact['predicate']}: {fact['object']}")
    print(f"  Source: {fact['citation']['source']}")
    print(f"  Confidence: {fact['citation']['confidence']:.0%}")`}
            />
            
            <CodeExample 
              title="JavaScript/Fetch"
              language="javascript"
              code={`const response = await fetch('/ai/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    predicate: 'http://schema.org/name',
    min_confidence: 'high'
  })
});

const { facts, sources_used } = await response.json();
facts.forEach(fact => {
  console.log(\`\${fact.object} (from \${fact.citation.source})\`);
});`}
            />
          </div>
        )}

        {/* Verify Section */}
        {activeSection === 'verify' && (
          <div className="ai-section">
            <h2>Verify Claims</h2>
            <p className="section-intro">
              Use <code>POST /ai/verify</code> to check if a claim is supported by the knowledge base 
              before including it in an AI response. Detects conflicting information and provides recommendations.
            </p>
            
            <VerifyDemo currentRepo={currentRepo} />
            
            <h3>Integration Examples</h3>
            <CodeExample 
              title="LangChain Tool Definition"
              language="python"
              code={`from langchain.tools import tool
import requests

@tool
def verify_claim(subject: str, predicate: str, expected_value: str = None) -> str:
    """Verify if a claim is supported by the knowledge base."""
    response = requests.post("http://localhost:8000/ai/verify", json={
        "subject": subject,
        "predicate": predicate,
        "expected_object": expected_value,
        "min_confidence": "medium"
    })
    result = response.json()
    
    if result["claim_supported"]:
        return f"VERIFIED: {result['recommendation']}"
    else:
        return f"NOT VERIFIED: {result['recommendation']}"`}
            />
          </div>
        )}

        {/* Context Section */}
        {activeSection === 'context' && (
          <div className="ai-section">
            <h2>Entity Context</h2>
            <p className="section-intro">
              Use <code>GET /ai/context/{'{iri}'}</code> to retrieve everything known about an entity.
              Useful for building comprehensive context before answering questions.
            </p>
            
            <ContextDemo currentRepo={currentRepo} />
            
            <h3>Integration Examples</h3>
            <CodeExample 
              title="RAG Context Building"
              language="python"
              code={`def build_context_for_rag(entity_iri: str) -> str:
    """Build a context string for RAG from entity facts."""
    import requests
    from urllib.parse import quote
    
    response = requests.get(
        f"http://localhost:8000/ai/context/{quote(entity_iri, safe='')}",
        params={"min_confidence": "medium", "limit": 50}
    )
    data = response.json()
    
    context_lines = [f"Information about {data['entity']}:"]
    for fact in data["facts"]:
        pred = fact["predicate"].split("/")[-1].split("#")[-1]
        conf = fact["citation"]["confidence"]
        context_lines.append(f"- {pred}: {fact['object']} ({conf:.0%} confidence)")
    
    return "\\n".join(context_lines)`}
            />
          </div>
        )}

        {/* Integration Section */}
        {activeSection === 'integration' && (
          <div className="ai-section">
            <h2>Integration Guide</h2>
            <p className="section-intro">
              Integrate the AI Grounding API with popular frameworks and tools.
            </p>
            
            <h3>REST API Base</h3>
            <CodeExample 
              title="Base URL"
              language="text"
              code={`http://localhost:8000/ai/

Endpoints:
  POST /ai/query      - Structured fact retrieval
  POST /ai/verify     - Claim verification
  GET  /ai/context/*  - Entity context
  POST /ai/materialize - Run reasoning
  GET  /ai/inferences  - List inferred facts
  GET  /ai/health      - Health check`}
            />
            
            <h3>OpenAI Function Calling</h3>
            <CodeExample 
              title="Function Definition"
              language="json"
              code={`{
  "type": "function",
  "function": {
    "name": "query_knowledge_base",
    "description": "Query the knowledge base for verified facts with provenance",
    "parameters": {
      "type": "object",
      "properties": {
        "subject": { "type": "string", "description": "Entity IRI to query" },
        "predicate": { "type": "string", "description": "Relationship to filter by" },
        "min_confidence": {
          "type": "string",
          "enum": ["high", "medium", "low", "any"]
        }
      }
    }
  }
}`}
            />
            
            <h3>LlamaIndex Tool</h3>
            <CodeExample 
              title="Custom Tool"
              language="python"
              code={`from llama_index.core.tools import FunctionTool
import requests

def get_entity_context(entity_iri: str) -> str:
    """Get all known facts about an entity from the knowledge base."""
    from urllib.parse import quote
    
    response = requests.get(
        f"http://localhost:8000/ai/context/{quote(entity_iri, safe='')}",
        params={"min_confidence": "low"}
    )
    
    if response.status_code != 200:
        return f"Error: {response.text}"
    
    data = response.json()
    lines = []
    for fact in data["facts"][:20]:
        pred = fact["predicate"].split("/")[-1]
        lines.append(f"{pred}: {fact['object']} (from {fact['citation']['source']})")
    
    return "\\n".join(lines) if lines else "No facts found"

entity_tool = FunctionTool.from_defaults(fn=get_entity_context)`}
            />
            
            <h3>cURL Examples</h3>
            <CodeExample 
              title="Query Facts"
              language="bash"
              code={`curl -X POST http://localhost:8000/ai/query \\
  -H "Content-Type: application/json" \\
  -d '{"min_confidence": "medium", "limit": 10}'`}
            />
            
            <CodeExample 
              title="Verify Claim"
              language="bash"
              code={`curl -X POST http://localhost:8000/ai/verify \\
  -H "Content-Type: application/json" \\
  -d '{
    "subject": "http://example.org/entity",
    "predicate": "http://schema.org/name",
    "expected_object": "Test Entity"
  }'`}
            />
          </div>
        )}
      </main>
    </div>
  )
}
