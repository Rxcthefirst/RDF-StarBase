import { useState, useEffect } from 'react'
import { 
  DatabaseIcon, ZapIcon, NetworkIcon, LayersIcon,
  StarIcon, CheckCircleIcon, BoxIcon, GitBranchIcon, 
  UploadIcon, PlayIcon, PlusIcon, RefreshIcon
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
// Repository Card - Shows actual repo status
// ============================================================================
function RepositoryCard({ repo, isSelected, onSelect }) {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (repo.name) {
      setLoading(true)
      fetchJson(`/repositories/${repo.name}/stats`)
        .then(setStats)
        .catch(() => setStats(null))
        .finally(() => setLoading(false))
    }
  }, [repo.name])

  return (
    <div 
      className={`repo-card ${isSelected ? 'selected' : ''}`}
      onClick={() => onSelect(repo.name)}
    >
      <div className="repo-header">
        <DatabaseIcon size={20} />
        <span className="repo-name">{repo.name}</span>
        {isSelected && <span className="selected-badge">Active</span>}
      </div>
      <div className="repo-stats">
        {loading ? (
          <span className="loading-text">Loading...</span>
        ) : stats ? (
          <>
            <div className="repo-stat">
              <LayersIcon size={14} />
              <span>{stats.triple_count?.toLocaleString() || 0} triples</span>
            </div>
            <div className="repo-stat">
              <BoxIcon size={14} />
              <span>{stats.subject_count?.toLocaleString() || 0} entities</span>
            </div>
            <div className="repo-stat">
              <GitBranchIcon size={14} />
              <span>{stats.predicate_count?.toLocaleString() || 0} predicates</span>
            </div>
          </>
        ) : (
          <span className="empty-text">Empty repository</span>
        )}
      </div>
    </div>
  )
}

// ============================================================================
// Workflow Step Component
// ============================================================================
function WorkflowStep({ number, title, description, action, actionLabel, icon: Icon, disabled, completed }) {
  return (
    <div className={`workflow-step ${disabled ? 'disabled' : ''} ${completed ? 'completed' : ''}`}>
      <div className="step-number">
        {completed ? <CheckCircleIcon size={20} /> : number}
      </div>
      <div className="step-content">
        <div className="step-header">
          <Icon size={18} />
          <h4>{title}</h4>
        </div>
        <p>{description}</p>
      </div>
      {action && (
        <button className="step-action" onClick={action} disabled={disabled}>
          {actionLabel}
        </button>
      )}
    </div>
  )
}

// ============================================================================
// Demo Data - Sample RDF-Star knowledge graph
// ============================================================================
const DEMO_DATA = `@prefix : <http://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfstar: <http://rdf-starbase.dev/> .

# People
:alice a :Person ;
    rdfs:label "Alice Johnson" ;
    :email "alice@example.org" ;
    :age 32 ;
    :worksAt :TechCorp .

:bob a :Person ;
    rdfs:label "Bob Smith" ;
    :email "bob@example.org" ;
    :age 28 ;
    :worksAt :TechCorp ;
    :reportsTo :alice .

:charlie a :Person ;
    rdfs:label "Charlie Brown" ;
    :email "charlie@example.org" ;
    :age 35 ;
    :worksAt :StartupInc .

# Organizations
:TechCorp a :Company ;
    rdfs:label "TechCorp Industries" ;
    :founded "2010-01-15"^^xsd:date ;
    :headquarters :SanFrancisco ;
    :employees 500 .

:StartupInc a :Company ;
    rdfs:label "Startup Inc" ;
    :founded "2020-06-01"^^xsd:date ;
    :headquarters :NewYork ;
    :employees 25 .

# Locations
:SanFrancisco a :City ;
    rdfs:label "San Francisco" ;
    :country "USA" ;
    :population 870000 .

:NewYork a :City ;
    rdfs:label "New York" ;
    :country "USA" ;
    :population 8300000 .

# Projects
:ProjectAlpha a :Project ;
    rdfs:label "Project Alpha" ;
    :status "active" ;
    :lead :alice ;
    :team :bob .

:ProjectBeta a :Project ;
    rdfs:label "Project Beta" ;
    :status "planning" ;
    :lead :charlie .

# RDF-Star Annotations - Provenance and Trust
<< :alice :worksAt :TechCorp >> 
    rdfstar:source "HR Database" ;
    rdfstar:confidence 0.99 ;
    rdfstar:recordedAt "2024-01-15"^^xsd:date .

<< :bob :reportsTo :alice >> 
    rdfstar:source "Org Chart" ;
    rdfstar:confidence 0.95 ;
    rdfstar:verifiedBy :hr_system .

<< :TechCorp :employees 500 >>
    rdfstar:source "Annual Report 2024" ;
    rdfstar:confidence 0.90 ;
    rdfstar:asOf "2024-12-31"^^xsd:date .

<< :charlie :worksAt :StartupInc >>
    rdfstar:source "LinkedIn" ;
    rdfstar:confidence 0.70 .

<< :charlie :worksAt :TechCorp >>
    rdfstar:source "Old Database" ;
    rdfstar:confidence 0.30 ;
    rdfstar:note "Outdated record" .

# Skills and expertise
:alice :hasSkill :Python, :Leadership, :DataScience .
:bob :hasSkill :JavaScript, :React, :Python .
:charlie :hasSkill :Rust, :SystemsDesign .

:Python a :Skill ; rdfs:label "Python" .
:JavaScript a :Skill ; rdfs:label "JavaScript" .
:React a :Skill ; rdfs:label "React" .
:Leadership a :Skill ; rdfs:label "Leadership" .
:DataScience a :Skill ; rdfs:label "Data Science" .
:Rust a :Skill ; rdfs:label "Rust" .
:SystemsDesign a :Skill ; rdfs:label "Systems Design" .
`

// ============================================================================
// Main Dashboard Component  
// ============================================================================
export default function Dashboard({ 
  repositories = [], 
  currentRepo,
  onNavigateToWorkbench,
  onRunQuery,
  onCreateRepo,
  onSelectRepo,
  onRefreshRepos,
  onOpenImport,
  theme 
}) {
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState(null)
  const [repoStats, setRepoStats] = useState(null)
  const [statsKey, setStatsKey] = useState(0) // Force refresh trigger

  // Load stats for current repo
  useEffect(() => {
    if (currentRepo) {
      fetchJson(`/repositories/${currentRepo}/stats`)
        .then(setRepoStats)
        .catch(() => setRepoStats(null))
    } else {
      setRepoStats(null)
    }
  }, [currentRepo, statsKey])

  // Refresh stats manually
  const refreshStats = () => setStatsKey(k => k + 1)

  // Create demo repository with sample data
  const loadDemoData = async () => {
    setLoading(true)
    setMessage(null)
    try {
      // Create demo repo if it doesn't exist
      const repoName = 'DemoKnowledgeGraph'
      const exists = repositories.some(r => r.name === repoName)
      
      if (!exists) {
        await fetchJson('/repositories', {
          method: 'POST',
          body: JSON.stringify({ name: repoName, description: 'Demo knowledge graph with RDF-Star annotations' })
        })
      }

      // Load the demo data using the import endpoint
      await fetchJson(`/repositories/${repoName}/import`, {
        method: 'POST',
        body: JSON.stringify({ 
          data: DEMO_DATA, 
          format: 'turtle'
        })
      })

      // Refresh repos list first
      if (onRefreshRepos) await onRefreshRepos()
      
      // Select the repo
      onSelectRepo(repoName)
      
      // Force stats refresh after a brief delay to ensure data is committed
      setTimeout(() => {
        setStatsKey(k => k + 1)
      }, 200)
      
      setMessage({ type: 'success', text: `Loaded demo data with RDF-Star annotations. Click a sample query to explore!` })
    } catch (err) {
      console.error('Demo data error:', err)
      setMessage({ type: 'error', text: `Failed to load demo data: ${err.message}` })
    } finally {
      setLoading(false)
    }
  }

  // Sample queries for the current repo
  const sampleQueries = [
    {
      label: 'All Triples',
      query: 'SELECT * WHERE { ?s ?p ?o } LIMIT 100',
      description: 'View all data'
    },
    {
      label: 'People',
      query: `PREFIX : <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?person ?name ?email WHERE {
  ?person a :Person ;
          rdfs:label ?name .
  OPTIONAL { ?person :email ?email }
}`,
      description: 'Find all people'
    },
    {
      label: 'RDF-Star Annotations',
      query: `PREFIX rdfstar: <http://rdf-starbase.dev/>
SELECT ?s ?p ?o ?source ?confidence WHERE {
  << ?s ?p ?o >> rdfstar:source ?source .
  OPTIONAL { << ?s ?p ?o >> rdfstar:confidence ?confidence }
}
ORDER BY DESC(?confidence)`,
      description: 'View statement metadata'
    },
    {
      label: 'Competing Claims',
      query: `PREFIX : <http://example.org/>
PREFIX rdfstar: <http://rdf-starbase.dev/>
SELECT ?person ?company ?source ?confidence WHERE {
  << ?person :worksAt ?company >> rdfstar:source ?source ;
                                   rdfstar:confidence ?confidence .
}
ORDER BY ?person DESC(?confidence)`,
      description: 'Find conflicting data'
    }
  ]

  const hasRepo = repositories.length > 0
  const hasData = repoStats?.triple_count > 0

  return (
    <div className="dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <div className="header-title">
          <h1>RDF-StarBase</h1>
          <span className="version-badge">v1.1.0</span>
        </div>
        <p>High-performance knowledge graph with native RDF-Star support</p>
      </div>

      {/* Status Message */}
      {message && (
        <div className={`status-message ${message.type}`}>
          {message.text}
          <button onClick={() => setMessage(null)}>×</button>
        </div>
      )}

      {/* Main Content Grid */}
      <div className="dashboard-grid">
        
        {/* Left Column - Repositories */}
        <section className="dashboard-section repositories-section">
          <div className="section-header">
            <h2><DatabaseIcon size={18} /> Repositories</h2>
            <button className="icon-btn" onClick={onCreateRepo} title="Create Repository">
              <PlusIcon size={16} />
            </button>
          </div>
          
          {repositories.length === 0 ? (
            <div className="empty-state">
              <DatabaseIcon size={32} />
              <p>No repositories yet</p>
              <button className="btn primary" onClick={onCreateRepo}>
                Create First Repository
              </button>
            </div>
          ) : (
            <div className="repo-list">
              {repositories.map(repo => (
                <RepositoryCard
                  key={repo.name}
                  repo={repo}
                  isSelected={currentRepo === repo.name}
                  onSelect={onSelectRepo}
                />
              ))}
            </div>
          )}

          {/* Quick Start */}
          <div className="quick-start">
            <h3>Quick Start</h3>
            <button 
              className="btn secondary full-width"
              onClick={loadDemoData}
              disabled={loading}
            >
              <StarIcon size={16} />
              {loading ? 'Loading...' : 'Load Demo Data'}
            </button>
            <p className="hint">Creates a sample knowledge graph with RDF-Star annotations</p>
          </div>
        </section>

        {/* Center Column - Workflow */}
        <section className="dashboard-section workflow-section">
          <div className="section-header">
            <h2><ZapIcon size={18} /> Getting Started</h2>
          </div>

          <div className="workflow-steps">
            <WorkflowStep
              number={1}
              icon={DatabaseIcon}
              title="Create or Select Repository"
              description="A repository stores your knowledge graph data"
              completed={hasRepo}
              action={!hasRepo ? onCreateRepo : undefined}
              actionLabel="Create"
            />
            
            <WorkflowStep
              number={2}
              icon={UploadIcon}
              title="Import Data"
              description="Load Turtle, JSON-LD, RDF/XML, or TriG files"
              disabled={!hasRepo}
              completed={hasData}
              action={hasRepo ? onOpenImport : undefined}
              actionLabel={hasData ? "Add More" : "Import"}
            />
            
            <WorkflowStep
              number={3}
              icon={NetworkIcon}
              title="Query & Explore"
              description="Query with SPARQL and visualize your knowledge graph"
              disabled={!hasData}
              action={hasData ? onNavigateToWorkbench : undefined}
              actionLabel="Open Workbench"
            />
          </div>
        </section>

        {/* Right Column - Current Repository Stats */}
        <section className="dashboard-section stats-section">
          <div className="section-header">
            <h2><LayersIcon size={18} /> {currentRepo || 'No Repository Selected'}</h2>
            {currentRepo && (
              <button className="icon-btn" onClick={() => {
                fetchJson(`/repositories/${currentRepo}/stats`).then(setRepoStats)
              }} title="Refresh">
                <RefreshIcon size={16} />
              </button>
            )}
          </div>

          {currentRepo && repoStats ? (
            <div className="stats-display">
              <div className="stat-item large">
                <span className="stat-value">{repoStats.triple_count?.toLocaleString() || 0}</span>
                <span className="stat-label">Triples</span>
              </div>
              <div className="stat-row">
                <div className="stat-item">
                  <span className="stat-value">{repoStats.subject_count?.toLocaleString() || 0}</span>
                  <span className="stat-label">Entities</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">{repoStats.predicate_count?.toLocaleString() || 0}</span>
                  <span className="stat-label">Predicates</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">{repoStats.object_count?.toLocaleString() || 0}</span>
                  <span className="stat-label">Objects</span>
                </div>
              </div>
              {repoStats.graph_count > 0 && (
                <div className="stat-item">
                  <span className="stat-value">{repoStats.graph_count}</span>
                  <span className="stat-label">Named Graphs</span>
                </div>
              )}
            </div>
          ) : currentRepo ? (
            <div className="empty-state small">
              <p>No data yet</p>
              <button className="btn secondary" onClick={loadDemoData} disabled={loading}>
                Load Demo Data
              </button>
            </div>
          ) : (
            <div className="empty-state small">
              <p>Select a repository to view stats</p>
            </div>
          )}

          {/* Sample Queries - only show when there's data */}
          {hasData && (
            <div className="sample-queries">
              <h3>Try These Queries</h3>
              {sampleQueries.map((q, i) => (
                <button 
                  key={i}
                  className="query-btn"
                  onClick={() => onRunQuery(q.query)}
                  title={q.description}
                >
                  <PlayIcon size={14} />
                  {q.label}
                </button>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
