"""
RDF-StarBase Web API

FastAPI-based REST API for querying and managing the knowledge graph.
Provides endpoints for:
- SPARQL queries
- Triple management
- Provenance inspection
- Competing claims analysis
- Source registry
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union
from uuid import UUID
import json
import os

from fastapi import FastAPI, HTTPException, Query, Depends, Header, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import polars as pl

from rdf_starbase import (
    TripleStore,
    ProvenanceContext,
    AssertionRegistry,
    SourceType,
    SourceStatus,
    execute_sparql,
    parse_query,
)
from rdf_starbase.storage.auth import APIKeyManager, Role, Operation, AuthContext
from rdf_starbase.ai_grounding import create_ai_router
from rdf_starbase.repository_api import create_repository_router


# Pydantic models for API
class ProvenanceInput(BaseModel):
    """Provenance context for adding triples."""
    source: str = Field(..., description="Source system or person")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    process: Optional[str] = None
    
    def to_context(self) -> ProvenanceContext:
        return ProvenanceContext(
            source=self.source,
            confidence=self.confidence,
            process=self.process,
        )


class TripleInput(BaseModel):
    """Input for adding a triple."""
    subject: str
    predicate: str
    object: Union[str, int, float, bool]
    provenance: ProvenanceInput
    graph: Optional[str] = None


class BatchTripleInput(BaseModel):
    """Input for batch adding triples."""
    triples: list[dict] = Field(..., description="List of triple dicts with subject, predicate, object, source, confidence, process")


class SPARQLQuery(BaseModel):
    """SPARQL query request."""
    query: str = Field(..., description="SPARQL-Star query string")


class SourceInput(BaseModel):
    """Input for registering a source."""
    name: str
    source_type: str = Field(..., description="One of: dataset, api, mapping, process, manual")
    uri: Optional[str] = None
    description: Optional[str] = None
    owner: Optional[str] = None
    sync_frequency: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


def dataframe_to_records(df: pl.DataFrame) -> list[dict[str, Any]]:
    """Convert Polars DataFrame to list of dicts for JSON serialization."""
    records = []
    for row in df.iter_rows(named=True):
        record = {}
        for k, v in row.items():
            if isinstance(v, datetime):
                record[k] = v.isoformat()
            elif v is None:
                record[k] = None
            else:
                record[k] = v
        records.append(record)
    return records


# =============================================================================
# Security Router for API Key Management
# =============================================================================

def create_security_router(key_manager: APIKeyManager) -> APIRouter:
    """Create router for security/API key management endpoints."""
    router = APIRouter(prefix="/security", tags=["security"])
    
    @router.get("/status")
    async def security_status():
        """Get security configuration status."""
        keys = key_manager.list_keys()
        return {
            "enabled": len(keys) > 0,
            "key_count": len(keys),
            "message": "Security is enabled. Use API keys for authenticated access." if keys else "Security not configured. All endpoints are open."
        }
    
    @router.post("/keys")
    async def create_api_key(
        name: str,
        role: str = "reader",
        expires_days: Optional[int] = None,
        rate_limit: Optional[int] = None,
    ):
        """Create a new API key.
        
        **Important**: The returned key is only shown once. Store it securely!
        """
        from datetime import timedelta
        
        try:
            role_enum = Role(role.lower())
        except ValueError:
            raise HTTPException(400, f"Invalid role: {role}. Use: reader, writer, admin")
        
        expires_in = timedelta(days=expires_days) if expires_days else None
        
        raw_key, api_key = key_manager.generate_key(
            name=name,
            role=role_enum,
            expires_in=expires_in,
            rate_limit_queries=rate_limit,
        )
        
        return {
            "key": raw_key,  # Only returned once!
            "key_id": api_key.key_id,
            "name": api_key.name,
            "role": api_key.role.value,
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
            "warning": "Store this key securely! It will not be shown again."
        }
    
    @router.get("/keys")
    async def list_api_keys():
        """List all API keys (without revealing the actual keys)."""
        keys = key_manager.list_keys()
        return [
            {
                "key_id": k.key_id,
                "name": k.name,
                "role": k.role.value,
                "created_at": k.created_at.isoformat(),
                "expires_at": k.expires_at.isoformat() if k.expires_at else None,
                "last_used": k.last_used.isoformat() if k.last_used else None,
                "is_expired": k.is_expired,
            }
            for k in keys
        ]
    
    @router.delete("/keys/{key_id}")
    async def revoke_api_key(key_id: str):
        """Revoke an API key."""
        if key_manager.revoke_key(key_id):
            return {"message": f"Key {key_id} revoked"}
        raise HTTPException(404, f"Key not found: {key_id}")
    
    @router.get("/roles")
    async def list_roles():
        """List available roles and their permissions."""
        return [
            {
                "role": "reader",
                "description": "Read-only access. Can query data but not modify.",
                "permissions": ["query", "describe", "export"]
            },
            {
                "role": "writer", 
                "description": "Read and write access. Can query and modify data.",
                "permissions": ["query", "describe", "export", "insert", "delete", "update", "load"]
            },
            {
                "role": "admin",
                "description": "Full access including security and configuration.",
                "permissions": ["all operations"]
            }
        ]
    
    return router


# =============================================================================
# SQL Router for DuckDB Interface
# =============================================================================

class SQLQuery(BaseModel):
    """SQL query request."""
    sql: str = Field(..., description="SQL query to execute")
    limit: Optional[int] = Field(1000, ge=1, le=10000, description="Max rows to return")


def create_sql_router(store: TripleStore) -> APIRouter:
    """Create router for DuckDB SQL interface with query helpers."""
    router = APIRouter(prefix="/sql", tags=["SQL"])
    
    # Lazy initialize DuckDB interface
    _interface = None
    
    def get_interface():
        nonlocal _interface
        if _interface is None:
            from rdf_starbase.storage.duckdb import DuckDBInterface, DUCKDB_AVAILABLE
            if not DUCKDB_AVAILABLE:
                raise HTTPException(503, "DuckDB not installed. Install with: pip install duckdb")
            _interface = DuckDBInterface(store, read_only=True)
        return _interface
    
    @router.post("/execute")
    async def execute_sql(query: SQLQuery):
        """
        Execute a SQL query against the triple store.
        
        Available tables:
        - **triples**: Main triple data (subject, predicate, object, graph, source, confidence, timestamp)
        - **facts**: Raw integer-encoded facts (for advanced analysis)
        - **terms**: Term dictionary (term_id, kind, lex)
        - **provenance**: View of triples with source/confidence info
        - **rdf_types**: View of rdf:type statements
        - **named_graphs**: List of named graphs
        """
        try:
            interface = get_interface()
            result = interface.execute(query.sql, limit=query.limit)
            return result.to_dict()
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(500, f"SQL Error: {str(e)}")
    
    @router.get("/tables")
    async def list_tables():
        """List all available tables and views with their row counts."""
        try:
            interface = get_interface()
            tables = interface.list_tables()
            return [
                {
                    "name": t.name,
                    "columns": t.columns,
                    "row_count": t.row_count,
                }
                for t in tables
            ]
        except Exception as e:
            raise HTTPException(500, str(e))
    
    @router.get("/schema/{table_name}")
    async def get_table_schema(table_name: str):
        """Get the schema (column names and types) for a table."""
        try:
            interface = get_interface()
            schema = interface.get_schema(table_name)
            return {
                "table": table_name,
                "columns": schema,
            }
        except Exception as e:
            raise HTTPException(404, f"Table not found or error: {str(e)}")
    
    @router.get("/sample/{table_name}")
    async def sample_table(table_name: str, n: int = Query(10, ge=1, le=100)):
        """Get a sample of rows from a table."""
        try:
            interface = get_interface()
            result = interface.sample(table_name, n)
            return result.to_dict()
        except Exception as e:
            raise HTTPException(500, str(e))
    
    @router.get("/helpers")
    async def get_query_helpers():
        """
        Get a list of helpful SQL query templates for common operations.
        These can be used directly or customized.
        """
        return {
            "description": "SQL query templates for accessing claims in the triplestore",
            "queries": [
                {
                    "name": "All Triples",
                    "description": "Get all triples with provenance information",
                    "sql": "SELECT subject, predicate, object, source, confidence, timestamp FROM triples LIMIT 100",
                    "category": "basic"
                },
                {
                    "name": "Count by Predicate",
                    "description": "Count how many times each predicate is used",
                    "sql": "SELECT predicate, COUNT(*) as count FROM triples GROUP BY predicate ORDER BY count DESC",
                    "category": "analytics"
                },
                {
                    "name": "Entity Types",
                    "description": "List all entity types in the knowledge graph",
                    "sql": "SELECT type, COUNT(*) as count FROM rdf_types GROUP BY type ORDER BY count DESC",
                    "category": "schema"
                },
                {
                    "name": "Claims by Source",
                    "description": "Count claims grouped by source system",
                    "sql": "SELECT source, COUNT(*) as claim_count, AVG(confidence) as avg_confidence FROM triples WHERE source IS NOT NULL GROUP BY source ORDER BY claim_count DESC",
                    "category": "provenance"
                },
                {
                    "name": "Low Confidence Claims",
                    "description": "Find claims with confidence below 0.8",
                    "sql": "SELECT subject, predicate, object, source, confidence FROM triples WHERE confidence < 0.8 ORDER BY confidence ASC LIMIT 50",
                    "category": "quality"
                },
                {
                    "name": "Recent Claims",
                    "description": "Get the most recently added claims",
                    "sql": "SELECT subject, predicate, object, source, timestamp FROM triples WHERE timestamp IS NOT NULL ORDER BY timestamp DESC LIMIT 50",
                    "category": "temporal"
                },
                {
                    "name": "Find Entity Properties",
                    "description": "Get all properties of a specific entity (replace URI)",
                    "sql": "SELECT predicate, object, source, confidence FROM triples WHERE subject = 'http://example.org/entity' LIMIT 100",
                    "category": "entity"
                },
                {
                    "name": "Competing Claims Detection",
                    "description": "Find subject-predicate pairs with multiple different values (potential conflicts)",
                    "sql": """SELECT subject, predicate, COUNT(DISTINCT object) as value_count, 
       ARRAY_AGG(DISTINCT object) as values
FROM triples 
GROUP BY subject, predicate 
HAVING COUNT(DISTINCT object) > 1
ORDER BY value_count DESC
LIMIT 20""",
                    "category": "quality"
                },
                {
                    "name": "Source Coverage",
                    "description": "Analyze which predicates each source provides",
                    "sql": """SELECT source, 
       COUNT(DISTINCT predicate) as predicate_types,
       COUNT(DISTINCT subject) as entities,
       COUNT(*) as total_claims
FROM triples 
WHERE source IS NOT NULL
GROUP BY source""",
                    "category": "provenance"
                },
                {
                    "name": "Named Graphs Summary",
                    "description": "Count triples per named graph",
                    "sql": "SELECT graph, COUNT(*) as triple_count FROM triples WHERE graph IS NOT NULL GROUP BY graph ORDER BY triple_count DESC",
                    "category": "organization"
                },
                {
                    "name": "Term Dictionary Stats",
                    "description": "Analyze the term dictionary by kind",
                    "sql": "SELECT kind_name, COUNT(*) as count FROM terms GROUP BY kind_name ORDER BY count DESC",
                    "category": "internals"
                },
                {
                    "name": "Full-Text Search",
                    "description": "Search for entities containing a keyword (replace 'keyword')",
                    "sql": "SELECT DISTINCT subject FROM triples WHERE subject LIKE '%keyword%' OR object LIKE '%keyword%' LIMIT 50",
                    "category": "search"
                },
            ]
        }
    
    @router.post("/refresh")
    async def refresh_tables():
        """Refresh the DuckDB table registrations after data changes."""
        try:
            interface = get_interface()
            interface.refresh_tables()
            return {"message": "Tables refreshed successfully"}
        except Exception as e:
            raise HTTPException(500, str(e))
    
    return router


def create_app(store: Optional[TripleStore] = None, registry: Optional[AssertionRegistry] = None) -> FastAPI:
    """
    Create the FastAPI application.
    
    Args:
        store: Optional TripleStore instance (creates new if not provided)
        registry: Optional AssertionRegistry instance
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="RDF-StarBase API",
        description="A blazingly fast RDF★ database with native provenance tracking",
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # State
    app.state.store = store or TripleStore()
    app.state.registry = registry or AssertionRegistry()
    
    # Security - API Key Manager
    from pathlib import Path
    data_dir = Path(os.getenv("RDFSTARBASE_DATA_DIR", "./data"))
    key_storage = data_dir / "security" / "api_keys.json"
    key_storage.parent.mkdir(parents=True, exist_ok=True)
    app.state.key_manager = APIKeyManager(storage_path=key_storage)
    
    # Add Repository Management router
    repo_router, repo_manager = create_repository_router()
    app.include_router(repo_router)
    app.state.repo_manager = repo_manager
    
    # Add AI Grounding API router
    ai_router = create_ai_router(app.state.store)
    app.include_router(ai_router)
    
    # Add Security router
    security_router = create_security_router(app.state.key_manager)
    app.include_router(security_router)
    
    # Add SQL/DuckDB router
    sql_router = create_sql_router(app.state.store)
    app.include_router(sql_router)
    
    # ==========================================================================
    # Health & Info
    # ==========================================================================
    
    @app.get("/", tags=["Info"])
    async def root():
        """API root with basic info."""
        return {
            "name": "RDF-StarBase",
            "version": "0.2.0",
            "description": "A blazingly fast RDF★ database with native provenance tracking",
            "docs": "/docs",
        }
    
    @app.get("/health", tags=["Info"])
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    @app.get("/stats", tags=["Info"])
    async def stats():
        """Get store and registry statistics."""
        return {
            "store": app.state.store.stats(),
            "registry": app.state.registry.get_stats(),
        }
    
    # ==========================================================================
    # Triples
    # ==========================================================================
    
    @app.post("/triples", tags=["Triples"])
    async def add_triple(triple: TripleInput):
        """Add a triple with provenance to the store."""
        try:
            assertion_id = app.state.store.add_triple(
                subject=triple.subject,
                predicate=triple.predicate,
                obj=triple.object,
                provenance=triple.provenance.to_context(),
                graph=triple.graph,
            )
            return {"assertion_id": str(assertion_id)}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/triples/batch", tags=["Triples"])
    async def add_triples_batch(batch: BatchTripleInput):
        """
        Add multiple triples in a single batch operation.
        
        This is MUCH faster than calling POST /triples repeatedly.
        Each triple dict should have: subject, predicate, object, source, confidence (optional), process (optional).
        """
        try:
            count = app.state.store.add_triples_batch(batch.triples)
            return {
                "success": True,
                "count": count,
                "message": f"Added {count} triples",
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/triples", tags=["Triples"])
    async def get_triples(
        subject: Optional[str] = Query(None, description="Filter by subject"),
        predicate: Optional[str] = Query(None, description="Filter by predicate"),
        object: Optional[str] = Query(None, description="Filter by object"),
        source: Optional[str] = Query(None, description="Filter by source"),
        min_confidence: Optional[float] = Query(None, ge=0, le=1, description="Minimum confidence"),
        limit: int = Query(100, ge=1, le=10000, description="Maximum results"),
    ):
        """Query triples with optional filters."""
        df = app.state.store.get_triples(
            subject=subject,
            predicate=predicate,
            obj=object,
            source=source,
            min_confidence=min_confidence,
        )
        
        df = df.head(limit)
        
        return {
            "count": len(df),
            "triples": dataframe_to_records(df),
        }
    
    @app.get("/triples/{subject_encoded:path}/claims", tags=["Triples"])
    async def get_competing_claims(
        subject_encoded: str,
        predicate: str = Query(..., description="Predicate to check for conflicts"),
    ):
        """Get competing claims for a subject-predicate pair."""
        # URL decode the subject
        import urllib.parse
        subject = urllib.parse.unquote(subject_encoded)
        
        df = app.state.store.get_competing_claims(subject, predicate)
        
        if len(df) == 0:
            return {"count": 0, "has_conflicts": False, "claims": []}
        
        unique_values = df["object"].n_unique()
        
        return {
            "count": len(df),
            "has_conflicts": unique_values > 1,
            "unique_values": unique_values,
            "claims": dataframe_to_records(df),
        }
    
    @app.get("/triples/{subject_encoded:path}/timeline", tags=["Triples"])
    async def get_provenance_timeline(
        subject_encoded: str,
        predicate: str = Query(..., description="Predicate for timeline"),
    ):
        """Get provenance timeline for a subject-predicate pair."""
        import urllib.parse
        subject = urllib.parse.unquote(subject_encoded)
        
        df = app.state.store.get_provenance_timeline(subject, predicate)
        
        return {
            "count": len(df),
            "timeline": dataframe_to_records(df),
        }
    
    # ==========================================================================
    # SPARQL
    # ==========================================================================
    
    @app.post("/sparql", tags=["SPARQL"])
    async def execute_sparql_query(request: SPARQLQuery):
        """Execute a SPARQL-Star query (SELECT, ASK, INSERT DATA, DELETE DATA)."""
        try:
            result = execute_sparql(app.state.store, request.query)
            
            if isinstance(result, bool):
                # ASK query
                return {"type": "ask", "result": result}
            elif isinstance(result, dict):
                # UPDATE operation (INSERT DATA, DELETE DATA)
                return {
                    "type": "update",
                    "operation": result.get("operation", "unknown"),
                    "count": result.get("count", 0),
                    "success": True,
                }
            elif isinstance(result, pl.DataFrame):
                # SELECT query
                return {
                    "type": "select",
                    "count": len(result),
                    "columns": result.columns,
                    "results": dataframe_to_records(result),
                }
            else:
                return {"type": "unknown", "result": str(result)}
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Query error: {str(e)}")
    
    @app.post("/sparql/update", tags=["SPARQL"])
    async def execute_sparql_update(request: SPARQLQuery):
        """Execute a SPARQL UPDATE operation (INSERT DATA, DELETE DATA).
        
        Supports provenance headers:
        - X-Provenance-Source: Source identifier (default: SPARQL_INSERT)
        - X-Provenance-Confidence: Confidence score 0.0-1.0 (default: 1.0)
        - X-Provenance-Process: Process identifier (optional)
        """
        try:
            # For now, use default provenance
            # TODO: Extract from headers
            from rdf_starbase.models import ProvenanceContext
            provenance = ProvenanceContext(source="SPARQL_UPDATE", confidence=1.0)
            
            result = execute_sparql(app.state.store, request.query, provenance)
            
            if isinstance(result, dict):
                return {
                    "type": "update",
                    "operation": result.get("operation", "unknown"),
                    "count": result.get("count", 0),
                    "success": result.get("status") != "not_implemented",
                    "message": f"Processed {result.get('count', 0)} triples",
                }
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="Expected an UPDATE operation (INSERT DATA, DELETE DATA)"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Update error: {str(e)}")
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Query error: {str(e)}")
    
    @app.post("/sparql/parse", tags=["SPARQL"])
    async def parse_sparql(request: SPARQLQuery):
        """Parse a SPARQL query and return the AST structure."""
        try:
            ast = parse_query(request.query)
            
            return {
                "type": type(ast).__name__,
                "prefixes": ast.prefixes,
                "pattern_count": len(ast.where.patterns) if ast.where else 0,
                "filter_count": len(ast.where.filters) if ast.where else 0,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Parse error: {str(e)}")
    
    # ==========================================================================
    # Registry
    # ==========================================================================
    
    @app.post("/sources", tags=["Registry"])
    async def register_source(source: SourceInput):
        """Register a new data source."""
        try:
            src_type = SourceType(source.source_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source_type. Must be one of: {[t.value for t in SourceType]}"
            )
        
        registered = app.state.registry.register_source(
            name=source.name,
            source_type=src_type,
            uri=source.uri,
            description=source.description,
            owner=source.owner,
            sync_frequency=source.sync_frequency,
            tags=source.tags,
        )
        
        return {
            "id": str(registered.id),
            "name": registered.name,
            "source_type": registered.source_type.value,
        }
    
    @app.get("/sources", tags=["Registry"])
    async def get_sources(
        source_type: Optional[str] = Query(None, description="Filter by type"),
        owner: Optional[str] = Query(None, description="Filter by owner"),
        tag: Optional[str] = Query(None, description="Filter by tag"),
    ):
        """List registered sources with optional filters."""
        kwargs = {}
        
        if source_type:
            try:
                kwargs["source_type"] = SourceType(source_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid source_type: {source_type}")
        
        if owner:
            kwargs["owner"] = owner
        if tag:
            kwargs["tag"] = tag
        
        sources = app.state.registry.get_sources(**kwargs)
        
        return {
            "count": len(sources),
            "sources": [
                {
                    "id": str(s.id),
                    "name": s.name,
                    "source_type": s.source_type.value,
                    "uri": s.uri,
                    "status": s.status.value,
                    "owner": s.owner,
                    "last_sync": s.last_sync.isoformat() if s.last_sync else None,
                    "tags": s.tags,
                }
                for s in sources
            ],
        }
    
    @app.get("/sources/{source_id}", tags=["Registry"])
    async def get_source(source_id: str):
        """Get details of a specific source."""
        try:
            uid = UUID(source_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid UUID format")
        
        source = app.state.registry.get_source(uid)
        if source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        
        return {
            "id": str(source.id),
            "name": source.name,
            "source_type": source.source_type.value,
            "uri": source.uri,
            "description": source.description,
            "status": source.status.value,
            "created_at": source.created_at.isoformat(),
            "last_sync": source.last_sync.isoformat() if source.last_sync else None,
            "owner": source.owner,
            "sync_frequency": source.sync_frequency,
            "tags": source.tags,
        }
    
    @app.get("/sources/{source_id}/syncs", tags=["Registry"])
    async def get_sync_history(
        source_id: str,
        limit: int = Query(20, ge=1, le=100),
    ):
        """Get sync history for a source."""
        try:
            uid = UUID(source_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid UUID format")
        
        history = app.state.registry.get_sync_history(uid, limit=limit)
        
        return {
            "count": len(history),
            "syncs": dataframe_to_records(history),
        }
    
    # ==========================================================================
    # Graph Visualization Data
    # ==========================================================================
    
    @app.get("/graph/nodes", tags=["Visualization"])
    async def get_graph_nodes(
        limit: int = Query(100, ge=1, le=1000),
    ):
        """Get unique nodes (subjects and objects) for graph visualization."""
        df = app.state.store._df
        
        subjects = df["subject"].unique().to_list()[:limit]
        objects = df.filter(
            pl.col("object_type") == "uri"
        )["object"].unique().to_list()[:limit]
        
        all_nodes = list(set(subjects + objects))[:limit]
        
        return {
            "count": len(all_nodes),
            "nodes": [{"id": n, "label": n.split("/")[-1]} for n in all_nodes],
        }
    
    @app.get("/graph/edges", tags=["Visualization"])
    async def get_graph_edges(
        limit: int = Query(500, ge=1, le=5000),
    ):
        """Get edges (triples) for graph visualization."""
        df = app.state.store._df.head(limit)
        
        # Only include edges where target is also a URI node (not literals)
        # This makes the graph visualizable
        df_uri_objects = df.filter(pl.col("object_type") == "uri")
        
        edges = []
        for row in df_uri_objects.iter_rows(named=True):
            edges.append({
                "source": row["subject"],
                "target": row["object"],
                "predicate": row["predicate"],
                "label": row["predicate"].split("/")[-1],
                "confidence": row["confidence"],
                "provenance_source": row["source"],
            })
        
        return {
            "count": len(edges),
            "edges": edges,
        }
    
    @app.get("/graph/subgraph/{node_encoded:path}", tags=["Visualization"])
    async def get_subgraph(
        node_encoded: str,
        depth: int = Query(1, ge=1, le=3, description="Traversal depth"),
    ):
        """Get subgraph around a specific node."""
        import urllib.parse
        node = urllib.parse.unquote(node_encoded)
        
        # Get triples where node is subject or object
        df = app.state.store._df
        
        outgoing = df.filter(pl.col("subject") == node)
        incoming = df.filter(pl.col("object") == node)
        
        related = pl.concat([outgoing, incoming]).unique()
        
        nodes = set()
        edges = []
        
        for row in related.iter_rows(named=True):
            nodes.add(row["subject"])
            # Only add object as a node if it's a URI, not a literal
            if row["object_type"] == "uri":
                nodes.add(row["object"])
                edges.append({
                    "source": row["subject"],
                    "target": row["object"],
                    "predicate": row["predicate"],
                    "confidence": row["confidence"],
                })
        
        return {
            "center": node,
            "nodes": [{"id": n, "label": n.split("/")[-1] if "/" in n else n} for n in nodes],
            "edges": edges,
        }
    
    return app


def get_static_dir() -> Optional[Path]:
    """Find the frontend static files directory."""
    # Check various possible locations
    candidates = [
        Path(__file__).parent.parent.parent / "frontend" / "dist",  # Development
        Path("/app/frontend/dist"),  # Docker
        Path.cwd() / "frontend" / "dist",  # Current directory
    ]
    for candidate in candidates:
        if candidate.exists() and (candidate / "index.html").exists():
            return candidate
    return None


def create_production_app() -> FastAPI:
    """Create app with static file serving for production."""
    base_app = create_app()
    
    static_dir = get_static_dir()
    if static_dir:
        # Mount static assets at /app/assets (matching Vite's base: '/app/')
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            base_app.mount("/app/assets", StaticFiles(directory=str(assets_dir)), name="assets")
        
        # Serve index.html for /app and /app/*
        @base_app.get("/app", include_in_schema=False)
        async def serve_spa_root():
            return FileResponse(static_dir / "index.html")
        
        @base_app.get("/app/{path:path}", include_in_schema=False)
        async def serve_spa(path: str = ""):
            # Check if it's a static file request that wasn't caught by mount
            file_path = static_dir / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
            # Otherwise serve index.html for SPA routing
            return FileResponse(static_dir / "index.html")
        
        # Serve favicon if present
        favicon_path = static_dir / "favicon.ico"
        if favicon_path.exists():
            @base_app.get("/favicon.ico", include_in_schema=False)
            async def favicon():
                return FileResponse(favicon_path)
    
    return base_app


# Default app instance for running directly
# In production (Docker), use create_production_app()
app = create_app()

# Check if we should serve static files (production mode)
if os.environ.get("RDFSTARBASE_SERVE_STATIC", "").lower() in ("1", "true", "yes"):
    app = create_production_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rdf_starbase.web:app", host="0.0.0.0", port=8000, reload=True)
