"""
Repository Management API Router.

Provides REST endpoints for managing multiple repositories:
- Create/delete repositories
- List repositories
- Get repository info
- Scoped SPARQL queries per repository
"""

from pathlib import Path
from typing import Optional, Union
import os
import time
import uuid
import json
import hashlib
import tempfile
import asyncio

from fastapi import APIRouter, HTTPException, Query, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import polars as pl

from rdf_starbase.repositories import RepositoryManager, RepositoryInfo
from rdf_starbase import execute_sparql


# =============================================================================
# Pydantic Models
# =============================================================================

class CreateRepositoryRequest(BaseModel):
    """Request to create a new repository."""
    name: str = Field(..., description="Unique repository name (alphanumeric, hyphens, underscores)")
    description: str = Field(default="", description="Human-readable description")
    tags: list[str] = Field(default_factory=list, description="Optional tags")
    # Reasoning configuration (like GraphDB)
    reasoning_level: str = Field(
        default="none", 
        description="Inference level: none, rdfs, rdfs_plus, owl_rl"
    )
    materialize_on_load: bool = Field(
        default=False, 
        description="Auto-run inference after data loads"
    )


class UpdateRepositoryRequest(BaseModel):
    """Request to update repository metadata."""
    description: Optional[str] = Field(None, description="New description")
    tags: Optional[list[str]] = Field(None, description="New tags")
    reasoning_level: Optional[str] = Field(None, description="Inference level: none, rdfs, rdfs_plus, owl_rl")
    materialize_on_load: Optional[bool] = Field(None, description="Auto-run inference after data loads")


class RenameRepositoryRequest(BaseModel):
    """Request to rename a repository."""
    new_name: str = Field(..., description="New repository name")


class SPARQLQueryRequest(BaseModel):
    """SPARQL query for a specific repository."""
    query: str = Field(..., description="SPARQL-Star query string")
    include_inferred: bool = Field(default=True, description="Include inferred triples in results")


class SQLQueryRequest(BaseModel):
    """SQL query for a specific repository."""
    sql: str = Field(..., description="SQL query string")
    limit: Optional[int] = Field(None, description="Row limit (optional)")


class SQLAggregateRequest(BaseModel):
    """SQL aggregation request."""
    group_by: str = Field(..., description="Column(s) to group by")
    aggregations: dict = Field(..., description="Dict of output_name -> aggregation expression")
    table: str = Field(default="triples", description="Table to query")
    where: Optional[str] = Field(None, description="WHERE clause")
    order_by: Optional[str] = Field(None, description="ORDER BY clause")
    limit: Optional[int] = Field(None, description="LIMIT")


class RepositoryResponse(BaseModel):
    """Response containing repository info."""
    name: str
    description: str
    tags: list[str]
    created_at: str
    triple_count: int
    subject_count: int
    predicate_count: int
    # Reasoning configuration
    reasoning_level: str = "none"
    materialize_on_load: bool = False
    inferred_count: int = 0
    
    @classmethod
    def from_info(cls, info: RepositoryInfo) -> "RepositoryResponse":
        return cls(
            name=info.name,
            description=info.description,
            tags=info.tags,
            created_at=info.created_at.isoformat(),
            triple_count=info.triple_count,
            subject_count=info.subject_count,
            predicate_count=info.predicate_count,
            reasoning_level=getattr(info, 'reasoning_level', 'none'),
            materialize_on_load=getattr(info, 'materialize_on_load', False),
            inferred_count=getattr(info, 'inferred_count', 0),
        )


def dataframe_to_records(df: pl.DataFrame) -> list[dict]:
    """Convert Polars DataFrame to list of dicts for JSON serialization."""
    from datetime import datetime
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


# Provenance predicate URIs to recognize in RDF-Star annotations
PROV_SOURCE_PREDICATES = {
    "http://www.w3.org/ns/prov#wasDerivedFrom",
    "http://www.w3.org/ns/prov#wasAttributedTo",
    "http://www.w3.org/ns/prov#hadPrimarySource",
    # RDF-StarBase native predicates
    "<http://rdf-starbase.dev/source>",
    "http://rdf-starbase.dev/source",
}
PROV_PROCESS_PREDICATES = {
    "http://www.w3.org/ns/prov#wasGeneratedBy",
    "<http://rdf-starbase.dev/process>",
    "http://rdf-starbase.dev/process",
}
PROV_CONFIDENCE_PREDICATES = {
    "http://www.w3.org/ns/prov#value",
    "<http://rdf-starbase.dev/confidence>",
    "http://rdf-starbase.dev/confidence",
}
PROV_TIMESTAMP_PREDICATES = {
    "http://www.w3.org/ns/prov#generatedAtTime",
    "<http://rdf-starbase.dev/recordedAt>",
    "http://rdf-starbase.dev/recordedAt",
    "<http://rdf-starbase.dev/asOf>",
    "http://rdf-starbase.dev/asOf",
}


def extract_rdfstar_annotations(parsed_result, default_source: str = "import") -> dict:
    """
    Extract RDF-Star annotations and separate base triples from annotation triples.
    
    Returns a dict with:
    - base_triples: List of non-annotation triples (s, p, o tuples)
    - annotations: Dict mapping (s, p, o) -> {source, confidence, process, timestamp}
    - default_source: Fallback source for triples without annotations
    """
    base_triples = []
    annotations = {}  # Key: (s, p, o) -> {source, confidence, process, timestamp}
    
    # Get triples list from parsed result
    if hasattr(parsed_result, 'triples'):
        triples = parsed_result.triples
    elif isinstance(parsed_result, list):
        triples = parsed_result
    else:
        triples = []
    
    for triple in triples:
        # Check if this is an RDF-Star annotation (subject_triple is set)
        if hasattr(triple, 'subject_triple') and triple.subject_triple is not None:
            # This is an annotation: << s p o >> predicate object
            qt = triple.subject_triple
            key = (qt.subject, qt.predicate, qt.object)
            
            if key not in annotations:
                annotations[key] = {}
            
            pred = triple.predicate
            obj_val = triple.object
            
            # Extract the value (strip datatype for literals)
            if isinstance(obj_val, str) and "^^" in obj_val:
                obj_val = obj_val.split("^^")[0].strip('"')
            elif isinstance(obj_val, str):
                obj_val = obj_val.strip('"')
            
            if pred in PROV_SOURCE_PREDICATES:
                annotations[key]["source"] = triple.object  # Keep full URI
            elif pred in PROV_PROCESS_PREDICATES:
                annotations[key]["process"] = triple.object  # Keep full URI
            elif pred in PROV_CONFIDENCE_PREDICATES:
                try:
                    annotations[key]["confidence"] = float(obj_val)
                except (ValueError, TypeError):
                    annotations[key]["confidence"] = 1.0
            elif pred in PROV_TIMESTAMP_PREDICATES:
                annotations[key]["timestamp"] = obj_val
        else:
            # Regular triple (not an annotation)
            if hasattr(triple, 'subject'):
                base_triples.append((triple.subject, triple.predicate, triple.object))
            elif isinstance(triple, dict):
                s = triple.get("subject", triple.get("s"))
                p = triple.get("predicate", triple.get("p"))
                o = triple.get("object", triple.get("o"))
                base_triples.append((s, p, o))
    
    return {
        "base_triples": base_triples,
        "annotations": annotations,
        "default_source": default_source,
    }


def extract_columnar(parsed_result) -> tuple[list[str], list[str], list[str]]:
    """
    Extract columnar triple data from parser output for fast ingestion.
    
    Returns (subjects, predicates, objects) lists for columnar insert.
    """
    # Fast path: ParsedDocument with to_columnar method
    if hasattr(parsed_result, 'to_columnar'):
        return parsed_result.to_columnar()
    
    # Handle ParsedDocument without to_columnar (older format)
    if hasattr(parsed_result, 'triples'):
        triples = parsed_result.triples
        return (
            [t.subject for t in triples],
            [t.predicate for t in triples],
            [t.object for t in triples],
        )
    
    # Handle list of Triple objects
    if parsed_result and hasattr(parsed_result[0], 'subject'):
        return (
            [t.subject for t in parsed_result],
            [t.predicate for t in parsed_result],
            [t.object for t in parsed_result],
        )
    
    # Handle list of dicts
    return (
        [t.get("subject", t.get("s")) for t in parsed_result],
        [t.get("predicate", t.get("p")) for t in parsed_result],
        [t.get("object", t.get("o")) for t in parsed_result],
    )


def _parse_rdf_data(data: str, format_type: str):
    """
    Parse RDF data in the specified format.
    
    Uses Oxigraph (Rust) acceleration for Turtle and NTriples when
    available and the data does not contain RDF-Star syntax.
    Falls back to hand-written Python parsers otherwise.
    
    Args:
        data: RDF content as string
        format_type: Format identifier (turtle, ntriples, jsonld, etc.)
        
    Returns:
        Parsed result with triples
    """
    format_type = format_type.lower().replace('-', '').replace('_', '').replace(' ', '')
    
    if format_type in ("turtle", "ttl", "turtlestar"):
        # Try Oxigraph (Rust) fast path for non-RDF-Star content
        has_rdfstar = '<<' in data and '>>' in data
        if not has_rdfstar:
            try:
                from pyoxigraph import parse as oxigraph_parse, RdfFormat
                quads = list(oxigraph_parse(data.encode('utf-8'), RdfFormat.TURTLE, base_iri="http://example.org/"))
                triples = []
                for q in quads:
                    s = str(q.subject)
                    p = str(q.predicate)
                    o = str(q.object)
                    triples.append({"subject": s, "predicate": p, "object": o})
                return triples
            except Exception:
                pass  # Fall back to Python parser
        from rdf_starbase.formats.turtle import parse_turtle
        return parse_turtle(data)
    
    elif format_type in ("ntriples", "nt", "ntriplesstar"):
        has_rdfstar = '<<' in data and '>>' in data
        if not has_rdfstar:
            try:
                from pyoxigraph import parse as oxigraph_parse, RdfFormat
                quads = list(oxigraph_parse(data.encode('utf-8'), RdfFormat.N_TRIPLES, base_iri="http://example.org/"))
                triples = []
                for q in quads:
                    s = str(q.subject)
                    p = str(q.predicate)
                    o = str(q.object)
                    triples.append({"subject": s, "predicate": p, "object": o})
                return triples
            except Exception:
                pass
        from rdf_starbase.formats.ntriples import parse_ntriples
        return parse_ntriples(data)
    
    elif format_type in ("nquads", "nq"):
        from rdf_starbase.formats.nquads import parse_nquads_as_triples
        return parse_nquads_as_triples(data)
    
    elif format_type in ("n3", "notation3"):
        from rdf_starbase.formats.n3 import parse_n3
        return parse_n3(data)
    
    elif format_type in ("trig", "trigstar"):
        from rdf_starbase.formats.trig import parse_trig_as_document
        return parse_trig_as_document(data)
    
    elif format_type in ("trix",):
        from rdf_starbase.formats.trix import parse_trix_as_document
        return parse_trix_as_document(data)
    
    elif format_type in ("jsonld", "json-ld"):
        from rdf_starbase.formats.jsonld import parse_jsonld
        return parse_jsonld(data)
    
    elif format_type in ("ndjsonld", "ndjson-ld", "jsonldstream"):
        from rdf_starbase.formats.ndjsonld import parse_ndjsonld_as_document
        return parse_ndjsonld_as_document(data)
    
    elif format_type in ("rdfjson", "rdf/json"):
        from rdf_starbase.formats.rdfjson import parse_rdfjson_as_document
        return parse_rdfjson_as_document(data)
    
    elif format_type in ("rdfxml", "rdf/xml", "xml", "rdf"):
        from rdf_starbase.formats.rdfxml import parse_rdfxml
        return parse_rdfxml(data)
    
    else:
        raise ValueError(f"Unsupported format: {format_type}. Supported formats: turtle, ntriples, nquads, n3, trig, trix, jsonld, ndjsonld, rdfjson, rdfxml")


def _parse_binary_rdf(data: bytes):
    """Parse Binary RDF format (requires bytes, not string)."""
    from rdf_starbase.formats.binaryrdf import parse_binaryrdf_as_document
    return parse_binaryrdf_as_document(data)


# File extension to format mapping
FORMAT_EXTENSIONS = {
    'ttl': 'turtle',
    'turtle': 'turtle',
    'nt': 'ntriples',
    'ntriples': 'ntriples',
    'nq': 'nquads',
    'nquads': 'nquads',
    'n3': 'n3',
    'trig': 'trig',
    'trix': 'trix',
    'rdf': 'rdfxml',
    'xml': 'rdfxml',
    'rdfxml': 'rdfxml',
    'jsonld': 'jsonld',
    'json': 'jsonld',
    'ndjsonld': 'ndjsonld',
    'ndjson': 'ndjsonld',
    'rdfjson': 'rdfjson',
    'brf': 'binaryrdf',
    'brdf': 'binaryrdf',
}


# In-memory staging area for uploaded files awaiting processing
_staged_uploads: dict[str, dict] = {}


def _sse(data: dict) -> str:
    """Format a dict as an SSE event line."""
    return f"data: {json.dumps(data)}\n\n"


def create_repository_router(
    workspace_path: Optional[str | Path] = None
) -> tuple[APIRouter, RepositoryManager]:
    """
    Create the repository management API router.
    
    Args:
        workspace_path: Path to store repositories (default: ./data/repositories)
        
    Returns:
        Tuple of (router, manager)
    """
    # Default workspace path
    if workspace_path is None:
        workspace_path = os.environ.get(
            "RDFSTARBASE_REPOSITORY_PATH",
            "./data/repositories"
        )
    
    manager = RepositoryManager(workspace_path)
    router = APIRouter(prefix="/repositories", tags=["Repositories"])
    
    # =========================================================================
    # Repository CRUD
    # =========================================================================
    
    @router.get("")
    async def list_repositories():
        """List all repositories with inline stats."""
        repos = manager.list_repositories()
        return {
            "count": len(repos),
            "repositories": [RepositoryResponse.from_info(r).model_dump() for r in repos]
        }
    
    @router.get("/bulk-stats")
    async def bulk_repository_stats():
        """Get stats for all repositories in a single call.
        
        Returns a dict keyed by repository name with stats for each.
        Avoids N separate /stats calls from the client.
        Only computes stats for already-loaded stores; unloaded repos
        return their persisted metadata counts.
        """
        def _compute_bulk():
            result: dict[str, dict] = {}
            for name in sorted(manager._info):
                info = manager._info[name]
                if name in manager._stores:
                    raw = manager._stores[name].stats()
                    result[name] = {
                        "name": name,
                        "description": info.description,
                        "created_at": info.created_at.isoformat(),
                        "triple_count": raw.get("active_assertions", 0),
                        "subject_count": raw.get("unique_subjects", 0),
                        "predicate_count": raw.get("unique_predicates", 0),
                        "object_count": raw.get("term_dict", {}).get("total_terms", 0),
                        "graph_count": raw.get("fact_store", {}).get("graphs", 0),
                        "source_count": raw.get("unique_sources", 0),
                        "loaded": True,
                    }
                else:
                    result[name] = {
                        "name": name,
                        "description": info.description,
                        "created_at": info.created_at.isoformat(),
                        "triple_count": info.triple_count,
                        "subject_count": info.subject_count,
                        "predicate_count": info.predicate_count,
                        "object_count": 0,
                        "graph_count": 0,
                        "source_count": 0,
                        "loaded": False,
                    }
            return result
        
        result = await asyncio.to_thread(_compute_bulk)
        return {"repositories": result}
    
    @router.post("")
    async def create_repository(request: CreateRepositoryRequest):
        """Create a new repository with optional reasoning configuration."""
        try:
            info = manager.create(
                name=request.name,
                description=request.description,
                tags=request.tags,
                reasoning_level=request.reasoning_level,
                materialize_on_load=request.materialize_on_load,
            )
            # Auto-save after creation
            manager.save(request.name)
            return {
                "success": True,
                "message": f"Repository '{request.name}' created with reasoning_level='{request.reasoning_level}'",
                "repository": RepositoryResponse.from_info(info).model_dump()
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.get("/{name}")
    async def get_repository(name: str):
        """Get repository info."""
        try:
            info = manager.get_info(name)
            return RepositoryResponse.from_info(info).model_dump()
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    @router.patch("/{name}")
    async def update_repository(name: str, request: UpdateRepositoryRequest):
        """Update repository metadata and reasoning configuration."""
        try:
            # Update basic info
            info = manager.update_info(
                name=name,
                description=request.description,
                tags=request.tags,
            )
            # Update reasoning config if provided
            if request.reasoning_level is not None or request.materialize_on_load is not None:
                info = manager.update_reasoning_config(
                    name=name,
                    reasoning_level=request.reasoning_level,
                    materialize_on_load=request.materialize_on_load,
                )
            return RepositoryResponse.from_info(info).model_dump()
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    @router.delete("/{name}")
    async def delete_repository(
        name: str,
        force: bool = Query(False, description="Force delete even if repository has data")
    ):
        """Delete a repository."""
        try:
            manager.delete(name, force=force)
            return {
                "success": True,
                "message": f"Repository '{name}' deleted"
            }
        except ValueError as e:
            if "does not exist" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.post("/{name}/rename")
    async def rename_repository(name: str, request: RenameRepositoryRequest):
        """Rename a repository."""
        try:
            info = manager.rename(name, request.new_name)
            return {
                "success": True,
                "message": f"Repository renamed from '{name}' to '{request.new_name}'",
                "repository": RepositoryResponse.from_info(info).model_dump()
            }
        except ValueError as e:
            if "does not exist" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))
    
    # =========================================================================
    # Repository Inference
    # =========================================================================
    
    @router.post("/{name}/inference")
    async def run_inference(
        name: str,
        level: Optional[str] = Query(None, description="Override reasoning level (uses repo config if not set)")
    ):
        """
        Run inference/reasoning on the repository.
        
        Materializes inferred triples based on RDFS/OWL rules.
        Similar to GraphDB's 'Reinfer' operation.
        """
        try:
            result = manager.materialize_inferences(name, level=level)
            if result.get("triples_inferred", 0) > 0:
                manager.save(name)
            return result
        except ValueError as e:
            if "does not exist" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
    @router.get("/{name}/inference/stats")
    async def get_inference_stats(name: str):
        """Get inference statistics for a repository."""
        try:
            info = manager.get_info(name)
            store = manager.get_store(name)
            
            # Count inferred vs asserted facts
            df = store._fact_store._df
            
            from rdf_starbase.storage.facts import FactFlags
            inferred_mask = (df["flags"] & FactFlags.INFERRED) != 0
            inferred_count = inferred_mask.sum()
            asserted_count = len(df) - inferred_count
            
            return {
                "repository": name,
                "reasoning_level": info.reasoning_level,
                "materialize_on_load": info.materialize_on_load,
                "asserted_count": int(asserted_count),
                "inferred_count": int(inferred_count),
                "total_count": len(df),
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    # =========================================================================
    # Repository SPARQL
    # =========================================================================
    
    @router.post("/{name}/sparql")
    async def repository_sparql(name: str, request: SPARQLQueryRequest):
        """Execute a SPARQL query against a specific repository."""
        try:
            store = await asyncio.to_thread(manager.get_store, name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        try:
            result = await asyncio.to_thread(execute_sparql, store, request.query)
            
            if isinstance(result, bool):
                # ASK query
                return {"type": "ask", "result": result}
            elif isinstance(result, dict):
                # UPDATE operation
                # Auto-save after update
                manager.save(name)
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
    
    # =========================================================================
    # Repository Ontology Discovery
    # =========================================================================
    
    @router.get("/{name}/ontology")
    async def get_repository_ontology(name: str, request: Request):
        """
        Extract ontology (classes and properties) from the repository.
        
        Queries for RDFS/OWL declarations:
        - Classes: rdfs:Class, owl:Class
        - Properties: rdf:Property, rdfs:Property, owl:DatatypeProperty, owl:ObjectProperty
        - Labels: rdfs:label, skos:altLabel
        - Descriptions: rdfs:comment
        - Domain/Range: rdfs:domain, rdfs:range
        
        Returns structured data for the Starchart mapper.
        """
        try:
            store = await asyncio.to_thread(manager.get_store, name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        try:
            # ── Fast integer-level ontology extraction ───────────────
            raw = await asyncio.to_thread(store.get_ontology_summary)

            # Build alias lookup (already built by store method)
            aliases_map = raw.get("aliases_map", {})
            
            # Standard prefixes
            known_prefixes = {
                'http://www.w3.org/2001/XMLSchema#': 'xsd',
                'http://www.w3.org/2000/01/rdf-schema#': 'rdfs',
                'http://www.w3.org/1999/02/22-rdf-syntax-ns#': 'rdf',
                'http://www.w3.org/2002/07/owl#': 'owl',
                'http://xmlns.com/foaf/0.1/': 'foaf',
                'http://schema.org/': 'schema',
                'http://purl.org/dc/terms/': 'dcterms',
                'http://www.w3.org/2004/02/skos/core#': 'skos',
                'http://www.w3.org/ns/prov#': 'prov',
            }
            
            # Extract namespaces from URIs and build prefix map
            discovered_namespaces = {}
            prefix_counter = 0
            
            def extract_namespace(uri):
                """Extract namespace from URI."""
                if not uri:
                    return None
                if '#' in uri:
                    return uri.rsplit('#', 1)[0] + '#'
                elif '/' in uri:
                    return uri.rsplit('/', 1)[0] + '/'
                return None
            
            def get_or_create_prefix(namespace):
                """Get prefix for namespace, creating one if needed."""
                nonlocal prefix_counter
                if not namespace:
                    return None
                # Check known prefixes
                if namespace in known_prefixes:
                    discovered_namespaces[namespace] = known_prefixes[namespace]
                    return known_prefixes[namespace]
                # Check already discovered
                if namespace in discovered_namespaces:
                    return discovered_namespaces[namespace]
                # Create new prefix based on namespace
                # Use last path segment as hint, but disambiguate collisions
                # e.g. gleif.org/ontology/L1/ and rdf.gleif.org/L1/ both have "L1"
                local_hint = namespace.rstrip('#/').rsplit('/', 1)[-1]
                if local_hint and local_hint[0].isalpha() and len(local_hint) <= 10:
                    prefix = local_hint.lower().replace('-', '_')
                else:
                    prefix = f'ns{prefix_counter}'
                    prefix_counter += 1
                # Disambiguate if prefix already used by a different namespace
                used_prefixes = set(discovered_namespaces.values())
                if prefix in used_prefixes:
                    # Try adding parent path segment(s) for disambiguation
                    parts = namespace.rstrip('#/').split('/')
                    # Walk backward through path segments to find unique combo
                    for depth in range(2, min(len(parts), 5)):
                        combo = '_'.join(
                            p.lower().replace('-', '_')
                            for p in parts[-depth:] if p
                        )
                        if combo and combo not in used_prefixes:
                            prefix = combo
                            break
                    else:
                        # Fallback to numbered prefix
                        prefix = f'ns{prefix_counter}'
                        prefix_counter += 1
                discovered_namespaces[namespace] = prefix
                return prefix
            
            # Pre-scan URIs to discover namespaces
            all_uris: list[str] = []
            for c in raw["classes"]:
                all_uris.append(c["uri"])
            for p in raw["properties"]:
                all_uris.append(p["uri"])
                if p.get("domain"):
                    all_uris.append(p["domain"])
                if p.get("range"):
                    all_uris.append(p["range"])
            
            for uri in all_uris:
                if uri:
                    ns = extract_namespace(uri)
                    if ns:
                        get_or_create_prefix(ns)
            
            def compact_uri(uri):
                """Compact URI using discovered prefixes."""
                if not uri:
                    return None
                ns = extract_namespace(uri)
                if ns and ns in discovered_namespaces:
                    prefix = discovered_namespaces[ns]
                    local = uri[len(ns):]
                    return f"{prefix}:{local}"
                # Fall back to full URI
                return uri
            
            def extract_label(uri):
                if '#' in uri:
                    return uri.split('#')[-1]
                elif '/' in uri:
                    return uri.split('/')[-1]
                return uri
            
            # Process classes
            classes = []
            for c in raw["classes"]:
                uri = c["uri"]
                classes.append({
                    'uri': compact_uri(uri),
                    'label': c.get("label") or extract_label(uri),
                    'description': c.get("comment"),
                    'properties': [],  # Will be filled by domain matching
                })
            
            # Process properties
            properties = []
            for p in raw["properties"]:
                uri = p["uri"]
                domain_compact = compact_uri(p.get("domain")) if p.get("domain") else None
                
                prop_obj = {
                    'uri': compact_uri(uri),
                    'label': p.get("label") or extract_label(uri),
                    'aliases': p.get("aliases", []),
                    'description': p.get("comment"),
                    'domain': domain_compact,
                    'range': compact_uri(p.get("range")) if p.get("range") else 'xsd:string',
                }
                properties.append(prop_obj)
                
                # Associate property with class by domain
                if domain_compact:
                    for cls in classes:
                        if cls['uri'] == domain_compact:
                            cls['properties'].append(prop_obj['uri'])
            
            body = {
                'success': True,
                'repository': name,
                'classes': classes,
                'properties': properties,
                'class_count': len(classes),
                'property_count': len(properties),
                # Return inverted prefix map (prefix -> namespace)
                'prefixes': {prefix: ns for ns, prefix in discovered_namespaces.items()},
            }
            
            # ETag based on class/property counts and store size
            etag_seed = f"{len(classes)}-{len(properties)}-{len(store)}"
            etag = f'W/"{hashlib.md5(etag_seed.encode()).hexdigest()[:16]}"'
            
            if_none_match = request.headers.get("if-none-match")
            if if_none_match and if_none_match == etag:
                return JSONResponse(status_code=304, content=None, headers={"ETag": etag})
            
            return JSONResponse(content=body, headers={"ETag": etag, "Cache-Control": "private, max-age=30"})
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to extract ontology: {str(e)}")
    
    # =========================================================================
    # Repository Triple Management
    # =========================================================================
    
    @router.get("/{name}/triples")
    async def get_repository_triples(
        name: str,
        subject: Optional[str] = Query(None, description="Filter by subject"),
        predicate: Optional[str] = Query(None, description="Filter by predicate"),
        object: Optional[str] = Query(None, description="Filter by object"),
        limit: int = Query(100, ge=1, le=10000, description="Maximum results"),
    ):
        """Get triples from a specific repository."""
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        df = store.get_triples(
            subject=subject,
            predicate=predicate,
            obj=object,
        )
        
        df = df.head(limit)
        
        return {
            "count": len(df),
            "triples": dataframe_to_records(df),
        }
    
    @router.post("/{name}/triples/batch")
    async def add_repository_triples_batch(
        name: str,
        triples: list[dict]
    ):
        """Add multiple triples to a specific repository."""
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        try:
            count = store.add_triples_batch(triples)
            # Auto-save after batch insert
            manager.save(name)
            return {
                "success": True,
                "count": count,
                "message": f"Added {count} triples to repository '{name}'",
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # =========================================================================
    # Repository Stats
    # =========================================================================
    
    @router.get("/{name}/stats")
    async def get_repository_stats(name: str, request: Request):
        """Get detailed statistics for a repository.
        
        Returns persisted metadata counts instantly if the store is not
        already loaded in memory.  Only computes live stats (unique
        subjects, graph count, etc.) when the store happens to be loaded.
        This prevents a 12+ second disk load just to show dashboard cards.
        """
        try:
            info = manager.get_info(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        # If store is already in memory, use live stats
        loaded = name in manager._stores
        if loaded:
            raw_stats = await asyncio.to_thread(manager._stores[name].stats)
            triple_count = raw_stats.get("active_assertions", 0)
            subject_count = raw_stats.get("unique_subjects", 0)
            predicate_count = raw_stats.get("unique_predicates", 0)
            object_count = raw_stats.get("term_dict", {}).get("total_terms", 0)
            graph_count = raw_stats.get("fact_store", {}).get("graphs", 0)
            source_count = raw_stats.get("unique_sources", 0)
            details = raw_stats
        else:
            # Use persisted metadata — instant, no disk load
            triple_count = info.triple_count
            subject_count = info.subject_count
            predicate_count = info.predicate_count
            object_count = 0
            graph_count = 0
            source_count = 0
            details = None
        
        # Build ETag
        etag_seed = f"{triple_count}-{subject_count}-{predicate_count}-{loaded}"
        etag = f'W/"{hashlib.md5(etag_seed.encode()).hexdigest()[:16]}"'
        
        # Check If-None-Match
        if_none_match = request.headers.get("if-none-match")
        if if_none_match and if_none_match == etag:
            return JSONResponse(status_code=304, content=None, headers={"ETag": etag})
        
        body = {
            "name": name,
            "description": info.description,
            "created_at": info.created_at.isoformat(),
            "triple_count": triple_count,
            "subject_count": subject_count,
            "predicate_count": predicate_count,
            "object_count": object_count,
            "graph_count": graph_count,
            "source_count": source_count,
            "loaded": loaded,
        }
        if details is not None:
            body["details"] = details
        return JSONResponse(content=body, headers={"ETag": etag, "Cache-Control": "private, max-age=5"})
    
    # =========================================================================
    # Import / Export
    # =========================================================================
    
    def _resolve_graph_target(graph_target: Optional[str], filename: Optional[str] = None) -> Optional[str]:
        """
        Resolve graph_target parameter to a graph URI.
        
        Args:
            graph_target: Graph target mode:
                - None or 'default': Use default graph (returns None)
                - 'named:<uri>': Use specified named graph URI
                - 'auto': Auto-generate from filename
            filename: Optional filename for 'auto' mode
            
        Returns:
            Graph URI string or None for default graph
        """
        if graph_target is None or graph_target == "" or graph_target.lower() == "default":
            return None
        
        if graph_target.lower() == "auto":
            if filename:
                # Generate graph URI from filename
                # Remove extension and sanitize
                base = filename.rsplit('.', 1)[0] if '.' in filename else filename
                # Replace spaces and special chars
                safe_name = base.replace(' ', '_').replace('\\', '/').split('/')[-1]
                return f"http://rdf-starbase.io/graph/{safe_name}"
            else:
                # No filename, use timestamp-based graph
                import time
                return f"http://rdf-starbase.io/graph/import_{int(time.time())}"
        
        if graph_target.lower().startswith("named:"):
            # Extract the URI after 'named:'
            return graph_target[6:].strip()
        
        # Treat as direct URI
        return graph_target

    @router.post("/{name}/import")
    async def import_data(name: str, request: dict):
        """Import RDF data in various formats.
        
        Supported formats:
        - turtle, turtle-star: Turtle and Turtle-Star
        - ntriples: N-Triples and N-Triples-Star  
        - nquads: N-Quads with named graphs
        - n3: Notation3
        - trig, trig-star: TriG with named graphs
        - trix: TriX XML format
        - jsonld: JSON-LD
        - ndjsonld: Newline-delimited JSON-LD
        - rdfjson: RDF/JSON (W3C format)
        - rdfxml: RDF/XML
        - binaryrdf: Binary RDF format
        
        Graph target options (graph_target parameter):
        - 'default' or omit: Load into the default graph
        - 'named:<uri>': Load into a specific named graph
        - 'auto': Auto-generate graph name from source
        - Direct URI: Use as named graph
        """
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        data = request.get("data", "")
        format_type = request.get("format", "turtle").lower()
        graph_target = request.get("graph_target", None)
        
        # Resolve graph target
        target_graph = _resolve_graph_target(graph_target)
        
        # Handle data that might come as dict (shouldn't happen but defensive)
        if not isinstance(data, str):
            raise HTTPException(status_code=400, detail="Data must be a string")
        
        if not data.strip():
            raise HTTPException(status_code=400, detail="No data provided")
        
        try:
            parsed = _parse_rdf_data(data, format_type)
            
            # Extract RDF-Star annotations and base triples
            rdfstar_data = extract_rdfstar_annotations(parsed, default_source="import")
            base_triples = rdfstar_data["base_triples"]
            annotations = rdfstar_data["annotations"]
            
            # Group triples by provenance for batch columnar inserts
            prov_groups = {}  # (source, confidence) -> [(s, p, o), ...]
            
            for s, p, o in base_triples:
                key = (s, p, o)
                if key in annotations:
                    ann = annotations[key]
                    source = ann.get("source", "import")
                    confidence = ann.get("confidence", 1.0)
                else:
                    source = "import"
                    confidence = 1.0
                
                prov_key = (source, confidence)
                if prov_key not in prov_groups:
                    prov_groups[prov_key] = []
                prov_groups[prov_key].append((s, p, o))
            
            # Batch columnar insert for each provenance group
            count = 0
            for (source, confidence), triples_list in prov_groups.items():
                subjects = [t[0] for t in triples_list]
                predicates = [t[1] for t in triples_list]
                objects = [t[2] for t in triples_list]
                count += store.add_triples_columnar(
                    subjects=subjects,
                    predicates=predicates,
                    objects=objects,
                    source=source,
                    confidence=confidence,
                    graph=target_graph,
                )
            
            manager.save(name)
            
            return {
                "success": True,
                "format": format_type,
                "triples_added": count,
                "graph": target_graph,
                "message": f"Imported {count} triples from {format_type} data" + (f" into graph <{target_graph}>" if target_graph else " into default graph")
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")
    
    @router.post("/{name}/upload")
    async def upload_file(
        name: str,
        file: UploadFile = File(...),
        format: str = Form(None, description="Format: turtle, ntriples, nquads, n3, trig, trix, jsonld, ndjsonld, rdfjson, rdfxml, binaryrdf (auto-detect from extension if not provided)"),
        graph_target: str = Form(None, description="Graph target: 'default', 'named:<uri>', 'auto' (generate from filename), or direct URI")
    ):
        """
        Upload an RDF file directly for fast import.
        
        Supported extensions: .ttl, .nt, .nq, .n3, .trig, .trix, .jsonld, .json, 
                             .ndjsonld, .ndjson, .rdfjson, .rdf, .xml, .brf
        
        Formats with RDF-Star support: turtle, ntriples, trig
        
        For Turtle/N-Triples files without RDF-Star, uses streaming Oxigraph
        parsing with chunked insertion to avoid loading the entire file into
        memory.
        
        Graph target options:
        - 'default' or omit: Load into the default graph
        - 'named:<uri>': Load into a specific named graph
        - 'auto': Auto-generate graph URI from filename
        - Direct URI: Use as named graph
        """
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        # Auto-detect format from filename
        if not format:
            ext = file.filename.split('.')[-1].lower() if file.filename else ''
            format = FORMAT_EXTENSIONS.get(ext, 'turtle')
        
        # Resolve graph target (with filename for 'auto' mode)
        target_graph = _resolve_graph_target(graph_target, file.filename)
        
        # Stream upload to temp file to avoid holding entire file in memory
        safe_name = (file.filename or 'data').replace(os.sep, '_')
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=f"_{safe_name}", dir=tempfile.gettempdir()
        )
        try:
            while True:
                chunk = await file.read(8 * 1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
        finally:
            tmp.close()
        tmp_path = tmp.name
        
        try:
            start_time = time.time()
            default_source = f"file:{file.filename}"
            
            # --- Try bulk_load_file (high-performance direct path) ---
            try:
                count = store.bulk_load_file(
                    tmp_path,
                    graph_uri=target_graph,
                    source=default_source,
                )
                manager.save(name)
                total_time = time.time() - start_time
                triples_per_sec = count / total_time if total_time > 0 else 0
                
                return {
                    "success": True,
                    "filename": file.filename,
                    "format": format,
                    "triples_added": count,
                    "graph": target_graph,
                    "timing": {
                        "parse_seconds": round(total_time, 3),
                        "insert_seconds": round(total_time, 3),
                        "total_seconds": round(total_time, 3),
                        "triples_per_second": round(triples_per_sec, 0),
                    },
                    "message": f"Imported {count} triples in {total_time:.2f}s"
                              + (f" into graph <{target_graph}>" if target_graph else "")
                }
            except Exception:
                pass  # fall through to legacy path
            
            # --- Legacy path: read file, parse in memory ---
            with open(tmp_path, 'rb') as f:
                content = f.read()
            
            if format == "binaryrdf":
                triples = _parse_binary_rdf(content)
            else:
                data = content.decode('utf-8')
                del content
                triples = _parse_rdf_data(data, format)
                del data
            
            parse_time = time.time() - start_time
            
            insert_start = time.time()
            
            # Extract RDF-Star annotations and base triples
            rdfstar_data = extract_rdfstar_annotations(triples, default_source=default_source)
            base_triples = rdfstar_data["base_triples"]
            annotations = rdfstar_data["annotations"]
            
            # Group triples by provenance for batch columnar inserts
            prov_groups = {}  # (source, confidence) -> [(s, p, o), ...]
            
            for s, p, o in base_triples:
                key = (s, p, o)
                if key in annotations:
                    ann = annotations[key]
                    source = ann.get("source", default_source)
                    confidence = ann.get("confidence", 1.0)
                else:
                    source = default_source
                    confidence = 1.0
                
                prov_key = (source, confidence)
                if prov_key not in prov_groups:
                    prov_groups[prov_key] = []
                prov_groups[prov_key].append((s, p, o))
            
            # Batch columnar insert for each provenance group
            store.begin_bulk_insert()
            count = 0
            for (source, confidence), triples_list in prov_groups.items():
                subjects = [t[0] for t in triples_list]
                predicates = [t[1] for t in triples_list]
                objects = [t[2] for t in triples_list]
                count += store.add_triples_columnar(
                    subjects=subjects,
                    predicates=predicates,
                    objects=objects,
                    source=source,
                    confidence=confidence,
                    graph=target_graph,
                )
            store.end_bulk_insert()
            
            insert_time = time.time() - insert_start
            
            manager.save(name)
            total_time = time.time() - start_time
            
            # Calculate throughput
            triples_per_sec = count / total_time if total_time > 0 else 0
            
            return {
                "success": True,
                "filename": file.filename,
                "format": format,
                "triples_added": count,
                "graph": target_graph,
                "timing": {
                    "parse_seconds": round(parse_time, 3),
                    "insert_seconds": round(insert_time, 3),
                    "total_seconds": round(total_time, 3),
                    "triples_per_second": round(triples_per_sec, 0),
                },
                "message": f"Imported {count} triples in {total_time:.2f}s" + (f" into graph <{target_graph}>" if target_graph else "")
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # =========================================================================
    # Staged Upload with Progress (two-phase)
    # =========================================================================

    @router.post("/{name}/upload-stage")
    async def upload_stage(
        name: str,
        file: UploadFile = File(...),
        format: str = Form(None),
        graph_target: str = Form(None),
    ):
        """
        Phase 1: Stage a file upload. Streams the file to a temp location
        and returns a stage_id. Use GET /{name}/upload-process/{stage_id}
        to start processing with SSE progress events.
        """
        try:
            manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # Auto-detect format
        if not format:
            ext = file.filename.split('.')[-1].lower() if file.filename else ''
            format = FORMAT_EXTENSIONS.get(ext, 'turtle')

        # Stream to temp file instead of holding entire file in memory
        safe_name = (file.filename or 'data').replace(os.sep, '_')
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=f"_{safe_name}", dir=tempfile.gettempdir()
        )
        file_size = 0
        try:
            while True:
                chunk = await file.read(8 * 1024 * 1024)  # 8 MB chunks
                if not chunk:
                    break
                tmp.write(chunk)
                file_size += len(chunk)
        finally:
            tmp.close()

        stage_id = str(uuid.uuid4())

        _staged_uploads[stage_id] = {
            "temp_path": tmp.name,
            "filename": file.filename,
            "format": format,
            "graph_target": graph_target,
            "repo_name": name,
            "size": file_size,
            "staged_at": time.time(),
        }

        return {
            "stage_id": stage_id,
            "filename": file.filename,
            "size": file_size,
            "format": format,
        }

    @router.get("/{name}/upload-process/{stage_id}")
    async def upload_process(name: str, stage_id: str):
        """
        Phase 2: Process a staged upload with SSE progress events.
        Streams events: parsing, inserting (with progress/ETA), done, error.
        
        For Turtle/N-Triples files without RDF-Star, uses streaming
        Oxigraph parsing with chunked insertion to avoid loading the
        entire file into memory.
        """
        if stage_id not in _staged_uploads:
            raise HTTPException(status_code=404, detail="Stage ID not found or expired")

        staged = _staged_uploads.pop(stage_id)
        if staged["repo_name"] != name:
            raise HTTPException(status_code=400, detail="Repository mismatch")

        async def event_stream():
            temp_path = staged.get("temp_path")
            try:
                store = manager.get_store(name)
                fmt = staged["format"]
                filename = staged["filename"]
                graph_target = staged.get("graph_target")
                target_graph = _resolve_graph_target(graph_target, filename)
                default_source = f"file:{filename}"

                # ----------------------------------------------------------
                # Try bulk_load_file (high-performance direct path)
                # ----------------------------------------------------------
                can_bulk = False
                if temp_path and hasattr(store, 'bulk_load_file'):
                    # bulk_load_file handles format detection, RDF-Star fallback,
                    # and pyoxigraph availability internally.
                    _bulk_formats = {"turtle", "ttl", "ntriples", "nt"}
                    fmt_key = fmt.lower().replace('-', '').replace('_', '').replace(' ', '')
                    if fmt_key in _bulk_formats:
                        can_bulk = True

                if can_bulk:
                    yield _sse({"phase": "parsing", "progress": 0,
                                "message": "Bulk loading RDF data…"})
                    await asyncio.sleep(0)

                    parse_start = time.time()
                    last_report = [parse_start]

                    def _progress(loaded):
                        """Called by bulk_load_file every 2M triples."""
                        now = time.time()
                        last_report[0] = now

                    try:
                        count = store.bulk_load_file(
                            temp_path,
                            graph_uri=target_graph,
                            source=default_source,
                            on_progress=_progress,
                        )
                    except Exception as bulk_err:
                        # fall through to legacy path
                        can_bulk = False

                if can_bulk:
                    parse_time = time.time() - parse_start

                    # --- Phase: Saving ---
                    yield _sse({"phase": "saving", "progress": 100,
                                "message": "Saving repository..."})
                    await asyncio.sleep(0)
                    manager.save(name)

                    total_time = parse_time
                    triples_per_sec = count / total_time if total_time > 0 else 0
                    yield _sse({
                        "phase": "done", "progress": 100,
                        "triples_added": count, "total_triples": count,
                        "timing": {
                            "parse_seconds": round(parse_time, 3),
                            "insert_seconds": round(parse_time, 3),
                            "total_seconds": round(total_time, 3),
                            "triples_per_second": round(triples_per_sec),
                        },
                        "message": f"Imported {count:,} triples in {total_time:.2f}s",
                    })
                    return  # done — skip legacy path

                # ----------------------------------------------------------
                # Legacy path: read file into memory, parse, then insert
                # ----------------------------------------------------------
                yield _sse({"phase": "parsing", "progress": 0,
                            "message": "Parsing RDF data..."})
                await asyncio.sleep(0)

                parse_start = time.time()
                if temp_path:
                    with open(temp_path, 'rb') as f:
                        content = f.read()
                else:
                    content = staged.get("content", b"")

                if fmt == "binaryrdf":
                    triples = _parse_binary_rdf(content)
                else:
                    data = content.decode('utf-8')
                    del content  # free the bytes copy
                    triples = _parse_rdf_data(data, fmt)
                    del data  # free the string copy
                parse_time = time.time() - parse_start

                total_triples = len(triples)
                yield _sse({"phase": "parsing", "progress": 100, "total_triples": total_triples,
                            "parse_seconds": round(parse_time, 3),
                            "message": f"Parsed {total_triples:,} triples in {parse_time:.2f}s"})
                await asyncio.sleep(0)

                # --- Phase: Preparing ---
                default_source = f"file:{filename}"
                rdfstar_data = extract_rdfstar_annotations(triples, default_source=default_source)
                base_triples = rdfstar_data["base_triples"]
                annotations = rdfstar_data["annotations"]

                # Flatten all triples into a single insertion list with provenance
                all_items = []
                for s, p, o in base_triples:
                    key = (s, p, o)
                    if key in annotations:
                        ann = annotations[key]
                        source = ann.get("source", default_source)
                        confidence = ann.get("confidence", 1.0)
                    else:
                        source = default_source
                        confidence = 1.0
                    all_items.append((s, p, o, source, confidence))

                total_items = len(all_items)
                if total_items == 0:
                    yield _sse({"phase": "done", "progress": 100, "triples_added": 0,
                                "message": "No triples found in file"})
                    return

                # --- Phase: Inserting (chunked with progress) ---
                CHUNK_SIZE = max(500, total_items // 100)  # ~100 progress updates max
                count = 0
                insert_start = time.time()

                # Enable deferred concat for O(1) per-chunk inserts
                store.begin_bulk_insert()

                # Group into provenance batches within each chunk
                for chunk_start in range(0, total_items, CHUNK_SIZE):
                    chunk = all_items[chunk_start : chunk_start + CHUNK_SIZE]

                    # Group chunk by (source, confidence)
                    prov_groups: dict[tuple, list] = {}
                    for s, p, o, src, conf in chunk:
                        key = (src, conf)
                        if key not in prov_groups:
                            prov_groups[key] = []
                        prov_groups[key].append((s, p, o))

                    for (source, confidence), triples_list in prov_groups.items():
                        subjects = [t[0] for t in triples_list]
                        predicates = [t[1] for t in triples_list]
                        objects = [t[2] for t in triples_list]
                        count += store.add_triples_columnar(
                            subjects=subjects,
                            predicates=predicates,
                            objects=objects,
                            source=source,
                            confidence=confidence,
                            graph=target_graph,
                        )

                    elapsed = time.time() - insert_start
                    progress = round((count / total_items) * 100)
                    rate = count / elapsed if elapsed > 0 else 0
                    remaining = total_items - count
                    eta = round(remaining / rate, 1) if rate > 0 else None

                    yield _sse({
                        "phase": "inserting",
                        "progress": progress,
                        "triples_inserted": count,
                        "total_triples": total_items,
                        "triples_per_second": round(rate),
                        "eta_seconds": eta,
                        "message": f"Inserted {count:,} / {total_items:,} triples",
                    })
                    await asyncio.sleep(0)  # yield control for SSE flush

                # Single concat for all buffered DataFrames
                store.end_bulk_insert()

                # --- Phase: Saving ---
                yield _sse({"phase": "saving", "progress": 100, "message": "Saving repository..."})
                await asyncio.sleep(0)
                manager.save(name)

                insert_time = time.time() - insert_start
                total_time = parse_time + insert_time
                triples_per_sec = count / total_time if total_time > 0 else 0

                yield _sse({
                    "phase": "done",
                    "progress": 100,
                    "triples_added": count,
                    "total_triples": total_items,
                    "timing": {
                        "parse_seconds": round(parse_time, 3),
                        "insert_seconds": round(insert_time, 3),
                        "total_seconds": round(total_time, 3),
                        "triples_per_second": round(triples_per_sec),
                    },
                    "message": f"Imported {count:,} triples in {total_time:.2f}s",
                })
            except Exception as e:
                yield _sse({"phase": "error", "message": str(e)})
            finally:
                # Clean up temp file
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/{name}/export")
    async def export_data(
        name: str,
        format: str = Query("turtle", description="Export format: turtle, ntriples, rdfxml, jsonld")
    ):
        """Export all repository data in various formats."""
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        try:
            # Get all triples
            df = store.get_triples()
            triples = []
            for row in df.iter_rows(named=True):
                triples.append({
                    "subject": row.get("subject"),
                    "predicate": row.get("predicate"),
                    "object": row.get("object"),
                })
            
            # Serialize based on format
            if format == "turtle":
                from rdf_starbase.formats.turtle import serialize_turtle
                content = serialize_turtle(triples)
            elif format == "ntriples":
                from rdf_starbase.formats.ntriples import serialize_ntriples
                content = serialize_ntriples(triples)
            elif format == "rdfxml":
                from rdf_starbase.formats.rdfxml import serialize_rdfxml
                content = serialize_rdfxml(triples)
            elif format == "jsonld":
                from rdf_starbase.formats.jsonld import serialize_jsonld
                content = serialize_jsonld(triples)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
            
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content=content, media_type="text/plain")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    @router.post("/{name}/save")
    async def save_repository(name: str):
        """Explicitly save a repository to disk."""
        try:
            manager.save(name)
            return {
                "success": True,
                "message": f"Repository '{name}' saved"
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    @router.post("/save-all")
    async def save_all_repositories():
        """Save all loaded repositories to disk."""
        manager.save_all()
        return {
            "success": True,
            "message": "All repositories saved"
        }
    
    # =========================================================================
    # SQL Interface (DuckDB)
    # =========================================================================
    
    @router.get("/{name}/sql/status")
    async def sql_status(name: str):
        """Check if SQL interface is available for a repository."""
        from rdf_starbase.storage.duckdb import check_duckdb_available
        
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        available = check_duckdb_available()
        stats = store.stats()
        
        return {
            "available": available,
            "repository": name,
            "message": "DuckDB SQL interface ready" if available else "Install duckdb: pip install duckdb",
            "triple_count": stats.get("total_assertions", 0),
            "tables": ["triples", "facts", "terms", "provenance", "named_graphs", "rdf_types"] if available else [],
        }
    
    @router.post("/{name}/sql/query")
    async def sql_query(name: str, request: SQLQueryRequest):
        """
        Execute a SQL query against the repository's triple store.
        
        Available tables:
        - triples: Main triple data (subject, predicate, object, graph, source, confidence)
        - facts: Raw integer-encoded facts (for advanced users)
        - terms: Term dictionary (term_id, kind, lex)
        - provenance: Filtered view of triples with provenance info
        - named_graphs: List of distinct named graphs
        - rdf_types: Subject-type pairs (filtered on rdf:type predicate)
        
        Example queries:
        - SELECT * FROM triples LIMIT 10
        - SELECT predicate, COUNT(*) as count FROM triples GROUP BY predicate ORDER BY count DESC
        - SELECT subject, object FROM triples WHERE predicate LIKE '%type%'
        
        Performance note: Uses cached DuckDB interface for connection reuse.
        Queries on 'facts' and 'terms' tables are fastest (no string materialization).
        """
        from rdf_starbase.storage.duckdb import get_cached_interface, check_duckdb_available
        
        if not check_duckdb_available():
            raise HTTPException(
                status_code=501,
                detail="DuckDB not installed. Install with: pip install duckdb"
            )
        
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        try:
            # Use cached interface for connection reuse
            sql = get_cached_interface(store, read_only=True)
            result = sql.execute(request.sql, limit=request.limit)
            return {
                "columns": result.columns,
                "rows": result.rows,
                "row_count": result.row_count,
                "execution_time_ms": result.execution_time_ms,
                "warnings": result.warnings,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"SQL Error: {str(e)}")
    
    @router.get("/{name}/sql/tables")
    async def sql_list_tables(name: str):
        """List all available SQL tables and views."""
        from rdf_starbase.storage.duckdb import get_cached_interface, check_duckdb_available
        
        if not check_duckdb_available():
            raise HTTPException(
                status_code=501,
                detail="DuckDB not installed. Install with: pip install duckdb"
            )
        
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        sql = get_cached_interface(store, read_only=True)
        tables = sql.list_tables()
        return {
            "tables": [
                {
                    "name": t.name,
                    "columns": t.columns,
                    "row_count": t.row_count,
                }
                for t in tables
            ]
        }
    
    @router.get("/{name}/sql/schema/{table_name}")
    async def sql_table_schema(name: str, table_name: str):
        """Get the schema (column types) for a table."""
        from rdf_starbase.storage.duckdb import get_cached_interface, check_duckdb_available
        
        if not check_duckdb_available():
            raise HTTPException(
                status_code=501,
                detail="DuckDB not installed. Install with: pip install duckdb"
            )
        
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        try:
            sql = get_cached_interface(store, read_only=True)
            schema = sql.get_schema(table_name)
            return {
                "table": table_name,
                "columns": schema,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.get("/{name}/sql/sample/{table_name}")
    async def sql_sample_table(
        name: str,
        table_name: str,
        n: int = Query(default=10, ge=1, le=1000),
    ):
        """Get a sample of rows from a table."""
        from rdf_starbase.storage.duckdb import get_cached_interface, check_duckdb_available
        
        if not check_duckdb_available():
            raise HTTPException(
                status_code=501,
                detail="DuckDB not installed. Install with: pip install duckdb"
            )
        
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        try:
            sql = get_cached_interface(store, read_only=True)
            result = sql.sample(table_name, n)
            return {
                "table": table_name,
                "columns": result.columns,
                "rows": result.rows,
                "sample_size": result.row_count,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.post("/{name}/sql/aggregate")
    async def sql_aggregate(name: str, request: SQLAggregateRequest):
        """
        Run an aggregation query.
        
        Example request:
        {
            "group_by": "predicate",
            "aggregations": {
                "count": "COUNT(*)",
                "subjects": "COUNT(DISTINCT subject)"
            },
            "order_by": "count DESC",
            "limit": 10
        }
        """
        from rdf_starbase.storage.duckdb import get_cached_interface, check_duckdb_available
        
        if not check_duckdb_available():
            raise HTTPException(
                status_code=501,
                detail="DuckDB not installed. Install with: pip install duckdb"
            )
        
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        try:
            sql = get_cached_interface(store, read_only=True)
            result = sql.aggregate(
                group_by=request.group_by,
                aggregations=request.aggregations,
                table=request.table,
                where=request.where,
                order_by=request.order_by,
                limit=request.limit,
            )
            return {
                "columns": result.columns,
                "rows": result.rows,
                "row_count": result.row_count,
                "execution_time_ms": result.execution_time_ms,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # =========================================================================
    # Example Datasets
    # =========================================================================
    
    @router.get("/examples/datasets")
    async def list_example_datasets():
        """List available example datasets."""
        return {
            "datasets": [
                {
                    "id": "movies",
                    "name": "Movies & Directors",
                    "description": "Sample movie data with directors, actors, and genres. Great for learning SPARQL.",
                    "triple_count": 45,
                    "tags": ["movies", "entertainment", "schema.org"]
                },
                {
                    "id": "techcorp",
                    "name": "TechCorp Customer Service",
                    "description": "Customer service scenario with tickets, products, and customer data. Includes conflicting data from multiple sources.",
                    "triple_count": 35,
                    "tags": ["enterprise", "CRM", "support"]
                },
                {
                    "id": "knowledge_graph",
                    "name": "Simple Knowledge Graph",
                    "description": "Basic knowledge graph with people, organizations, and relationships.",
                    "triple_count": 28,
                    "tags": ["people", "organizations", "relationships"]
                },
                {
                    "id": "rdf_star_demo",
                    "name": "RDF-Star Demo",
                    "description": "Demonstrates RDF-Star features with quoted triples, annotations, and provenance metadata.",
                    "triple_count": 22,
                    "tags": ["rdf-star", "annotations", "provenance"]
                }
            ]
        }
    
    @router.post("/{name}/load-example/{dataset_id}")
    async def load_example_dataset(name: str, dataset_id: str):
        """Load an example dataset into a repository."""
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        datasets = {
            "movies": get_movies_dataset_triples,
            "techcorp": get_techcorp_dataset_triples,
            "knowledge_graph": get_knowledge_graph_dataset_triples,
            "rdf_star_demo": get_rdf_star_demo_dataset_triples,
        }
        
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset_id}")
        
        triples = datasets[dataset_id]()
        
        try:
            count = store.add_triples_batch(triples)
            manager.save(name)
            return {
                "success": True,
                "dataset": dataset_id,
                "message": f"Loaded example dataset '{dataset_id}' into repository '{name}'",
                "stats": store.stats()
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load dataset: {str(e)}")
    
    # =========================================================================
    # Repository AI Grounding Endpoints
    # =========================================================================
    
    @router.get("/{name}/ai/health")
    async def repository_ai_health(name: str):
        """AI Grounding health check for a specific repository."""
        try:
            store = manager.get_store(name)
            stats = store.stats()
            return {
                "status": "healthy",
                "api": "ai_grounding",
                "repository": name,
                "version": "1.0.0",
                "store_stats": {
                    "total_triples": stats.get("total_assertions", 0),
                    "unique_subjects": stats.get("unique_subjects", 0),
                },
                "capabilities": [
                    "query",
                    "verify",
                    "context",
                    "materialize",
                    "inferences",
                ],
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            return {"status": "degraded", "error": str(e)}
    
    @router.post("/{name}/ai/query")
    async def repository_ai_query(name: str, request: dict):
        """Query facts for AI grounding from a specific repository."""
        from datetime import datetime, timedelta
        import hashlib
        
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        retrieval_time = datetime.utcnow()
        
        # Parse confidence level
        min_confidence_str = request.get("min_confidence", "any")
        confidence_thresholds = {
            "high": 0.9, "medium": 0.7, "low": 0.5, "any": 0.0
        }
        confidence_threshold = confidence_thresholds.get(
            min_confidence_str, 
            float(min_confidence_str) if isinstance(min_confidence_str, (int, float)) else 0.0
        )
        
        # Get triples with filters
        df = store.get_triples(
            subject=request.get("subject"),
            predicate=request.get("predicate"),
            obj=request.get("object"),
            min_confidence=confidence_threshold,
        )
        
        total_count = len(df)
        
        # Apply source filter
        sources = request.get("sources", [])
        if sources:
            df = df.filter(pl.col("source").is_in(sources))
        
        # Apply freshness filter
        max_age_days = request.get("max_age_days")
        if max_age_days:
            cutoff = datetime.utcnow() - timedelta(days=max_age_days)
            if "timestamp" in df.columns:
                df = df.filter(pl.col("timestamp") >= cutoff)
        
        # Apply limit
        limit = request.get("limit", 100)
        df = df.head(limit)
        
        # Convert to grounded facts
        facts = []
        for row in df.iter_rows(named=True):
            fact_hash = f"{row['subject']}|{row['predicate']}|{row['object']}|{row.get('source', 'unknown')}"
            hash_id = hashlib.sha256(fact_hash.encode()).hexdigest()[:12]
            
            facts.append({
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object"],
                "citation": {
                    "fact_hash": hash_id,
                    "source": row.get("source", "unknown"),
                    "confidence": row.get("confidence", 1.0),
                    "timestamp": row.get("timestamp", retrieval_time).isoformat() if hasattr(row.get("timestamp", retrieval_time), "isoformat") else str(row.get("timestamp", "")),
                    "retrieval_time": retrieval_time.isoformat(),
                },
            })
        
        sources_used = df["source"].unique().to_list() if len(df) > 0 and "source" in df.columns else []
        
        return {
            "facts": facts,
            "total_count": total_count,
            "filtered_count": len(facts),
            "confidence_threshold": confidence_threshold,
            "retrieval_timestamp": retrieval_time.isoformat(),
            "sources_used": sources_used,
            "repository": name,
        }
    
    @router.post("/{name}/ai/verify")
    async def repository_ai_verify(name: str, request: dict):
        """Verify a claim against a specific repository's knowledge base."""
        from datetime import datetime
        import hashlib
        
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        retrieval_time = datetime.utcnow()
        
        subject = request.get("subject")
        predicate = request.get("predicate")
        expected_object = request.get("expected_object")
        
        min_confidence_str = request.get("min_confidence", "low")
        confidence_thresholds = {"high": 0.9, "medium": 0.7, "low": 0.5, "any": 0.0}
        confidence_threshold = confidence_thresholds.get(min_confidence_str, 0.5)
        
        # Get matching facts
        df = store.get_triples(
            subject=subject,
            predicate=predicate,
            min_confidence=confidence_threshold,
        )
        
        def df_to_facts(df_subset):
            facts = []
            for row in df_subset.iter_rows(named=True):
                fact_hash = f"{row['subject']}|{row['predicate']}|{row['object']}|{row.get('source', 'unknown')}"
                hash_id = hashlib.sha256(fact_hash.encode()).hexdigest()[:12]
                facts.append({
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "object": row["object"],
                    "citation": {
                        "fact_hash": hash_id,
                        "source": row.get("source", "unknown"),
                        "confidence": row.get("confidence", 1.0),
                        "retrieval_time": retrieval_time.isoformat(),
                    },
                })
            return facts
        
        if len(df) == 0:
            return {
                "claim_supported": False,
                "confidence": None,
                "supporting_facts": [],
                "contradicting_facts": [],
                "has_conflicts": False,
                "recommendation": "No facts found for this subject-predicate pair.",
                "repository": name,
            }
        
        all_facts = df_to_facts(df)
        
        if expected_object:
            supporting = [f for f in all_facts if str(f["object"]) == str(expected_object)]
            contradicting = [f for f in all_facts if str(f["object"]) != str(expected_object)]
            has_conflicts = len(supporting) > 0 and len(contradicting) > 0
            best_confidence = max((f["citation"]["confidence"] for f in supporting), default=None)
            
            return {
                "claim_supported": len(supporting) > 0,
                "confidence": best_confidence,
                "supporting_facts": supporting,
                "contradicting_facts": contradicting,
                "has_conflicts": has_conflicts,
                "recommendation": f"Claim {'supported' if supporting else 'not supported'} by knowledge base.",
                "repository": name,
            }
        else:
            unique_values = len(set(str(f["object"]) for f in all_facts))
            has_conflicts = unique_values > 1
            best_confidence = max((f["citation"]["confidence"] for f in all_facts), default=None)
            
            return {
                "claim_supported": True,
                "confidence": best_confidence,
                "supporting_facts": all_facts,
                "contradicting_facts": [],
                "has_conflicts": has_conflicts,
                "recommendation": f"Found {len(all_facts)} fact(s) with {unique_values} unique value(s).",
                "repository": name,
            }
    
    @router.get("/{name}/ai/context/{iri:path}")
    async def repository_ai_context(
        name: str,
        iri: str,
        min_confidence: str = Query("low", description="Minimum confidence: high, medium, low, any"),
        include_incoming: bool = Query(True, description="Include facts where entity is the object"),
        limit: int = Query(100, ge=1, le=500),
    ):
        """Get full context for an entity from a specific repository."""
        from datetime import datetime
        import urllib.parse
        import hashlib
        
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        entity = urllib.parse.unquote(iri)
        retrieval_time = datetime.utcnow()
        
        confidence_thresholds = {"high": 0.9, "medium": 0.7, "low": 0.5, "any": 0.0}
        confidence_threshold = confidence_thresholds.get(min_confidence, 0.5)
        
        # Get outgoing facts
        df_out = store.get_triples(subject=entity, min_confidence=confidence_threshold)
        
        # Get incoming facts if requested
        if include_incoming:
            df_in = store.get_triples(obj=entity, min_confidence=confidence_threshold)
            df = pl.concat([df_out, df_in]).unique()
        else:
            df = df_out
        
        df = df.head(limit)
        
        # Convert to facts
        facts = []
        related = set()
        for row in df.iter_rows(named=True):
            fact_hash = f"{row['subject']}|{row['predicate']}|{row['object']}|{row.get('source', 'unknown')}"
            hash_id = hashlib.sha256(fact_hash.encode()).hexdigest()[:12]
            
            facts.append({
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object"],
                "citation": {
                    "fact_hash": hash_id,
                    "source": row.get("source", "unknown"),
                    "confidence": row.get("confidence", 1.0),
                    "retrieval_time": retrieval_time.isoformat(),
                },
            })
            
            # Track related entities
            if row["subject"] != entity and row["subject"].startswith("http"):
                related.add(row["subject"])
            obj_str = str(row["object"])
            if obj_str != entity and obj_str.startswith("http"):
                related.add(obj_str)
        
        # Source and confidence summaries
        sources = list(set(f["citation"]["source"] for f in facts))
        confidences = [f["citation"]["confidence"] for f in facts]
        
        return {
            "entity": entity,
            "facts": facts,
            "related_entities": list(related)[:20],
            "sources": sources,
            "confidence_summary": {
                "min": min(confidences) if confidences else 0,
                "max": max(confidences) if confidences else 0,
                "avg": sum(confidences) / len(confidences) if confidences else 0,
            },
            "retrieval_timestamp": retrieval_time.isoformat(),
            "repository": name,
        }
    
    @router.get("/{name}/ai/inferences")
    async def repository_ai_inferences(
        name: str,
        limit: int = Query(100, ge=1, le=1000),
    ):
        """List materialized inferences from a specific repository."""
        from datetime import datetime
        import hashlib
        
        try:
            store = manager.get_store(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        retrieval_time = datetime.utcnow()
        df = store.get_triples()
        
        # Filter for inferred triples
        if "process" in df.columns:
            df = df.filter(
                (pl.col("process") == "reasoner") |
                (pl.col("process") == "inference_engine") |
                (pl.col("source") == "reasoner")
            )
        elif "source" in df.columns:
            df = df.filter(pl.col("source") == "reasoner")
        else:
            return {"count": 0, "inferences": [], "repository": name}
        
        df = df.head(limit)
        
        inferences = []
        for row in df.iter_rows(named=True):
            fact_hash = f"{row['subject']}|{row['predicate']}|{row['object']}|reasoner"
            hash_id = hashlib.sha256(fact_hash.encode()).hexdigest()[:12]
            inferences.append({
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object"],
                "citation": {
                    "fact_hash": hash_id,
                    "source": "reasoner",
                    "confidence": 1.0,
                    "retrieval_time": retrieval_time.isoformat(),
                },
            })
        
        return {
            "count": len(inferences),
            "inferences": inferences,
            "retrieval_timestamp": retrieval_time.isoformat(),
            "repository": name,
        }
    
    return router, manager


# =============================================================================
# Example Datasets with Provenance
# =============================================================================

def get_movies_dataset_triples() -> list[dict]:
    """Movies & Directors dataset with varied sources and confidence."""
    RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    SCHEMA = "http://schema.org/"
    EX = "http://example.org/"
    
    triples = []
    
    # Helper to add triple with provenance
    def add(s, p, o, source, confidence):
        triples.append({
            "subject": s,
            "predicate": p,
            "object": o,
            "source": source,
            "confidence": confidence
        })
    
    # Directors - from Wikipedia (high confidence)
    add(f"{EX}person/nolan", RDF_TYPE, f"{SCHEMA}Person", "Wikipedia", 0.98)
    add(f"{EX}person/nolan", f"{SCHEMA}name", "Christopher Nolan", "Wikipedia", 0.99)
    add(f"{EX}person/nolan", f"{SCHEMA}birthDate", "1970-07-30", "Wikipedia", 0.95)
    add(f"{EX}person/nolan", f"{SCHEMA}nationality", "British-American", "Wikipedia", 0.92)
    
    add(f"{EX}person/spielberg", RDF_TYPE, f"{SCHEMA}Person", "Wikipedia", 0.98)
    add(f"{EX}person/spielberg", f"{SCHEMA}name", "Steven Spielberg", "Wikipedia", 0.99)
    add(f"{EX}person/spielberg", f"{SCHEMA}birthDate", "1946-12-18", "Wikipedia", 0.97)
    
    add(f"{EX}person/greta", RDF_TYPE, f"{SCHEMA}Person", "IMDb", 0.94)
    add(f"{EX}person/greta", f"{SCHEMA}name", "Greta Gerwig", "IMDb", 0.96)
    add(f"{EX}person/greta", f"{SCHEMA}birthDate", "1983-08-04", "IMDb", 0.90)
    
    # Actors - from IMDb (good confidence)
    add(f"{EX}person/dicaprio", RDF_TYPE, f"{SCHEMA}Person", "IMDb", 0.97)
    add(f"{EX}person/dicaprio", f"{SCHEMA}name", "Leonardo DiCaprio", "IMDb", 0.99)
    add(f"{EX}person/dicaprio", f"{SCHEMA}birthDate", "1974-11-11", "IMDb", 0.95)
    
    add(f"{EX}person/cillian", RDF_TYPE, f"{SCHEMA}Person", "IMDb", 0.96)
    add(f"{EX}person/cillian", f"{SCHEMA}name", "Cillian Murphy", "IMDb", 0.98)
    
    add(f"{EX}person/margot", RDF_TYPE, f"{SCHEMA}Person", "IMDb", 0.97)
    add(f"{EX}person/margot", f"{SCHEMA}name", "Margot Robbie", "IMDb", 0.99)
    
    # Movies - mixed sources
    # Inception - from multiple sources
    add(f"{EX}movie/inception", RDF_TYPE, f"{SCHEMA}Movie", "IMDb", 0.99)
    add(f"{EX}movie/inception", f"{SCHEMA}name", "Inception", "IMDb", 0.99)
    add(f"{EX}movie/inception", f"{SCHEMA}datePublished", "2010", "IMDb", 0.98)
    add(f"{EX}movie/inception", f"{SCHEMA}director", f"{EX}person/nolan", "IMDb", 0.99)
    add(f"{EX}movie/inception", f"{SCHEMA}actor", f"{EX}person/dicaprio", "IMDb", 0.99)
    add(f"{EX}movie/inception", f"{SCHEMA}genre", "Sci-Fi", "RottenTomatoes", 0.85)
    add(f"{EX}movie/inception", f"{SCHEMA}duration", "PT2H28M", "IMDb", 0.97)
    
    # Oppenheimer - recent film, high confidence
    add(f"{EX}movie/oppenheimer", RDF_TYPE, f"{SCHEMA}Movie", "IMDb", 0.99)
    add(f"{EX}movie/oppenheimer", f"{SCHEMA}name", "Oppenheimer", "IMDb", 0.99)
    add(f"{EX}movie/oppenheimer", f"{SCHEMA}datePublished", "2023", "BoxOfficeMojo", 0.98)
    add(f"{EX}movie/oppenheimer", f"{SCHEMA}director", f"{EX}person/nolan", "Wikipedia", 0.99)
    add(f"{EX}movie/oppenheimer", f"{SCHEMA}actor", f"{EX}person/cillian", "IMDb", 0.98)
    add(f"{EX}movie/oppenheimer", f"{SCHEMA}genre", "Drama", "RottenTomatoes", 0.88)
    add(f"{EX}movie/oppenheimer", f"{SCHEMA}genre", "Biography", "Wikipedia", 0.82)  # Multiple genres!
    
    # Interstellar
    add(f"{EX}movie/interstellar", RDF_TYPE, f"{SCHEMA}Movie", "IMDb", 0.99)
    add(f"{EX}movie/interstellar", f"{SCHEMA}name", "Interstellar", "IMDb", 0.99)
    add(f"{EX}movie/interstellar", f"{SCHEMA}datePublished", "2014", "IMDb", 0.98)
    add(f"{EX}movie/interstellar", f"{SCHEMA}director", f"{EX}person/nolan", "Wikipedia", 0.99)
    add(f"{EX}movie/interstellar", f"{SCHEMA}genre", "Sci-Fi", "IMDb", 0.92)
    
    # Jurassic Park - classic film
    add(f"{EX}movie/jurassic_park", RDF_TYPE, f"{SCHEMA}Movie", "Wikipedia", 0.99)
    add(f"{EX}movie/jurassic_park", f"{SCHEMA}name", "Jurassic Park", "Wikipedia", 0.99)
    add(f"{EX}movie/jurassic_park", f"{SCHEMA}datePublished", "1993", "Wikipedia", 0.99)
    add(f"{EX}movie/jurassic_park", f"{SCHEMA}director", f"{EX}person/spielberg", "Wikipedia", 0.99)
    add(f"{EX}movie/jurassic_park", f"{SCHEMA}genre", "Adventure", "IMDb", 0.90)
    
    # Barbie - from entertainment news (slightly lower confidence)
    add(f"{EX}movie/barbie", RDF_TYPE, f"{SCHEMA}Movie", "BoxOfficeMojo", 0.97)
    add(f"{EX}movie/barbie", f"{SCHEMA}name", "Barbie", "BoxOfficeMojo", 0.99)
    add(f"{EX}movie/barbie", f"{SCHEMA}datePublished", "2023", "BoxOfficeMojo", 0.98)
    add(f"{EX}movie/barbie", f"{SCHEMA}director", f"{EX}person/greta", "IMDb", 0.97)
    add(f"{EX}movie/barbie", f"{SCHEMA}actor", f"{EX}person/margot", "IMDb", 0.98)
    add(f"{EX}movie/barbie", f"{SCHEMA}genre", "Comedy", "RottenTomatoes", 0.80)
    
    return triples


def get_techcorp_dataset_triples() -> list[dict]:
    """TechCorp Customer Service dataset with conflicting data from multiple sources."""
    RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    TC = "http://techcorp.com/"
    FOAF = "http://xmlns.com/foaf/0.1/"
    RDFS = "http://www.w3.org/2000/01/rdf-schema#"
    
    triples = []
    
    def add(s, p, o, source, confidence):
        triples.append({
            "subject": s,
            "predicate": p,
            "object": o,
            "source": source,
            "confidence": confidence
        })
    
    # Customer C001 - Alice Johnson (CONFLICTING DATA from different systems!)
    add(f"{TC}customer/C001", RDF_TYPE, f"{TC}Customer", "CRM_System", 0.99)
    add(f"{TC}customer/C001", f"{FOAF}name", "Alice Johnson", "CRM_System", 0.98)
    add(f"{TC}customer/C001", f"{TC}email", "alice@example.com", "CRM_System", 0.95)
    add(f"{TC}customer/C001", f"{TC}tier", "Premium", "CRM_System", 0.90)
    add(f"{TC}customer/C001", f"{TC}since", "2020-01-15", "CRM_System", 0.99)
    
    # CONFLICT: Billing system says different tier!
    add(f"{TC}customer/C001", f"{TC}tier", "Enterprise", "Billing_System", 0.85)
    # CONFLICT: Support portal has different email
    add(f"{TC}customer/C001", f"{TC}email", "a.johnson@corp.example.com", "Support_Portal", 0.75)
    
    # Customer C002 - Bob Smith
    add(f"{TC}customer/C002", RDF_TYPE, f"{TC}Customer", "CRM_System", 0.99)
    add(f"{TC}customer/C002", f"{FOAF}name", "Bob Smith", "CRM_System", 0.97)
    add(f"{TC}customer/C002", f"{TC}email", "bob@example.com", "CRM_System", 0.96)
    add(f"{TC}customer/C002", f"{TC}tier", "Standard", "CRM_System", 0.92)
    add(f"{TC}customer/C002", f"{TC}since", "2021-06-20", "CRM_System", 0.99)
    
    # Customer C003 - Carol White (from legacy system)
    add(f"{TC}customer/C003", RDF_TYPE, f"{TC}Customer", "Legacy_DB", 0.88)
    add(f"{TC}customer/C003", f"{FOAF}name", "Carol White", "Legacy_DB", 0.85)
    add(f"{TC}customer/C003", f"{TC}email", "carol@example.com", "Legacy_DB", 0.80)
    add(f"{TC}customer/C003", f"{TC}tier", "Enterprise", "Billing_System", 0.95)
    
    # Products - from Product Catalog (high confidence)
    add(f"{TC}product/P001", RDF_TYPE, f"{TC}Product", "Product_Catalog", 0.99)
    add(f"{TC}product/P001", f"{RDFS}label", "CloudSync Pro", "Product_Catalog", 0.99)
    add(f"{TC}product/P001", f"{TC}category", "Software", "Product_Catalog", 0.98)
    add(f"{TC}product/P001", f"{TC}price", "299.99", "Product_Catalog", 0.97)
    # CONFLICT: Sales team has different price!
    add(f"{TC}product/P001", f"{TC}price", "279.99", "Sales_Team", 0.70)
    
    add(f"{TC}product/P002", RDF_TYPE, f"{TC}Product", "Product_Catalog", 0.99)
    add(f"{TC}product/P002", f"{RDFS}label", "DataVault", "Product_Catalog", 0.99)
    add(f"{TC}product/P002", f"{TC}category", "Storage", "Product_Catalog", 0.98)
    add(f"{TC}product/P002", f"{TC}price", "499.99", "Product_Catalog", 0.97)
    
    add(f"{TC}product/P003", RDF_TYPE, f"{TC}Product", "Product_Catalog", 0.99)
    add(f"{TC}product/P003", f"{RDFS}label", "SecureNet", "Product_Catalog", 0.99)
    add(f"{TC}product/P003", f"{TC}category", "Security", "Product_Catalog", 0.98)
    add(f"{TC}product/P003", f"{TC}price", "199.99", "Product_Catalog", 0.97)
    
    # Support Tickets - from different support channels
    add(f"{TC}ticket/T001", RDF_TYPE, f"{TC}SupportTicket", "Support_Portal", 0.99)
    add(f"{TC}ticket/T001", f"{TC}customer", f"{TC}customer/C001", "Support_Portal", 0.99)
    add(f"{TC}ticket/T001", f"{TC}product", f"{TC}product/P001", "Support_Portal", 0.98)
    add(f"{TC}ticket/T001", f"{TC}status", "Open", "Support_Portal", 0.95)
    add(f"{TC}ticket/T001", f"{TC}priority", "High", "Support_Portal", 0.90)
    add(f"{TC}ticket/T001", f"{TC}description", "Sync failing intermittently", "Support_Portal", 0.99)
    
    add(f"{TC}ticket/T002", RDF_TYPE, f"{TC}SupportTicket", "Email_Integration", 0.95)
    add(f"{TC}ticket/T002", f"{TC}customer", f"{TC}customer/C002", "Email_Integration", 0.92)
    add(f"{TC}ticket/T002", f"{TC}product", f"{TC}product/P002", "Email_Integration", 0.90)
    add(f"{TC}ticket/T002", f"{TC}status", "Resolved", "Support_Portal", 0.98)
    add(f"{TC}ticket/T002", f"{TC}priority", "Medium", "Email_Integration", 0.85)
    add(f"{TC}ticket/T002", f"{TC}description", "Storage quota question", "Email_Integration", 0.88)
    
    add(f"{TC}ticket/T003", RDF_TYPE, f"{TC}SupportTicket", "Security_Ops", 0.99)
    add(f"{TC}ticket/T003", f"{TC}customer", f"{TC}customer/C003", "Security_Ops", 0.98)
    add(f"{TC}ticket/T003", f"{TC}product", f"{TC}product/P003", "Security_Ops", 0.99)
    add(f"{TC}ticket/T003", f"{TC}status", "Open", "Security_Ops", 0.99)
    add(f"{TC}ticket/T003", f"{TC}priority", "Critical", "Security_Ops", 0.99)
    add(f"{TC}ticket/T003", f"{TC}description", "Security alert investigation", "Security_Ops", 0.99)
    
    return triples


def get_knowledge_graph_dataset_triples() -> list[dict]:
    """Simple Knowledge Graph with people and organizations."""
    RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    FOAF = "http://xmlns.com/foaf/0.1/"
    ORG = "http://www.w3.org/ns/org#"
    EX = "http://example.org/"
    
    triples = []
    
    def add(s, p, o, source, confidence):
        triples.append({
            "subject": s,
            "predicate": p,
            "object": o,
            "source": source,
            "confidence": confidence
        })
    
    # People - from LinkedIn (professional network)
    add(f"{EX}person/jane", RDF_TYPE, f"{FOAF}Person", "LinkedIn", 0.95)
    add(f"{EX}person/jane", f"{FOAF}name", "Jane Doe", "LinkedIn", 0.98)
    add(f"{EX}person/jane", f"{FOAF}age", "32", "LinkedIn", 0.75)  # Age less reliable
    add(f"{EX}person/jane", f"{FOAF}knows", f"{EX}person/john", "LinkedIn", 0.90)
    add(f"{EX}person/jane", f"{FOAF}knows", f"{EX}person/alice", "LinkedIn", 0.92)
    add(f"{EX}person/jane", f"{ORG}memberOf", f"{EX}org/acme", "LinkedIn", 0.97)
    
    add(f"{EX}person/john", RDF_TYPE, f"{FOAF}Person", "LinkedIn", 0.95)
    add(f"{EX}person/john", f"{FOAF}name", "John Smith", "LinkedIn", 0.97)
    add(f"{EX}person/john", f"{FOAF}age", "28", "Facebook", 0.70)  # Different source
    add(f"{EX}person/john", f"{FOAF}knows", f"{EX}person/jane", "LinkedIn", 0.90)
    add(f"{EX}person/john", f"{ORG}memberOf", f"{EX}org/globex", "CompanyWebsite", 0.99)
    
    add(f"{EX}person/alice", RDF_TYPE, f"{FOAF}Person", "LinkedIn", 0.96)
    add(f"{EX}person/alice", f"{FOAF}name", "Alice Chen", "LinkedIn", 0.98)
    add(f"{EX}person/alice", f"{FOAF}age", "35", "LinkedIn", 0.72)
    add(f"{EX}person/alice", f"{FOAF}knows", f"{EX}person/jane", "LinkedIn", 0.92)
    add(f"{EX}person/alice", f"{FOAF}knows", f"{EX}person/bob", "Email_Analysis", 0.65)
    add(f"{EX}person/alice", f"{ORG}headOf", f"{EX}org/acme", "CompanyWebsite", 0.99)
    
    add(f"{EX}person/bob", RDF_TYPE, f"{FOAF}Person", "HR_System", 0.98)
    add(f"{EX}person/bob", f"{FOAF}name", "Bob Williams", "HR_System", 0.99)
    add(f"{EX}person/bob", f"{FOAF}age", "42", "HR_System", 0.95)
    add(f"{EX}person/bob", f"{ORG}memberOf", f"{EX}org/initech", "HR_System", 0.98)
    
    # Organizations - from company registries
    add(f"{EX}org/acme", RDF_TYPE, f"{ORG}Organization", "SEC_Filings", 0.99)
    add(f"{EX}org/acme", f"{FOAF}name", "Acme Corp", "SEC_Filings", 0.99)
    add(f"{EX}org/acme", f"{ORG}hasSite", f"{EX}location/sf", "CompanyWebsite", 0.95)
    add(f"{EX}org/acme", f"{EX}industry", "Technology", "CrunchBase", 0.88)
    
    add(f"{EX}org/globex", RDF_TYPE, f"{ORG}Organization", "SEC_Filings", 0.99)
    add(f"{EX}org/globex", f"{FOAF}name", "Globex Inc", "SEC_Filings", 0.99)
    add(f"{EX}org/globex", f"{ORG}hasSite", f"{EX}location/nyc", "CompanyWebsite", 0.96)
    add(f"{EX}org/globex", f"{EX}industry", "Finance", "Bloomberg", 0.95)
    
    add(f"{EX}org/initech", RDF_TYPE, f"{ORG}Organization", "State_Registry", 0.97)
    add(f"{EX}org/initech", f"{FOAF}name", "Initech", "State_Registry", 0.98)
    add(f"{EX}org/initech", f"{ORG}hasSite", f"{EX}location/austin", "GoogleMaps", 0.85)
    add(f"{EX}org/initech", f"{EX}industry", "Software", "LinkedIn", 0.80)
    
    # Locations - from geographic databases
    add(f"{EX}location/sf", RDF_TYPE, f"{EX}Location", "GeoNames", 0.99)
    add(f"{EX}location/sf", f"{FOAF}name", "San Francisco", "GeoNames", 0.99)
    add(f"{EX}location/sf", f"{EX}country", "USA", "GeoNames", 0.99)
    
    add(f"{EX}location/nyc", RDF_TYPE, f"{EX}Location", "GeoNames", 0.99)
    add(f"{EX}location/nyc", f"{FOAF}name", "New York City", "GeoNames", 0.99)
    add(f"{EX}location/nyc", f"{EX}country", "USA", "GeoNames", 0.99)
    
    add(f"{EX}location/austin", RDF_TYPE, f"{EX}Location", "GeoNames", 0.99)
    add(f"{EX}location/austin", f"{FOAF}name", "Austin", "GeoNames", 0.99)
    add(f"{EX}location/austin", f"{EX}country", "USA", "GeoNames", 0.99)
    
    return triples


def get_rdf_star_demo_dataset_triples() -> list[dict]:
    """RDF-Star Demo showing statement-level metadata."""
    RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    EX = "http://example.org/"
    
    triples = []
    
    def add(s, p, o, source, confidence):
        triples.append({
            "subject": s,
            "predicate": p,
            "object": o,
            "source": source,
            "confidence": confidence
        })
    
    # Employee data from HR System (high confidence)
    add(f"{EX}alice", f"{EX}name", "Alice", "HR_System", 0.99)
    add(f"{EX}alice", f"{EX}worksAt", f"{EX}acme", "HR_System", 0.98)
    add(f"{EX}alice", f"{EX}department", "Engineering", "HR_System", 0.97)
    add(f"{EX}alice", f"{EX}startDate", "2020-03-15", "HR_System", 0.99)
    
    # Salary from Payroll (very high confidence, sensitive)
    add(f"{EX}alice", f"{EX}salary", "95000", "Payroll_System", 0.999)
    
    # Manager relationship
    add(f"{EX}bob", f"{EX}name", "Bob", "HR_System", 0.99)
    add(f"{EX}bob", f"{EX}worksAt", f"{EX}acme", "HR_System", 0.98)
    add(f"{EX}bob", f"{EX}manages", f"{EX}alice", "HR_System", 0.95)
    add(f"{EX}bob", f"{EX}title", "Engineering Manager", "HR_System", 0.97)
    
    # Company info from different sources
    add(f"{EX}acme", f"{EX}name", "Acme Corporation", "SEC_Filings", 0.99)
    add(f"{EX}acme", f"{EX}founded", "2010", "CrunchBase", 0.92)
    add(f"{EX}acme", f"{EX}headquarters", "San Francisco", "CompanyWebsite", 0.95)
    add(f"{EX}acme", f"{EX}employees", "500", "LinkedIn", 0.75)  # Less reliable
    add(f"{EX}acme", f"{EX}employees", "487", "SEC_Filings", 0.98)  # Conflicting!
    
    # Performance reviews (varying confidence)
    add(f"{EX}alice", f"{EX}performanceRating", "Exceeds Expectations", "Performance_System", 0.90)
    add(f"{EX}alice", f"{EX}lastReview", "2024-12-01", "Performance_System", 0.99)
    
    # Skills from different sources
    add(f"{EX}alice", f"{EX}skill", "Python", "LinkedIn", 0.85)
    add(f"{EX}alice", f"{EX}skill", "Machine Learning", "LinkedIn", 0.80)
    add(f"{EX}alice", f"{EX}skill", "Distributed Systems", "Manager_Assessment", 0.92)
    
    # Project assignments
    add(f"{EX}project/alpha", f"{EX}name", "Project Alpha", "Jira", 0.99)
    add(f"{EX}project/alpha", f"{EX}status", "Active", "Jira", 0.98)
    add(f"{EX}project/alpha", f"{EX}lead", f"{EX}alice", "Jira", 0.97)
    add(f"{EX}alice", f"{EX}assignedTo", f"{EX}project/alpha", "Jira", 0.96)
    add(f"{EX}bob", f"{EX}sponsors", f"{EX}project/alpha", "Executive_Dashboard", 0.88)
    
    return triples


# Keep old SPARQL functions for backwards compatibility (unused)
def get_movies_dataset() -> str:
    """Deprecated: Use get_movies_dataset_triples instead."""
    return ""

def get_techcorp_dataset() -> str:
    """Deprecated: Use get_techcorp_dataset_triples instead."""
    return ""

def get_knowledge_graph_dataset() -> str:
    """Deprecated: Use get_knowledge_graph_dataset_triples instead."""
    return ""

def get_rdf_star_demo_dataset() -> str:
    """Deprecated: Use get_rdf_star_demo_dataset_triples instead."""
    return ""
