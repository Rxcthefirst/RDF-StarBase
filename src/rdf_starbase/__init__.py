"""
RDF-StarBase: A blazingly fast RDF-Star database powered by Polars.

Native RDF★ storage for assertions with provenance, trust, and temporal context.
"""

__version__ = "0.1.0"

from rdf_starbase.store import TripleStore
from rdf_starbase.models import Triple, QuotedTriple, ProvenanceContext
from rdf_starbase.sparql import parse_query, SPARQLExecutor
from rdf_starbase.sparql.executor import execute_sparql
from rdf_starbase.registry import (
    AssertionRegistry,
    RegisteredSource,
    SyncRun,
    SourceType,
    SourceStatus,
)

__all__ = [
    "TripleStore",
    "Triple",
    "QuotedTriple",
    "ProvenanceContext",
    "parse_query",
    "SPARQLExecutor",
    "execute_sparql",
    "AssertionRegistry",
    "RegisteredSource",
    "SyncRun",
    "SourceType",
    "SourceStatus",
]
