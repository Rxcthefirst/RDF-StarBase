"""
SPARQL-Star query language support for RDF-StarBase.

Implements parsing and execution of SPARQL-Star queries following
the W3C SPARQL-Star Community Group specification.
"""

from rdf_starbase.sparql.parser import SPARQLStarParser, parse_query
from rdf_starbase.sparql.ast import (
    Query,
    SelectQuery,
    AskQuery,
    TriplePattern,
    QuotedTriplePattern,
    Variable,
    IRI,
    Literal,
    Filter,
)
from rdf_starbase.sparql.executor import SPARQLExecutor

__all__ = [
    "SPARQLStarParser",
    "SPARQLExecutor",
    "parse_query",
    "Query",
    "SelectQuery",
    "AskQuery",
    "TriplePattern",
    "QuotedTriplePattern",
    "Variable",
    "IRI",
    "Literal",
    "Filter",
]
