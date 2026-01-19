"""
rdflib Compatibility Module.

Provides drop-in replacements for rdflib classes backed by RDF-StarBase.
"""

from rdf_starbase.compat.rdflib import (
    # Term classes
    URIRef, Literal, BNode, Identifier,
    # Namespace
    Namespace, ClosedNamespace, NamespaceManager,
    # Well-known namespaces
    RDF, RDFS, OWL, XSD, FOAF, DC, DCTERMS, SKOS, PROV,
    # Graph
    Graph,
    # Query
    QueryResult, QueryRow,
)

__all__ = [
    'URIRef', 'Literal', 'BNode', 'Identifier',
    'Namespace', 'ClosedNamespace', 'NamespaceManager',
    'RDF', 'RDFS', 'OWL', 'XSD', 'FOAF', 'DC', 'DCTERMS', 'SKOS', 'PROV',
    'Graph',
    'QueryResult', 'QueryRow',
]
