"""
RDF Format Parsers and Serializers.

Supports:
- Turtle (.ttl) with Turtle-Star extensions
- N-Triples (.nt) with N-Triples-Star extensions
- N-Quads (.nq) with named graphs
- N3 / Notation3 (.n3)
- TriG (.trig) with TriG-Star extensions
- TriX (.trix) XML format
- JSON-LD (.jsonld)
- NDJSON-LD (.ndjsonld) streaming format
- RDF/JSON (.json) W3C RDF JSON format
- RDF/XML (.rdf, .xml)
- Binary RDF (.brf) compact binary format
"""

from rdf_starbase.formats.turtle import TurtleParser, TurtleSerializer, parse_turtle, serialize_turtle, Triple, ParsedDocument
from rdf_starbase.formats.ntriples import NTriplesParser, NTriplesSerializer, parse_ntriples, serialize_ntriples
from rdf_starbase.formats.nquads import NQuadsParser, NQuadsSerializer, parse_nquads, parse_nquads_as_triples
from rdf_starbase.formats.n3 import N3Parser, N3Serializer, parse_n3, serialize_n3
from rdf_starbase.formats.trig import TriGParser, TriGSerializer, parse_trig, parse_trig_as_document
from rdf_starbase.formats.trix import TriXParser, TriXSerializer, parse_trix, parse_trix_as_document
from rdf_starbase.formats.jsonld import JSONLDParser, JSONLDSerializer, parse_jsonld, serialize_jsonld
from rdf_starbase.formats.ndjsonld import NDJSONLDParser, NDJSONLDSerializer, parse_ndjsonld, parse_ndjsonld_as_document
from rdf_starbase.formats.rdfjson import RDFJSONParser, RDFJSONSerializer, parse_rdfjson, parse_rdfjson_as_document
from rdf_starbase.formats.rdfxml import RDFXMLParser, RDFXMLSerializer, parse_rdfxml, serialize_rdfxml
from rdf_starbase.formats.binaryrdf import BinaryRDFParser, BinaryRDFSerializer, parse_binaryrdf, parse_binaryrdf_as_document, serialize_binaryrdf

__all__ = [
    # Core types
    "Triple",
    "ParsedDocument",
    # Turtle
    "TurtleParser",
    "TurtleSerializer",
    "parse_turtle",
    "serialize_turtle",
    # N-Triples
    "NTriplesParser",
    "NTriplesSerializer",
    "parse_ntriples",
    "serialize_ntriples",
    # N-Quads
    "NQuadsParser",
    "NQuadsSerializer",
    "parse_nquads",
    "parse_nquads_as_triples",
    # N3
    "N3Parser",
    "N3Serializer",
    "parse_n3",
    "serialize_n3",
    # TriG
    "TriGParser",
    "TriGSerializer",
    "parse_trig",
    "parse_trig_as_document",
    # TriX
    "TriXParser",
    "TriXSerializer",
    "parse_trix",
    "parse_trix_as_document",
    # JSON-LD
    "JSONLDParser",
    "JSONLDSerializer",
    "parse_jsonld",
    "serialize_jsonld",
    # NDJSON-LD
    "NDJSONLDParser",
    "NDJSONLDSerializer",
    "parse_ndjsonld",
    "parse_ndjsonld_as_document",
    # RDF/JSON
    "RDFJSONParser",
    "RDFJSONSerializer",
    "parse_rdfjson",
    "parse_rdfjson_as_document",
    # RDF/XML
    "RDFXMLParser",
    "RDFXMLSerializer",
    "parse_rdfxml",
    "serialize_rdfxml",
    # Binary RDF
    "BinaryRDFParser",
    "BinaryRDFSerializer",
    "parse_binaryrdf",
    "parse_binaryrdf_as_document",
    "serialize_binaryrdf",
]
