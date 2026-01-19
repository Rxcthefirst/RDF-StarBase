"""
JSON-LD parser and serializer for RDF-StarBase.

JSON-LD (JSON for Linked Data) is a W3C standard for embedding RDF
data in JSON. It's web-friendly and widely used for structured data
on the web (Schema.org, etc.).

Key features:
- @context: Defines prefixes and term mappings
- @id: Identifies resources (IRIs)
- @type: Specifies rdf:type
- @value/@language/@type: Literal representation
- @graph: Named graphs
- Quoted triples via annotation syntax for RDF-Star

Example JSON-LD:
{
  "@context": {
    "foaf": "http://xmlns.com/foaf/0.1/",
    "name": "foaf:name",
    "knows": {"@id": "foaf:knows", "@type": "@id"}
  },
  "@id": "http://example.org/alice",
  "@type": "foaf:Person",
  "name": "Alice",
  "knows": "http://example.org/bob"
}
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Union
import re


@dataclass
class Triple:
    """A simple triple representation for parsing."""
    subject: str
    predicate: str
    object: str
    subject_triple: Optional["Triple"] = None
    object_triple: Optional["Triple"] = None


@dataclass
class JSONLDDocument:
    """Parsed JSON-LD document."""
    context: dict[str, Any] = field(default_factory=dict)
    triples: list[Triple] = field(default_factory=list)
    base: Optional[str] = None


# Standard prefixes
STANDARD_PREFIXES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "schema": "http://schema.org/",
}


class JSONLDParser:
    """
    Parser for JSON-LD documents.
    
    Supports:
    - @context with prefix definitions
    - @id for resource identification
    - @type for rdf:type
    - @value, @language, @type for literals
    - @list for RDF lists
    - Nested objects
    - RDF-Star annotation syntax (experimental)
    """
    
    def __init__(self):
        self.context = {}
        self.base = None
        self.triples = []
        self.blank_counter = 0
    
    def parse(self, source: Union[str, dict]) -> JSONLDDocument:
        """Parse JSON-LD from string or dict."""
        if isinstance(source, str):
            data = json.loads(source)
        else:
            data = source
        
        self.context = {}
        self.base = None
        self.triples = []
        self.blank_counter = 0
        
        # Process @context
        if "@context" in data:
            self._process_context(data["@context"])
        
        # Process @base
        if "@base" in data:
            self.base = data["@base"]
        
        # Process the document
        if "@graph" in data:
            # Multiple resources in @graph
            for item in data["@graph"]:
                self._process_node(item)
        else:
            # Single resource at top level
            self._process_node(data)
        
        return JSONLDDocument(
            context=self.context.copy(),
            triples=self.triples.copy(),
            base=self.base
        )
    
    def _process_context(self, ctx: Union[str, dict, list]):
        """Process @context to extract prefixes and term mappings."""
        if isinstance(ctx, str):
            # Remote context URL - not fully supported yet
            return
        
        if isinstance(ctx, list):
            for item in ctx:
                self._process_context(item)
            return
        
        if isinstance(ctx, dict):
            for key, value in ctx.items():
                if key.startswith("@"):
                    continue  # Skip keywords
                
                if isinstance(value, str):
                    # Simple prefix: "foaf": "http://..."
                    self.context[key] = {"@id": value}
                elif isinstance(value, dict):
                    # Complex term: "name": {"@id": "foaf:name", "@type": "@id"}
                    self.context[key] = value
    
    def _expand_iri(self, value: str) -> str:
        """Expand a compact IRI or term to full IRI."""
        if not value or value.startswith("@"):
            return value
        
        # Already a full IRI
        if value.startswith("http://") or value.startswith("https://") or value.startswith("urn:"):
            return value
        
        # Check for term mapping first (e.g., "name" -> "foaf:name")
        if value in self.context:
            term = self.context[value]
            if isinstance(term, dict):
                expanded = term.get("@id", value)
            else:
                expanded = str(term)
            # Recursively expand in case it's a prefixed name
            if expanded != value:
                return self._expand_iri(expanded)
            return expanded
        
        # Check for prefix:localName
        if ":" in value:
            prefix, local = value.split(":", 1)
            if prefix in self.context:
                term = self.context[prefix]
                if isinstance(term, dict):
                    base = term.get("@id", "")
                else:
                    base = str(term)
                return base + local
            if prefix in STANDARD_PREFIXES:
                return STANDARD_PREFIXES[prefix] + local
        
        # Apply base if available
        if self.base and not value.startswith("_:"):
            return self.base + value
        
        return value
    
    def _gen_blank_node(self) -> str:
        """Generate a new blank node ID."""
        self.blank_counter += 1
        return f"_:b{self.blank_counter}"
    
    def _process_node(self, node: dict, subject: Optional[str] = None) -> str:
        """Process a JSON-LD node and extract triples."""
        if not isinstance(node, dict):
            return str(node)
        
        # Get or generate subject
        if "@id" in node:
            subject = self._expand_iri(node["@id"])
        elif subject is None:
            subject = self._gen_blank_node()
        
        for key, value in node.items():
            if key.startswith("@"):
                if key == "@type":
                    # Handle @type
                    self._process_type(subject, value)
                # Skip other keywords
                continue
            
            # Get predicate
            predicate = self._expand_iri(key)
            
            # Get term definition for type coercion
            term_def = self.context.get(key, {})
            if isinstance(term_def, str):
                term_def = {"@id": term_def}
            
            # Process value(s)
            if isinstance(value, list):
                for item in value:
                    self._process_value(subject, predicate, item, term_def)
            else:
                self._process_value(subject, predicate, value, term_def)
        
        return subject
    
    def _process_type(self, subject: str, type_value: Union[str, list]):
        """Process @type values."""
        rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        
        if isinstance(type_value, list):
            for t in type_value:
                type_iri = self._expand_iri(t)
                self.triples.append(Triple(subject, rdf_type, type_iri))
        else:
            type_iri = self._expand_iri(type_value)
            self.triples.append(Triple(subject, rdf_type, type_iri))
    
    def _process_value(self, subject: str, predicate: str, value: Any, term_def: dict):
        """Process a property value and create triples."""
        type_coercion = term_def.get("@type")
        
        if isinstance(value, dict):
            # Check for value object
            if "@value" in value:
                obj = self._make_literal(value)
            elif "@list" in value:
                obj = self._process_list(value["@list"])
            elif "@set" in value:
                # @set is just syntactic sugar - process items
                for item in value["@set"]:
                    self._process_value(subject, predicate, item, term_def)
                return
            else:
                # Nested node
                obj = self._process_node(value)
        elif isinstance(value, bool):
            obj = f'"{str(value).lower()}"^^<http://www.w3.org/2001/XMLSchema#boolean>'
        elif isinstance(value, int):
            obj = f'"{value}"^^<http://www.w3.org/2001/XMLSchema#integer>'
        elif isinstance(value, float):
            obj = f'"{value}"^^<http://www.w3.org/2001/XMLSchema#double>'
        elif type_coercion == "@id":
            # IRI reference
            obj = self._expand_iri(value)
        else:
            # String literal
            obj = f'"{self._escape_string(value)}"'
        
        self.triples.append(Triple(subject, predicate, obj))
    
    def _make_literal(self, value_obj: dict) -> str:
        """Create a literal from a value object."""
        val = value_obj["@value"]
        escaped = self._escape_string(str(val))
        
        if "@language" in value_obj:
            return f'"{escaped}"@{value_obj["@language"]}'
        elif "@type" in value_obj:
            datatype = self._expand_iri(value_obj["@type"])
            return f'"{escaped}"^^<{datatype}>'
        else:
            return f'"{escaped}"'
    
    def _process_list(self, items: list) -> str:
        """Process an @list and create RDF list structure."""
        rdf_first = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first"
        rdf_rest = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest"
        rdf_nil = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"
        
        if not items:
            return rdf_nil
        
        head = self._gen_blank_node()
        current = head
        
        for i, item in enumerate(items):
            # Process the item
            if isinstance(item, dict):
                if "@value" in item:
                    item_value = self._make_literal(item)
                else:
                    item_value = self._process_node(item)
            elif isinstance(item, str):
                item_value = f'"{self._escape_string(item)}"'
            else:
                item_value = str(item)
            
            self.triples.append(Triple(current, rdf_first, item_value))
            
            if i < len(items) - 1:
                next_node = self._gen_blank_node()
                self.triples.append(Triple(current, rdf_rest, next_node))
                current = next_node
            else:
                self.triples.append(Triple(current, rdf_rest, rdf_nil))
        
        return head
    
    def _escape_string(self, s: str) -> str:
        """Escape special characters in a string."""
        s = s.replace("\\", "\\\\")
        s = s.replace('"', '\\"')
        s = s.replace("\n", "\\n")
        s = s.replace("\r", "\\r")
        s = s.replace("\t", "\\t")
        return s


class JSONLDSerializer:
    """
    Serializer for JSON-LD output.
    
    Converts triples to compact JSON-LD format.
    """
    
    def __init__(self, context: Optional[dict] = None):
        self.context = context or {}
        self.inverse_context = {}
        self._build_inverse_context()
    
    def _build_inverse_context(self):
        """Build inverse mappings for compaction."""
        for term, value in self.context.items():
            if isinstance(value, str):
                self.inverse_context[value] = term
            elif isinstance(value, dict) and "@id" in value:
                self.inverse_context[value["@id"]] = term
    
    def serialize(self, triples: list[Triple], pretty: bool = True) -> str:
        """Serialize triples to JSON-LD."""
        # Group triples by subject
        subjects = {}
        for triple in triples:
            if triple.subject not in subjects:
                subjects[triple.subject] = []
            subjects[triple.subject].append(triple)
        
        # Build JSON-LD nodes
        nodes = []
        for subject, subject_triples in subjects.items():
            node = self._build_node(subject, subject_triples)
            nodes.append(node)
        
        # Build result
        if len(nodes) == 1 and not self.context:
            result = nodes[0]
        elif len(nodes) == 1:
            result = {"@context": self._build_context(), **nodes[0]}
        else:
            result = {
                "@context": self._build_context(),
                "@graph": nodes
            }
        
        if pretty:
            return json.dumps(result, indent=2, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)
    
    def _build_context(self) -> dict:
        """Build @context for output."""
        ctx = {}
        for term, value in self.context.items():
            ctx[term] = value
        return ctx
    
    def _build_node(self, subject: str, triples: list[Triple]) -> dict:
        """Build a JSON-LD node from triples."""
        node = {}
        
        if not subject.startswith("_:"):
            node["@id"] = self._compact_iri(subject)
        
        rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        
        for triple in triples:
            pred = triple.predicate
            obj = triple.object
            
            if pred == rdf_type:
                # Handle @type
                types = node.setdefault("@type", [])
                types.append(self._compact_iri(obj))
            else:
                # Regular property
                key = self._compact_iri(pred)
                value = self._compact_value(obj)
                
                if key in node:
                    # Multiple values - make array
                    if not isinstance(node[key], list):
                        node[key] = [node[key]]
                    node[key].append(value)
                else:
                    node[key] = value
        
        # Simplify single @type
        if "@type" in node and len(node["@type"]) == 1:
            node["@type"] = node["@type"][0]
        
        return node
    
    def _compact_iri(self, iri: str) -> str:
        """Compact an IRI using context."""
        if iri in self.inverse_context:
            return self.inverse_context[iri]
        
        # Try to find matching prefix
        for prefix, ns in STANDARD_PREFIXES.items():
            if iri.startswith(ns):
                local = iri[len(ns):]
                return f"{prefix}:{local}"
        
        return iri
    
    def _compact_value(self, value: str) -> Any:
        """Compact an object value."""
        # Check for literal
        if value.startswith('"'):
            return self._parse_literal(value)
        
        # IRI
        return self._compact_iri(value)
    
    def _parse_literal(self, lit: str) -> Any:
        """Parse a literal string."""
        # Extract value, language, datatype
        match = re.match(r'"((?:[^"\\]|\\.)*)"\s*(?:@([a-z-]+)|(?:\^\^<([^>]+)>))?', lit, re.I)
        if not match:
            return lit
        
        value = match.group(1)
        value = value.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
        value = value.replace('\\"', '"').replace("\\\\", "\\")
        
        lang = match.group(2)
        datatype = match.group(3)
        
        if lang:
            return {"@value": value, "@language": lang}
        
        if datatype:
            xsd = "http://www.w3.org/2001/XMLSchema#"
            if datatype == f"{xsd}integer":
                return int(value)
            elif datatype == f"{xsd}double" or datatype == f"{xsd}decimal":
                return float(value)
            elif datatype == f"{xsd}boolean":
                return value.lower() == "true"
            else:
                return {"@value": value, "@type": self._compact_iri(datatype)}
        
        return value


def parse_jsonld(source: Union[str, dict]) -> JSONLDDocument:
    """Convenience function to parse JSON-LD."""
    parser = JSONLDParser()
    return parser.parse(source)


def serialize_jsonld(
    triples: list[Triple],
    context: Optional[dict] = None,
    pretty: bool = True
) -> str:
    """Convenience function to serialize to JSON-LD."""
    serializer = JSONLDSerializer(context)
    return serializer.serialize(triples, pretty)
