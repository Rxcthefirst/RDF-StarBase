"""
RDF/XML parser and serializer for RDF-StarBase.

RDF/XML is the original W3C standard serialization format for RDF.
While less human-readable than Turtle, it's still widely used in
legacy systems and some enterprise applications.

Key features:
- XML namespaces for prefixes
- rdf:Description for resource descriptions
- rdf:about for subject identification
- rdf:resource for object references
- Typed literals with rdf:datatype
- Language tags with xml:lang

Note: RDF-Star embedded triples are NOT supported in RDF/XML as there
is no standard syntax for them.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, Union, TextIO
from io import StringIO


@dataclass
class Triple:
    """A simple triple representation for parsing."""
    subject: str
    predicate: str
    object: str
    subject_triple: Optional["Triple"] = None
    object_triple: Optional["Triple"] = None


@dataclass
class RDFXMLDocument:
    """Parsed RDF/XML document."""
    prefixes: dict[str, str] = field(default_factory=dict)
    triples: list[Triple] = field(default_factory=list)
    base: Optional[str] = None


# XML namespace URIs
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"
XML_NS = "http://www.w3.org/XML/1998/namespace"

# Standard prefixes
STANDARD_PREFIXES = {
    "rdf": RDF_NS,
    "rdfs": RDFS_NS,
    "xsd": XSD_NS,
    "owl": "http://www.w3.org/2002/07/owl#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
}


class RDFXMLParser:
    """
    Parser for RDF/XML documents.
    
    Supports:
    - rdf:RDF root element
    - rdf:Description with rdf:about
    - Property elements with rdf:resource
    - Typed nodes (abbreviated syntax)
    - Literals with rdf:datatype and xml:lang
    - rdf:parseType="Resource" for blank nodes
    - rdf:parseType="Collection" for lists
    - rdf:nodeID for blank nodes
    """
    
    def __init__(self):
        self.prefixes = {}
        self.base = None
        self.triples = []
        self.blank_counter = 0
    
    def parse(self, source: Union[str, TextIO]) -> RDFXMLDocument:
        """Parse RDF/XML from string or file-like object."""
        if isinstance(source, str):
            # Check if it's a file path or XML content
            if source.strip().startswith("<"):
                root = ET.fromstring(source)
            else:
                root = ET.parse(source).getroot()
        else:
            root = ET.parse(source).getroot()
        
        self.prefixes = {}
        self.base = None
        self.triples = []
        self.blank_counter = 0
        
        # Extract namespaces from root
        self._extract_namespaces(root)
        
        # Extract base URI
        base_attr = root.get(f"{{{XML_NS}}}base")
        if base_attr:
            self.base = base_attr
        
        # Process root element
        if root.tag == f"{{{RDF_NS}}}RDF":
            # Standard rdf:RDF wrapper
            for child in root:
                self._process_element(child)
        else:
            # Document element is a typed node
            self._process_element(root)
        
        return RDFXMLDocument(
            prefixes=self.prefixes.copy(),
            triples=self.triples.copy(),
            base=self.base
        )
    
    def _extract_namespaces(self, element: ET.Element):
        """Extract namespace declarations."""
        # ElementTree handles namespaces with {uri}localname format
        # We need to extract from the raw XML for proper prefix handling
        # For now, use standard prefixes
        for prefix, uri in STANDARD_PREFIXES.items():
            self.prefixes[prefix] = uri
    
    def _gen_blank_node(self) -> str:
        """Generate a new blank node ID."""
        self.blank_counter += 1
        return f"_:b{self.blank_counter}"
    
    def _get_subject(self, element: ET.Element) -> str:
        """Get the subject URI from an element."""
        # Check rdf:about
        about = element.get(f"{{{RDF_NS}}}about")
        if about:
            return self._resolve_uri(about)
        
        # Check rdf:ID
        rdf_id = element.get(f"{{{RDF_NS}}}ID")
        if rdf_id:
            base = self.base or ""
            return f"{base}#{rdf_id}"
        
        # Check rdf:nodeID
        node_id = element.get(f"{{{RDF_NS}}}nodeID")
        if node_id:
            return f"_:{node_id}"
        
        # Generate blank node
        return self._gen_blank_node()
    
    def _resolve_uri(self, uri: str) -> str:
        """Resolve a URI against the base."""
        if uri.startswith("#") and self.base:
            return self.base + uri
        if not uri.startswith(("http://", "https://", "urn:", "_:")):
            if self.base:
                return self.base + uri
        return uri
    
    def _process_element(self, element: ET.Element, subject: Optional[str] = None) -> str:
        """Process an RDF/XML element and extract triples."""
        tag = element.tag
        
        # Handle typed nodes (not rdf:Description)
        if tag != f"{{{RDF_NS}}}Description":
            # This is a typed node - extract type triple
            subject = self._get_subject(element)
            type_uri = tag.replace("{", "").replace("}", "")
            self.triples.append(Triple(
                subject=subject,
                predicate=f"{RDF_NS}type",
                object=type_uri
            ))
        else:
            subject = self._get_subject(element)
        
        # Process property elements
        for prop_elem in element:
            self._process_property(subject, prop_elem)
        
        return subject
    
    def _process_property(self, subject: str, prop_elem: ET.Element):
        """Process a property element."""
        predicate = prop_elem.tag.replace("{", "").replace("}", "")
        
        # Check for rdf:resource (object reference)
        resource = prop_elem.get(f"{{{RDF_NS}}}resource")
        if resource is not None:
            obj = self._resolve_uri(resource)
            self.triples.append(Triple(subject, predicate, obj))
            return
        
        # Check for rdf:nodeID (blank node reference)
        node_id = prop_elem.get(f"{{{RDF_NS}}}nodeID")
        if node_id is not None:
            obj = f"_:{node_id}"
            self.triples.append(Triple(subject, predicate, obj))
            return
        
        # Check for rdf:parseType
        parse_type = prop_elem.get(f"{{{RDF_NS}}}parseType")
        if parse_type == "Resource":
            # Blank node with properties
            obj = self._gen_blank_node()
            self.triples.append(Triple(subject, predicate, obj))
            for child in prop_elem:
                self._process_property(obj, child)
            return
        elif parse_type == "Collection":
            # RDF list
            obj = self._process_collection(prop_elem)
            self.triples.append(Triple(subject, predicate, obj))
            return
        elif parse_type == "Literal":
            # XML literal (preserve as string)
            xml_content = ET.tostring(prop_elem, encoding='unicode', method='xml')
            # Extract inner content
            start = xml_content.find(">") + 1
            end = xml_content.rfind("<")
            literal_content = xml_content[start:end]
            obj = f'"{self._escape_string(literal_content)}"^^<{RDF_NS}XMLLiteral>'
            self.triples.append(Triple(subject, predicate, obj))
            return
        
        # Check for nested element (object is another resource)
        children = list(prop_elem)
        if children:
            # Nested resource
            obj = self._process_element(children[0])
            self.triples.append(Triple(subject, predicate, obj))
            return
        
        # Simple literal
        text = prop_elem.text or ""
        
        # Check for datatype
        datatype = prop_elem.get(f"{{{RDF_NS}}}datatype")
        if datatype:
            obj = f'"{self._escape_string(text)}"^^<{datatype}>'
        else:
            # Check for language
            lang = prop_elem.get(f"{{{XML_NS}}}lang")
            if lang:
                obj = f'"{self._escape_string(text)}"@{lang}'
            else:
                obj = f'"{self._escape_string(text)}"'
        
        self.triples.append(Triple(subject, predicate, obj))
    
    def _process_collection(self, prop_elem: ET.Element) -> str:
        """Process an rdf:parseType="Collection" element."""
        rdf_first = f"{RDF_NS}first"
        rdf_rest = f"{RDF_NS}rest"
        rdf_nil = f"{RDF_NS}nil"
        
        children = list(prop_elem)
        if not children:
            return rdf_nil
        
        head = self._gen_blank_node()
        current = head
        
        for i, child in enumerate(children):
            item = self._process_element(child)
            self.triples.append(Triple(current, rdf_first, item))
            
            if i < len(children) - 1:
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


class RDFXMLSerializer:
    """
    Serializer for RDF/XML output.
    
    Produces standard RDF/XML with rdf:Description elements.
    """
    
    def __init__(self, prefixes: Optional[dict[str, str]] = None):
        self.prefixes = prefixes or STANDARD_PREFIXES.copy()
        self.inverse_prefixes = {v: k for k, v in self.prefixes.items()}
    
    def serialize(self, triples: list[Triple], pretty: bool = True) -> str:
        """Serialize triples to RDF/XML."""
        # Register namespaces so ElementTree uses proper prefixes
        for prefix, uri in self.prefixes.items():
            ET.register_namespace(prefix, uri)
        ET.register_namespace("rdf", RDF_NS)
        
        # Create root element
        root = ET.Element(f"{{{RDF_NS}}}RDF")
        
        # Note: Don't manually add xmlns attributes - ElementTree handles this
        # when we use register_namespace and reference namespaces in elements
        
        # Group triples by subject
        subjects = {}
        for triple in triples:
            if triple.subject not in subjects:
                subjects[triple.subject] = []
            subjects[triple.subject].append(triple)
        
        # Create Description elements
        for subject, subject_triples in subjects.items():
            self._add_description(root, subject, subject_triples)
        
        # Generate XML
        if pretty:
            self._indent(root)
        
        return ET.tostring(root, encoding='unicode', xml_declaration=True)
    
    def _add_description(self, parent: ET.Element, subject: str, triples: list[Triple]):
        """Add an rdf:Description element."""
        desc = ET.SubElement(parent, f"{{{RDF_NS}}}Description")
        
        if subject.startswith("_:"):
            desc.set(f"{{{RDF_NS}}}nodeID", subject[2:])
        else:
            desc.set(f"{{{RDF_NS}}}about", subject)
        
        for triple in triples:
            self._add_property(desc, triple)
    
    def _add_property(self, parent: ET.Element, triple: Triple):
        """Add a property element."""
        pred = triple.predicate
        obj = triple.object
        
        # Convert IRI to Clark notation {namespace}localname for ElementTree
        prop_tag = self._uri_to_clark(pred)
        
        # Create property element
        prop = ET.SubElement(parent, prop_tag)
        
        # Check if object is a literal
        if obj.startswith('"'):
            # Parse literal
            self._set_literal(prop, obj)
        elif obj.startswith("_:"):
            # Blank node reference
            prop.set(f"{{{RDF_NS}}}nodeID", obj[2:])
        else:
            # IRI reference
            prop.set(f"{{{RDF_NS}}}resource", obj)
    
    def _uri_to_clark(self, uri: str) -> str:
        """Convert a URI to Clark notation {namespace}localname."""
        # Try to split at last # or last /
        if "#" in uri:
            namespace, localname = uri.rsplit("#", 1)
            return f"{{{namespace}#}}{localname}"
        elif "/" in uri:
            namespace, localname = uri.rsplit("/", 1)
            return f"{{{namespace}/}}{localname}"
        else:
            # Can't split - use as-is (will fail if not a valid name)
            return uri
    
    def _set_literal(self, element: ET.Element, literal: str):
        """Set literal value on element."""
        # Parse literal format: "value"@lang or "value"^^<datatype>
        match = re.match(r'"((?:[^"\\]|\\.)*)"\s*(?:@([a-z-]+)|(?:\^\^<([^>]+)>))?', literal, re.I)
        if not match:
            element.text = literal
            return
        
        value = match.group(1)
        # Unescape
        value = value.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
        value = value.replace('\\"', '"').replace("\\\\", "\\")
        
        element.text = value
        
        lang = match.group(2)
        datatype = match.group(3)
        
        if lang:
            element.set(f"{{{XML_NS}}}lang", lang)
        elif datatype:
            element.set(f"{{{RDF_NS}}}datatype", datatype)
    
    def _indent(self, elem: ET.Element, level: int = 0):
        """Add indentation to XML elements."""
        i = "\n" + "  " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


def parse_rdfxml(source: Union[str, TextIO]) -> RDFXMLDocument:
    """Convenience function to parse RDF/XML."""
    parser = RDFXMLParser()
    return parser.parse(source)


def serialize_rdfxml(
    triples: list[Triple],
    prefixes: Optional[dict[str, str]] = None,
    pretty: bool = True
) -> str:
    """Convenience function to serialize to RDF/XML."""
    serializer = RDFXMLSerializer(prefixes)
    return serializer.serialize(triples, pretty)
