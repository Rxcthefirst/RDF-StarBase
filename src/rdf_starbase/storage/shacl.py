"""
SHACL (Shapes Constraint Language) validation for RDF-StarBase.

Implements W3C SHACL Core for validating RDF graphs against shape constraints.
Supports validation on import with detailed validation reports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .duckdb import RDFStarStore

# SHACL namespace
SH = "http://www.w3.org/ns/shacl#"
XSD = "http://www.w3.org/2001/XMLSchema#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"


class Severity(Enum):
    """SHACL validation severity levels."""
    
    VIOLATION = f"{SH}Violation"
    WARNING = f"{SH}Warning"
    INFO = f"{SH}Info"


class NodeKind(Enum):
    """SHACL node kinds for sh:nodeKind constraint."""
    
    IRI = f"{SH}IRI"
    BLANK_NODE = f"{SH}BlankNode"
    LITERAL = f"{SH}Literal"
    BLANK_NODE_OR_IRI = f"{SH}BlankNodeOrIRI"
    BLANK_NODE_OR_LITERAL = f"{SH}BlankNodeOrLiteral"
    IRI_OR_LITERAL = f"{SH}IRIOrLiteral"


class ConstraintType(Enum):
    """Types of SHACL constraints."""
    
    # Cardinality
    MIN_COUNT = "minCount"
    MAX_COUNT = "maxCount"
    
    # Value type
    DATATYPE = "datatype"
    CLASS = "class"
    NODE_KIND = "nodeKind"
    
    # Value range
    MIN_EXCLUSIVE = "minExclusive"
    MIN_INCLUSIVE = "minInclusive"
    MAX_EXCLUSIVE = "maxExclusive"
    MAX_INCLUSIVE = "maxInclusive"
    
    # String-based
    MIN_LENGTH = "minLength"
    MAX_LENGTH = "maxLength"
    PATTERN = "pattern"
    FLAGS = "flags"
    LANGUAGE_IN = "languageIn"
    UNIQUE_LANG = "uniqueLang"
    
    # Property pair
    EQUALS = "equals"
    DISJOINT = "disjoint"
    LESS_THAN = "lessThan"
    LESS_THAN_OR_EQUALS = "lessThanOrEquals"
    
    # Logical
    NOT = "not"
    AND = "and"
    OR = "or"
    XONE = "xone"
    
    # Shape-based
    NODE = "node"
    PROPERTY = "property"
    QUALIFIED_VALUE_SHAPE = "qualifiedValueShape"
    QUALIFIED_MIN_COUNT = "qualifiedMinCount"
    QUALIFIED_MAX_COUNT = "qualifiedMaxCount"
    
    # Other
    CLOSED = "closed"
    IGNORED_PROPERTIES = "ignoredProperties"
    HAS_VALUE = "hasValue"
    IN = "in"


@dataclass
class ValidationResult:
    """A single validation result."""
    
    focus_node: str
    result_path: str | None
    value: Any | None
    source_shape: str
    source_constraint: str
    message: str
    severity: Severity = Severity.VIOLATION
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "focusNode": self.focus_node,
            "resultPath": self.result_path,
            "value": self.value,
            "sourceShape": self.source_shape,
            "sourceConstraintComponent": self.source_constraint,
            "resultMessage": self.message,
            "resultSeverity": self.severity.value,
        }


@dataclass
class ValidationReport:
    """SHACL validation report."""
    
    conforms: bool
    results: list[ValidationResult] = field(default_factory=list)
    
    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results.append(result)
        if result.severity == Severity.VIOLATION:
            self.conforms = False
    
    def violations(self) -> list[ValidationResult]:
        """Get all violations."""
        return [r for r in self.results if r.severity == Severity.VIOLATION]
    
    def warnings(self) -> list[ValidationResult]:
        """Get all warnings."""
        return [r for r in self.results if r.severity == Severity.WARNING]
    
    def infos(self) -> list[ValidationResult]:
        """Get all info messages."""
        return [r for r in self.results if r.severity == Severity.INFO]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "conforms": self.conforms,
            "results": [r.to_dict() for r in self.results],
            "violationCount": len(self.violations()),
            "warningCount": len(self.warnings()),
            "infoCount": len(self.infos()),
        }
    
    def to_rdf(self) -> str:
        """Convert to RDF Turtle representation."""
        lines = [
            "@prefix sh: <http://www.w3.org/ns/shacl#> .",
            "",
            "[] a sh:ValidationReport ;",
            f"    sh:conforms {'true' if self.conforms else 'false'} ;",
        ]
        
        if self.results:
            result_lines = []
            for i, result in enumerate(self.results):
                result_turtle = self._result_to_turtle(result)
                result_lines.append(result_turtle)
            
            lines.append("    sh:result " + ", ".join(result_lines) + " .")
        else:
            # Remove trailing semicolon
            lines[-1] = lines[-1].rstrip(" ;") + " ."
        
        return "\n".join(lines)
    
    def _result_to_turtle(self, result: ValidationResult) -> str:
        """Convert a single result to Turtle."""
        parts = ["[", "        a sh:ValidationResult"]
        parts.append(f"        sh:focusNode <{result.focus_node}>")
        
        if result.result_path:
            parts.append(f"        sh:resultPath <{result.result_path}>")
        
        if result.value is not None:
            if isinstance(result.value, str) and result.value.startswith("http"):
                parts.append(f"        sh:value <{result.value}>")
            else:
                escaped = str(result.value).replace('"', '\\"')
                parts.append(f'        sh:value "{escaped}"')
        
        parts.append(f"        sh:sourceShape <{result.source_shape}>")
        parts.append(f"        sh:sourceConstraintComponent <{result.source_constraint}>")
        parts.append(f'        sh:resultMessage "{result.message}"')
        parts.append(f"        sh:resultSeverity <{result.severity.value}>")
        parts.append("    ]")
        
        return " ;\n".join(parts[:-1]) + "\n    ]"


@dataclass
class PropertyConstraint:
    """A constraint on a property."""
    
    constraint_type: ConstraintType
    value: Any
    message: str | None = None


@dataclass
class PropertyShape:
    """A SHACL property shape."""
    
    shape_iri: str
    path: str
    constraints: list[PropertyConstraint] = field(default_factory=list)
    name: str | None = None
    description: str | None = None
    severity: Severity = Severity.VIOLATION
    deactivated: bool = False
    
    # Nested shapes
    node_shape: str | None = None
    qualified_value_shape: str | None = None
    qualified_min_count: int | None = None
    qualified_max_count: int | None = None


@dataclass
class NodeShape:
    """A SHACL node shape."""
    
    shape_iri: str
    target_class: list[str] = field(default_factory=list)
    target_node: list[str] = field(default_factory=list)
    target_subjects_of: list[str] = field(default_factory=list)
    target_objects_of: list[str] = field(default_factory=list)
    property_shapes: list[PropertyShape] = field(default_factory=list)
    constraints: list[PropertyConstraint] = field(default_factory=list)
    name: str | None = None
    description: str | None = None
    severity: Severity = Severity.VIOLATION
    deactivated: bool = False
    closed: bool = False
    ignored_properties: list[str] = field(default_factory=list)


class ShapesGraph:
    """A collection of SHACL shapes parsed from RDF."""
    
    def __init__(self) -> None:
        """Initialize empty shapes graph."""
        self.node_shapes: dict[str, NodeShape] = {}
        self.property_shapes: dict[str, PropertyShape] = {}
        self._triples: list[tuple[str, str, str]] = []
        self._blank_node_counter = 0
    
    def load_from_triples(self, triples: list[tuple[str, str, str]]) -> None:
        """Load shapes from a list of (subject, predicate, object) triples."""
        self._triples = triples
        self._parse_shapes()
    
    def load_from_turtle(self, turtle: str) -> None:
        """Load shapes from Turtle format."""
        # Simple Turtle parser for SHACL shapes
        triples = self._parse_turtle(turtle)
        self.load_from_triples(triples)
    
    def _new_blank_node(self) -> str:
        """Generate a new blank node ID."""
        self._blank_node_counter += 1
        return f"_:b{self._blank_node_counter}"
    
    def _parse_turtle(self, turtle: str) -> list[tuple[str, str, str]]:
        """Parse Turtle to triples (simplified parser)."""
        triples: list[tuple[str, str, str]] = []
        prefixes: dict[str, str] = {}
        
        # Extract prefixes
        prefix_pattern = r'@prefix\s+(\w*):\s*<([^>]+)>\s*\.'
        for match in re.finditer(prefix_pattern, turtle):
            prefixes[match.group(1)] = match.group(2)
        
        # Add standard prefixes if not defined
        if "sh" not in prefixes:
            prefixes["sh"] = SH
        if "xsd" not in prefixes:
            prefixes["xsd"] = XSD
        if "rdf" not in prefixes:
            prefixes["rdf"] = RDF
        if "rdfs" not in prefixes:
            prefixes["rdfs"] = RDFS
        
        def expand_uri(uri: str) -> str:
            """Expand prefixed URI to full IRI."""
            if uri.startswith("<") and uri.endswith(">"):
                return uri[1:-1]
            if uri.startswith("_:"):
                return uri
            if ":" in uri and not uri.startswith("http"):
                prefix, local = uri.split(":", 1)
                if prefix in prefixes:
                    return prefixes[prefix] + local
            return uri
        
        # Remove prefixes and comments
        content = re.sub(r'@prefix\s+\w*:\s*<[^>]+>\s*\.', '', turtle)
        content = re.sub(r'#[^\n]*', '', content)
        
        # Tokenize the content
        tokens = self._tokenize_turtle(content)
        
        # Parse tokens into triples
        self._parse_tokens(tokens, triples, expand_uri)
        
        return triples
    
    def _tokenize_turtle(self, text: str) -> list[str]:
        """Tokenize Turtle text."""
        tokens = []
        i = 0
        
        while i < len(text):
            # Skip whitespace
            while i < len(text) and text[i] in " \t\n\r":
                i += 1
            
            if i >= len(text):
                break
            
            # IRI
            if text[i] == "<":
                end = text.find(">", i)
                if end != -1:
                    tokens.append(text[i:end + 1])
                    i = end + 1
                    continue
            
            # Literal
            if text[i] == '"':
                i += 1
                start = i
                while i < len(text) and text[i] != '"':
                    if text[i] == "\\" and i + 1 < len(text):
                        i += 2
                    else:
                        i += 1
                if i < len(text):
                    tokens.append('"' + text[start:i] + '"')
                    i += 1
                continue
            
            # Special characters
            if text[i] in ";,.[]()":
                tokens.append(text[i])
                i += 1
                continue
            
            # ^^ for datatype
            if text[i:i + 2] == "^^":
                tokens.append("^^")
                i += 2
                continue
            
            # @prefix already removed, handle @language tags
            if text[i] == "@":
                start = i
                i += 1
                while i < len(text) and text[i].isalpha():
                    i += 1
                tokens.append(text[start:i])
                continue
            
            # Prefixed name or keyword
            start = i
            while i < len(text) and text[i] not in " \t\n\r;,.[]()" and text[i:i + 2] != "^^":
                i += 1
            if i > start:
                tokens.append(text[start:i])
        
        return tokens
    
    def _parse_tokens(
        self,
        tokens: list[str],
        triples: list[tuple[str, str, str]],
        expand_uri: Callable[[str], str],
    ) -> None:
        """Parse tokens into triples."""
        i = 0
        subject_stack: list[str] = []
        predicate_stack: list[str] = []
        current_subject: str | None = None
        current_predicate: str | None = None
        
        while i < len(tokens):
            token = tokens[i]
            
            if token == "[":
                # Start of blank node
                blank_id = self._new_blank_node()
                
                if current_predicate and current_subject:
                    # This blank node is an object
                    triples.append((current_subject, current_predicate, blank_id))
                
                # Push current context
                if current_subject:
                    subject_stack.append(current_subject)
                if current_predicate:
                    predicate_stack.append(current_predicate)
                
                current_subject = blank_id
                current_predicate = None
                i += 1
                continue
            
            if token == "]":
                # End of blank node
                if subject_stack:
                    current_subject = subject_stack.pop()
                else:
                    current_subject = None
                
                if predicate_stack:
                    current_predicate = predicate_stack.pop()
                else:
                    current_predicate = None
                
                i += 1
                continue
            
            if token == ";":
                # Next predicate-object pair
                current_predicate = None
                i += 1
                continue
            
            if token == ",":
                # Same predicate, next object
                i += 1
                continue
            
            if token == ".":
                # End of statement
                current_subject = None
                current_predicate = None
                i += 1
                continue
            
            # Handle 'a' as rdf:type
            if token == "a":
                token = f"{RDF}type"
            
            # Determine what this token is
            if current_subject is None:
                # This is a subject
                current_subject = expand_uri(token)
                i += 1
                continue
            
            if current_predicate is None:
                # This is a predicate
                current_predicate = expand_uri(token)
                i += 1
                continue
            
            # This is an object
            obj = token
            
            # Handle literals with datatype
            if obj.startswith('"'):
                obj_value = obj[1:-1] if obj.endswith('"') else obj[1:]
                
                # Check for datatype
                if i + 2 < len(tokens) and tokens[i + 1] == "^^":
                    datatype = expand_uri(tokens[i + 2])
                    obj = f'"{obj_value}"^^<{datatype}>'
                    i += 3
                else:
                    obj = f'"{obj_value}"'
                    i += 1
            else:
                obj = expand_uri(obj)
                i += 1
            
            triples.append((current_subject, current_predicate, obj))
    
    def _parse_statement(
        self, statement: str, expand_uri: Callable[[str], str]
    ) -> list[tuple[str, str, str]]:
        """Parse a single Turtle statement."""
        triples = []
        
        # Tokenize
        tokens = self._tokenize(statement)
        if len(tokens) < 3:
            return triples
        
        subject = expand_uri(tokens[0])
        i = 1
        
        while i < len(tokens):
            if tokens[i] == ";":
                i += 1
                continue
            if tokens[i] == ",":
                i += 1
                continue
            
            if i + 1 >= len(tokens):
                break
            
            predicate = expand_uri(tokens[i])
            i += 1
            
            # Handle object (could be list with commas)
            while i < len(tokens) and tokens[i] not in (";", "."):
                obj = tokens[i]
                
                # Handle literals
                if obj.startswith('"'):
                    # Find end of literal
                    if obj.endswith('"') and len(obj) > 1:
                        obj_value = obj[1:-1]
                    else:
                        obj_value = obj[1:]
                    
                    # Check for datatype
                    if i + 1 < len(tokens) and tokens[i + 1] == "^^":
                        i += 2
                        if i < len(tokens):
                            datatype = expand_uri(tokens[i])
                            obj_value = f'"{obj_value}"^^<{datatype}>'
                    else:
                        obj_value = f'"{obj_value}"'
                    
                    triples.append((subject, predicate, obj_value))
                elif obj == ",":
                    pass  # Continue with same predicate
                else:
                    triples.append((subject, predicate, expand_uri(obj)))
                
                i += 1
                
                # Check for comma (same predicate, different object)
                if i < len(tokens) and tokens[i] == ",":
                    i += 1
                    continue
                else:
                    break
        
        return triples
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize Turtle text."""
        tokens = []
        i = 0
        
        while i < len(text):
            # Skip whitespace
            while i < len(text) and text[i] in " \t\n\r":
                i += 1
            
            if i >= len(text):
                break
            
            # IRI
            if text[i] == "<":
                end = text.find(">", i)
                if end != -1:
                    tokens.append(text[i:end + 1])
                    i = end + 1
                    continue
            
            # Literal
            if text[i] == '"':
                i += 1
                start = i
                while i < len(text) and text[i] != '"':
                    if text[i] == "\\" and i + 1 < len(text):
                        i += 2
                    else:
                        i += 1
                if i < len(text):
                    tokens.append('"' + text[start:i] + '"')
                    i += 1
                continue
            
            # Special characters
            if text[i] in ";,.[]()":
                tokens.append(text[i])
                i += 1
                continue
            
            # ^^ for datatype
            if text[i:i + 2] == "^^":
                tokens.append("^^")
                i += 2
                continue
            
            # Prefixed name or keyword
            start = i
            while i < len(text) and text[i] not in " \t\n\r;,.[]()" and text[i:i + 2] != "^^":
                i += 1
            if i > start:
                tokens.append(text[start:i])
        
        return tokens
    
    def _parse_shapes(self) -> None:
        """Parse shapes from loaded triples."""
        # Index triples by subject
        by_subject: dict[str, list[tuple[str, str]]] = {}
        for s, p, o in self._triples:
            by_subject.setdefault(s, []).append((p, o))
        
        # Find NodeShapes
        for subject, predicates in by_subject.items():
            types = [o for p, o in predicates if p == f"{RDF}type"]
            
            if f"{SH}NodeShape" in types:
                node_shape = self._parse_node_shape(subject, predicates, by_subject)
                self.node_shapes[subject] = node_shape
            
            if f"{SH}PropertyShape" in types:
                prop_shape = self._parse_property_shape(subject, predicates, by_subject)
                self.property_shapes[subject] = prop_shape
    
    def _parse_node_shape(
        self,
        shape_iri: str,
        predicates: list[tuple[str, str]],
        by_subject: dict[str, list[tuple[str, str]]],
    ) -> NodeShape:
        """Parse a NodeShape from its predicates."""
        shape = NodeShape(shape_iri=shape_iri)
        
        for pred, obj in predicates:
            if pred == f"{SH}targetClass":
                shape.target_class.append(obj)
            elif pred == f"{SH}targetNode":
                shape.target_node.append(obj)
            elif pred == f"{SH}targetSubjectsOf":
                shape.target_subjects_of.append(obj)
            elif pred == f"{SH}targetObjectsOf":
                shape.target_objects_of.append(obj)
            elif pred == f"{SH}property":
                if obj in by_subject:
                    prop_shape = self._parse_property_shape(obj, by_subject[obj], by_subject)
                    shape.property_shapes.append(prop_shape)
            elif pred == f"{SH}name":
                shape.name = self._parse_literal(obj)
            elif pred == f"{SH}description":
                shape.description = self._parse_literal(obj)
            elif pred == f"{SH}severity":
                shape.severity = self._parse_severity(obj)
            elif pred == f"{SH}deactivated":
                shape.deactivated = self._parse_boolean(obj)
            elif pred == f"{SH}closed":
                shape.closed = self._parse_boolean(obj)
            elif pred == f"{SH}ignoredProperties":
                # Parse RDF list
                shape.ignored_properties = self._parse_list(obj, by_subject)
            else:
                # Check for constraint
                constraint = self._parse_constraint(pred, obj)
                if constraint:
                    shape.constraints.append(constraint)
        
        return shape
    
    def _parse_property_shape(
        self,
        shape_iri: str,
        predicates: list[tuple[str, str]],
        by_subject: dict[str, list[tuple[str, str]]],
    ) -> PropertyShape:
        """Parse a PropertyShape from its predicates."""
        shape = PropertyShape(shape_iri=shape_iri, path="")
        
        for pred, obj in predicates:
            if pred == f"{SH}path":
                shape.path = obj
            elif pred == f"{SH}name":
                shape.name = self._parse_literal(obj)
            elif pred == f"{SH}description":
                shape.description = self._parse_literal(obj)
            elif pred == f"{SH}severity":
                shape.severity = self._parse_severity(obj)
            elif pred == f"{SH}deactivated":
                shape.deactivated = self._parse_boolean(obj)
            elif pred == f"{SH}node":
                shape.node_shape = obj
            elif pred == f"{SH}qualifiedValueShape":
                shape.qualified_value_shape = obj
            elif pred == f"{SH}qualifiedMinCount":
                shape.qualified_min_count = self._parse_integer(obj)
            elif pred == f"{SH}qualifiedMaxCount":
                shape.qualified_max_count = self._parse_integer(obj)
            else:
                constraint = self._parse_constraint(pred, obj)
                if constraint:
                    shape.constraints.append(constraint)
        
        return shape
    
    def _parse_constraint(self, predicate: str, value: str) -> PropertyConstraint | None:
        """Parse a constraint from predicate and value."""
        if not predicate.startswith(SH):
            return None
        
        local_name = predicate[len(SH):]
        
        try:
            constraint_type = ConstraintType(local_name)
        except ValueError:
            return None
        
        # Parse value based on constraint type
        if constraint_type in (
            ConstraintType.MIN_COUNT,
            ConstraintType.MAX_COUNT,
            ConstraintType.MIN_LENGTH,
            ConstraintType.MAX_LENGTH,
            ConstraintType.QUALIFIED_MIN_COUNT,
            ConstraintType.QUALIFIED_MAX_COUNT,
        ):
            parsed_value = self._parse_integer(value)
        elif constraint_type in (
            ConstraintType.MIN_EXCLUSIVE,
            ConstraintType.MIN_INCLUSIVE,
            ConstraintType.MAX_EXCLUSIVE,
            ConstraintType.MAX_INCLUSIVE,
        ):
            parsed_value = self._parse_numeric(value)
        elif constraint_type in (ConstraintType.UNIQUE_LANG, ConstraintType.CLOSED):
            parsed_value = self._parse_boolean(value)
        elif constraint_type == ConstraintType.PATTERN:
            parsed_value = self._parse_literal(value)
        elif constraint_type == ConstraintType.FLAGS:
            parsed_value = self._parse_literal(value)
        else:
            parsed_value = value
        
        return PropertyConstraint(constraint_type=constraint_type, value=parsed_value)
    
    def _parse_literal(self, value: str) -> str:
        """Extract literal value from typed/untyped literal."""
        if value.startswith('"'):
            # Remove quotes and potential datatype/language
            match = re.match(r'"([^"]*)"', value)
            if match:
                return match.group(1)
        return value
    
    def _parse_integer(self, value: str) -> int:
        """Parse an integer literal."""
        literal = self._parse_literal(value)
        try:
            return int(literal)
        except ValueError:
            return 0
    
    def _parse_numeric(self, value: str) -> float:
        """Parse a numeric literal."""
        literal = self._parse_literal(value)
        try:
            return float(literal)
        except ValueError:
            return 0.0
    
    def _parse_boolean(self, value: str) -> bool:
        """Parse a boolean literal."""
        literal = self._parse_literal(value)
        return literal.lower() in ("true", "1")
    
    def _parse_severity(self, value: str) -> Severity:
        """Parse severity from IRI."""
        if value == f"{SH}Warning":
            return Severity.WARNING
        if value == f"{SH}Info":
            return Severity.INFO
        return Severity.VIOLATION
    
    def _parse_list(
        self, head: str, by_subject: dict[str, list[tuple[str, str]]]
    ) -> list[str]:
        """Parse an RDF list."""
        result = []
        current = head
        
        while current and current != f"{RDF}nil":
            if current not in by_subject:
                break
            
            predicates = dict(by_subject[current])
            first = predicates.get(f"{RDF}first")
            rest = predicates.get(f"{RDF}rest")
            
            if first:
                result.append(first)
            
            current = rest
        
        return result


class SHACLValidator:
    """Validates RDF data against SHACL shapes."""
    
    def __init__(self, shapes_graph: ShapesGraph) -> None:
        """Initialize validator with shapes graph."""
        self.shapes = shapes_graph
    
    def validate(
        self,
        data_triples: list[tuple[str, str, str, str | None]],
    ) -> ValidationReport:
        """
        Validate data triples against loaded shapes.
        
        Args:
            data_triples: List of (subject, predicate, object, graph) tuples
            
        Returns:
            ValidationReport with results
        """
        report = ValidationReport(conforms=True)
        
        # Index data by subject
        by_subject: dict[str, list[tuple[str, str]]] = {}
        for s, p, o, _g in data_triples:
            by_subject.setdefault(s, []).append((p, o))
        
        # Get types for each subject
        subject_types: dict[str, set[str]] = {}
        for s, predicates in by_subject.items():
            types = {o for p, o in predicates if p == f"{RDF}type"}
            subject_types[s] = types
        
        # Validate each shape
        for shape in self.shapes.node_shapes.values():
            if shape.deactivated:
                continue
            
            # Find focus nodes
            focus_nodes = self._get_focus_nodes(
                shape, by_subject, subject_types, data_triples
            )
            
            # Validate each focus node
            for focus_node in focus_nodes:
                predicates = by_subject.get(focus_node, [])
                self._validate_node_shape(
                    focus_node, predicates, shape, by_subject, report
                )
        
        return report
    
    def _get_focus_nodes(
        self,
        shape: NodeShape,
        by_subject: dict[str, list[tuple[str, str]]],
        subject_types: dict[str, set[str]],
        data_triples: list[tuple[str, str, str, str | None]],
    ) -> set[str]:
        """Get focus nodes for a shape based on targets."""
        focus_nodes: set[str] = set()
        
        # sh:targetClass - all instances of the class
        for target_class in shape.target_class:
            for subject, types in subject_types.items():
                if target_class in types:
                    focus_nodes.add(subject)
        
        # sh:targetNode - explicit nodes
        focus_nodes.update(shape.target_node)
        
        # sh:targetSubjectsOf - subjects of specific predicates
        for pred in shape.target_subjects_of:
            for s, p, _o, _g in data_triples:
                if p == pred:
                    focus_nodes.add(s)
        
        # sh:targetObjectsOf - objects of specific predicates
        for pred in shape.target_objects_of:
            for _s, p, o, _g in data_triples:
                if p == pred and not o.startswith('"'):  # Skip literals
                    focus_nodes.add(o)
        
        # If no explicit targets, shape could be used implicitly
        # (we don't auto-apply to everything)
        
        return focus_nodes
    
    def _validate_node_shape(
        self,
        focus_node: str,
        predicates: list[tuple[str, str]],
        shape: NodeShape,
        by_subject: dict[str, list[tuple[str, str]]],
        report: ValidationReport,
    ) -> None:
        """Validate a focus node against a node shape."""
        # Validate node constraints
        for constraint in shape.constraints:
            self._validate_constraint(
                focus_node, None, [focus_node], constraint, shape.shape_iri, report
            )
        
        # Validate property shapes
        for prop_shape in shape.property_shapes:
            if prop_shape.deactivated:
                continue
            
            # Get values for the property path
            values = [o for p, o in predicates if p == prop_shape.path]
            
            self._validate_property_shape(
                focus_node, values, prop_shape, by_subject, report
            )
        
        # Validate closed constraint
        if shape.closed:
            allowed = {ps.path for ps in shape.property_shapes}
            allowed.update(shape.ignored_properties)
            allowed.add(f"{RDF}type")  # Always allowed
            
            for pred, _obj in predicates:
                if pred not in allowed:
                    report.add_result(ValidationResult(
                        focus_node=focus_node,
                        result_path=pred,
                        value=None,
                        source_shape=shape.shape_iri,
                        source_constraint=f"{SH}ClosedConstraintComponent",
                        message=f"Property {pred} not allowed by closed shape",
                        severity=shape.severity,
                    ))
    
    def _validate_property_shape(
        self,
        focus_node: str,
        values: list[str],
        shape: PropertyShape,
        by_subject: dict[str, list[tuple[str, str]]],
        report: ValidationReport,
    ) -> None:
        """Validate values against a property shape."""
        for constraint in shape.constraints:
            self._validate_constraint(
                focus_node, shape.path, values, constraint, shape.shape_iri, report
            )
        
        # Validate nested node shape
        if shape.node_shape and shape.node_shape in self.shapes.node_shapes:
            nested_shape = self.shapes.node_shapes[shape.node_shape]
            for value in values:
                if not value.startswith('"'):  # Skip literals
                    predicates = by_subject.get(value, [])
                    self._validate_node_shape(
                        value, predicates, nested_shape, by_subject, report
                    )
    
    def _validate_constraint(
        self,
        focus_node: str,
        path: str | None,
        values: list[str],
        constraint: PropertyConstraint,
        source_shape: str,
        report: ValidationReport,
    ) -> None:
        """Validate a single constraint against values."""
        ct = constraint.constraint_type
        cv = constraint.value
        
        # Cardinality constraints
        if ct == ConstraintType.MIN_COUNT:
            if len(values) < cv:
                report.add_result(ValidationResult(
                    focus_node=focus_node,
                    result_path=path,
                    value=None,
                    source_shape=source_shape,
                    source_constraint=f"{SH}MinCountConstraintComponent",
                    message=f"Expected at least {cv} values, found {len(values)}",
                ))
            return
        
        if ct == ConstraintType.MAX_COUNT:
            if len(values) > cv:
                report.add_result(ValidationResult(
                    focus_node=focus_node,
                    result_path=path,
                    value=None,
                    source_shape=source_shape,
                    source_constraint=f"{SH}MaxCountConstraintComponent",
                    message=f"Expected at most {cv} values, found {len(values)}",
                ))
            return
        
        # Value constraints - apply to each value
        for value in values:
            if ct == ConstraintType.DATATYPE:
                self._validate_datatype(focus_node, path, value, cv, source_shape, report)
            elif ct == ConstraintType.CLASS:
                self._validate_class(focus_node, path, value, cv, source_shape, report)
            elif ct == ConstraintType.NODE_KIND:
                self._validate_node_kind(focus_node, path, value, cv, source_shape, report)
            elif ct == ConstraintType.MIN_LENGTH:
                self._validate_min_length(focus_node, path, value, cv, source_shape, report)
            elif ct == ConstraintType.MAX_LENGTH:
                self._validate_max_length(focus_node, path, value, cv, source_shape, report)
            elif ct == ConstraintType.PATTERN:
                self._validate_pattern(focus_node, path, value, cv, source_shape, report)
            elif ct == ConstraintType.MIN_INCLUSIVE:
                self._validate_min_inclusive(focus_node, path, value, cv, source_shape, report)
            elif ct == ConstraintType.MAX_INCLUSIVE:
                self._validate_max_inclusive(focus_node, path, value, cv, source_shape, report)
            elif ct == ConstraintType.MIN_EXCLUSIVE:
                self._validate_min_exclusive(focus_node, path, value, cv, source_shape, report)
            elif ct == ConstraintType.MAX_EXCLUSIVE:
                self._validate_max_exclusive(focus_node, path, value, cv, source_shape, report)
            elif ct == ConstraintType.HAS_VALUE:
                self._validate_has_value(focus_node, path, values, cv, source_shape, report)
                return  # Only check once for hasValue
            elif ct == ConstraintType.IN:
                self._validate_in(focus_node, path, value, cv, source_shape, report)
    
    def _validate_datatype(
        self, focus: str, path: str | None, value: str, expected: str,
        source: str, report: ValidationReport
    ) -> None:
        """Validate datatype constraint."""
        if not value.startswith('"'):
            report.add_result(ValidationResult(
                focus_node=focus,
                result_path=path,
                value=value,
                source_shape=source,
                source_constraint=f"{SH}DatatypeConstraintComponent",
                message=f"Expected literal with datatype {expected}, got IRI",
            ))
            return
        
        # Check datatype in literal
        if "^^" in value:
            match = re.search(r'\^\^<([^>]+)>', value)
            if match:
                actual_dt = match.group(1)
                if actual_dt != expected:
                    report.add_result(ValidationResult(
                        focus_node=focus,
                        result_path=path,
                        value=value,
                        source_shape=source,
                        source_constraint=f"{SH}DatatypeConstraintComponent",
                        message=f"Expected datatype {expected}, got {actual_dt}",
                    ))
        else:
            # Plain literal - check if expected is xsd:string
            if expected != f"{XSD}string":
                report.add_result(ValidationResult(
                    focus_node=focus,
                    result_path=path,
                    value=value,
                    source_shape=source,
                    source_constraint=f"{SH}DatatypeConstraintComponent",
                    message=f"Expected datatype {expected}, got plain literal",
                ))
    
    def _validate_class(
        self, focus: str, path: str | None, value: str, expected: str,
        source: str, report: ValidationReport
    ) -> None:
        """Validate class constraint."""
        # This would need access to type information
        # For now, just check if it's not a literal
        if value.startswith('"'):
            report.add_result(ValidationResult(
                focus_node=focus,
                result_path=path,
                value=value,
                source_shape=source,
                source_constraint=f"{SH}ClassConstraintComponent",
                message=f"Expected instance of {expected}, got literal",
            ))
    
    def _validate_node_kind(
        self, focus: str, path: str | None, value: str, expected: str,
        source: str, report: ValidationReport
    ) -> None:
        """Validate nodeKind constraint."""
        is_iri = not value.startswith('"') and not value.startswith("_:")
        is_blank = value.startswith("_:")
        is_literal = value.startswith('"')
        
        valid = False
        if expected == f"{SH}IRI":
            valid = is_iri
        elif expected == f"{SH}BlankNode":
            valid = is_blank
        elif expected == f"{SH}Literal":
            valid = is_literal
        elif expected == f"{SH}BlankNodeOrIRI":
            valid = is_blank or is_iri
        elif expected == f"{SH}BlankNodeOrLiteral":
            valid = is_blank or is_literal
        elif expected == f"{SH}IRIOrLiteral":
            valid = is_iri or is_literal
        
        if not valid:
            report.add_result(ValidationResult(
                focus_node=focus,
                result_path=path,
                value=value,
                source_shape=source,
                source_constraint=f"{SH}NodeKindConstraintComponent",
                message=f"Value does not match node kind {expected}",
            ))
    
    def _validate_min_length(
        self, focus: str, path: str | None, value: str, min_len: int,
        source: str, report: ValidationReport
    ) -> None:
        """Validate minLength constraint."""
        if value.startswith('"'):
            literal = self._extract_literal_value(value)
            if len(literal) < min_len:
                report.add_result(ValidationResult(
                    focus_node=focus,
                    result_path=path,
                    value=value,
                    source_shape=source,
                    source_constraint=f"{SH}MinLengthConstraintComponent",
                    message=f"String length {len(literal)} is less than minimum {min_len}",
                ))
    
    def _validate_max_length(
        self, focus: str, path: str | None, value: str, max_len: int,
        source: str, report: ValidationReport
    ) -> None:
        """Validate maxLength constraint."""
        if value.startswith('"'):
            literal = self._extract_literal_value(value)
            if len(literal) > max_len:
                report.add_result(ValidationResult(
                    focus_node=focus,
                    result_path=path,
                    value=value,
                    source_shape=source,
                    source_constraint=f"{SH}MaxLengthConstraintComponent",
                    message=f"String length {len(literal)} exceeds maximum {max_len}",
                ))
    
    def _validate_pattern(
        self, focus: str, path: str | None, value: str, pattern: str,
        source: str, report: ValidationReport
    ) -> None:
        """Validate pattern constraint."""
        if value.startswith('"'):
            literal = self._extract_literal_value(value)
            if not re.search(pattern, literal):
                report.add_result(ValidationResult(
                    focus_node=focus,
                    result_path=path,
                    value=value,
                    source_shape=source,
                    source_constraint=f"{SH}PatternConstraintComponent",
                    message=f"Value does not match pattern '{pattern}'",
                ))
    
    def _validate_min_inclusive(
        self, focus: str, path: str | None, value: str, min_val: float,
        source: str, report: ValidationReport
    ) -> None:
        """Validate minInclusive constraint."""
        if value.startswith('"'):
            try:
                num = float(self._extract_literal_value(value))
                if num < min_val:
                    report.add_result(ValidationResult(
                        focus_node=focus,
                        result_path=path,
                        value=value,
                        source_shape=source,
                        source_constraint=f"{SH}MinInclusiveConstraintComponent",
                        message=f"Value {num} is less than minimum {min_val}",
                    ))
            except ValueError:
                pass
    
    def _validate_max_inclusive(
        self, focus: str, path: str | None, value: str, max_val: float,
        source: str, report: ValidationReport
    ) -> None:
        """Validate maxInclusive constraint."""
        if value.startswith('"'):
            try:
                num = float(self._extract_literal_value(value))
                if num > max_val:
                    report.add_result(ValidationResult(
                        focus_node=focus,
                        result_path=path,
                        value=value,
                        source_shape=source,
                        source_constraint=f"{SH}MaxInclusiveConstraintComponent",
                        message=f"Value {num} exceeds maximum {max_val}",
                    ))
            except ValueError:
                pass
    
    def _validate_min_exclusive(
        self, focus: str, path: str | None, value: str, min_val: float,
        source: str, report: ValidationReport
    ) -> None:
        """Validate minExclusive constraint."""
        if value.startswith('"'):
            try:
                num = float(self._extract_literal_value(value))
                if num <= min_val:
                    report.add_result(ValidationResult(
                        focus_node=focus,
                        result_path=path,
                        value=value,
                        source_shape=source,
                        source_constraint=f"{SH}MinExclusiveConstraintComponent",
                        message=f"Value {num} is not greater than {min_val}",
                    ))
            except ValueError:
                pass
    
    def _validate_max_exclusive(
        self, focus: str, path: str | None, value: str, max_val: float,
        source: str, report: ValidationReport
    ) -> None:
        """Validate maxExclusive constraint."""
        if value.startswith('"'):
            try:
                num = float(self._extract_literal_value(value))
                if num >= max_val:
                    report.add_result(ValidationResult(
                        focus_node=focus,
                        result_path=path,
                        value=value,
                        source_shape=source,
                        source_constraint=f"{SH}MaxExclusiveConstraintComponent",
                        message=f"Value {num} is not less than {max_val}",
                    ))
            except ValueError:
                pass
    
    def _validate_has_value(
        self, focus: str, path: str | None, values: list[str], expected: str,
        source: str, report: ValidationReport
    ) -> None:
        """Validate hasValue constraint."""
        if expected not in values:
            report.add_result(ValidationResult(
                focus_node=focus,
                result_path=path,
                value=None,
                source_shape=source,
                source_constraint=f"{SH}HasValueConstraintComponent",
                message=f"Missing required value {expected}",
            ))
    
    def _validate_in(
        self, focus: str, path: str | None, value: str, allowed: Any,
        source: str, report: ValidationReport
    ) -> None:
        """Validate in constraint."""
        # allowed should be a list
        if isinstance(allowed, str):
            allowed_list = [allowed]
        elif isinstance(allowed, list):
            allowed_list = allowed
        else:
            return
        
        if value not in allowed_list:
            report.add_result(ValidationResult(
                focus_node=focus,
                result_path=path,
                value=value,
                source_shape=source,
                source_constraint=f"{SH}InConstraintComponent",
                message=f"Value not in allowed list",
            ))
    
    def _extract_literal_value(self, value: str) -> str:
        """Extract the string value from a literal."""
        if value.startswith('"'):
            # Find closing quote
            match = re.match(r'"([^"]*)"', value)
            if match:
                return match.group(1)
        return value


class SHACLManager:
    """High-level SHACL management for a repository."""
    
    def __init__(self, store: "RDFStarStore") -> None:
        """Initialize SHACL manager."""
        self.store = store
        self.shapes_graph = ShapesGraph()
        self.shapes_loaded = False
    
    def load_shapes(self, shapes_turtle: str) -> int:
        """
        Load SHACL shapes from Turtle format.
        
        Args:
            shapes_turtle: SHACL shapes in Turtle format
            
        Returns:
            Number of shapes loaded
        """
        self.shapes_graph = ShapesGraph()
        self.shapes_graph.load_from_turtle(shapes_turtle)
        self.shapes_loaded = True
        return len(self.shapes_graph.node_shapes)
    
    def load_shapes_from_graph(self, graph_name: str) -> int:
        """
        Load SHACL shapes from a named graph in the store.
        
        Args:
            graph_name: Name of the graph containing SHACL shapes
            
        Returns:
            Number of shapes loaded
        """
        # Query all triples from the shapes graph
        triples = []
        result = self.store.query(f"""
            SELECT ?s ?p ?o WHERE {{
                GRAPH <{graph_name}> {{ ?s ?p ?o }}
            }}
        """)
        
        for row in result.get("results", {}).get("bindings", []):
            s = self._binding_to_value(row.get("s", {}))
            p = self._binding_to_value(row.get("p", {}))
            o = self._binding_to_value(row.get("o", {}))
            if s and p and o:
                triples.append((s, p, o))
        
        self.shapes_graph = ShapesGraph()
        self.shapes_graph.load_from_triples(triples)
        self.shapes_loaded = True
        return len(self.shapes_graph.node_shapes)
    
    def _binding_to_value(self, binding: dict) -> str:
        """Convert SPARQL binding to value string."""
        t = binding.get("type", "")
        v = binding.get("value", "")
        
        if t == "uri":
            return v
        elif t == "literal":
            dt = binding.get("datatype")
            lang = binding.get("xml:lang")
            if dt:
                return f'"{v}"^^<{dt}>'
            elif lang:
                return f'"{v}"@{lang}'
            else:
                return f'"{v}"'
        elif t == "bnode":
            return f"_:{v}"
        return v
    
    def validate_graph(self, graph_name: str | None = None) -> ValidationReport:
        """
        Validate a graph against loaded shapes.
        
        Args:
            graph_name: Graph to validate (None for default graph)
            
        Returns:
            ValidationReport with results
        """
        if not self.shapes_loaded:
            return ValidationReport(conforms=True)
        
        # Query data triples
        if graph_name:
            query = f"""
                SELECT ?s ?p ?o WHERE {{
                    GRAPH <{graph_name}> {{ ?s ?p ?o }}
                }}
            """
        else:
            query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
        
        result = self.store.query(query)
        
        triples = []
        for row in result.get("results", {}).get("bindings", []):
            s = self._binding_to_value(row.get("s", {}))
            p = self._binding_to_value(row.get("p", {}))
            o = self._binding_to_value(row.get("o", {}))
            if s and p and o:
                triples.append((s, p, o, graph_name))
        
        validator = SHACLValidator(self.shapes_graph)
        return validator.validate(triples)
    
    def validate_triples(
        self,
        triples: list[tuple[str, str, str]],
        graph_name: str | None = None,
    ) -> ValidationReport:
        """
        Validate triples before import.
        
        Args:
            triples: List of (subject, predicate, object) tuples
            graph_name: Target graph name
            
        Returns:
            ValidationReport with results
        """
        if not self.shapes_loaded:
            return ValidationReport(conforms=True)
        
        # Convert to 4-tuples
        data = [(s, p, o, graph_name) for s, p, o in triples]
        
        validator = SHACLValidator(self.shapes_graph)
        return validator.validate(data)
    
    def get_shapes_summary(self) -> dict[str, Any]:
        """Get a summary of loaded shapes."""
        return {
            "loaded": self.shapes_loaded,
            "nodeShapeCount": len(self.shapes_graph.node_shapes),
            "propertyShapeCount": len(self.shapes_graph.property_shapes),
            "shapes": [
                {
                    "iri": shape.shape_iri,
                    "name": shape.name,
                    "targetClasses": shape.target_class,
                    "propertyCount": len(shape.property_shapes),
                    "constraintCount": len(shape.constraints),
                }
                for shape in self.shapes_graph.node_shapes.values()
            ],
        }
