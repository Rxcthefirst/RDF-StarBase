"""
Abstract Syntax Tree (AST) nodes for SPARQL-Star queries.

These classes represent the parsed structure of a SPARQL-Star query,
enabling type-safe query manipulation and execution.
"""

from dataclasses import dataclass, field
from typing import Union, Optional, Any
from enum import Enum, auto


class ComparisonOp(Enum):
    """Comparison operators for FILTER expressions."""
    EQ = auto()      # =
    NE = auto()      # !=
    LT = auto()      # <
    LE = auto()      # <=
    GT = auto()      # >
    GE = auto()      # >=
    
    @classmethod
    def from_str(cls, op: str) -> "ComparisonOp":
        mapping = {
            "=": cls.EQ, "==": cls.EQ,
            "!=": cls.NE, "<>": cls.NE,
            "<": cls.LT, "<=": cls.LE,
            ">": cls.GT, ">=": cls.GE,
        }
        return mapping[op]


class LogicalOp(Enum):
    """Logical operators for combining FILTER expressions."""
    AND = auto()
    OR = auto()
    NOT = auto()


# =============================================================================
# Term Types (subjects, predicates, objects)
# =============================================================================

@dataclass(frozen=True)
class Variable:
    """
    A SPARQL variable (e.g., ?name, $person).
    
    Variables are bound during query execution to values from matching triples.
    """
    name: str
    
    def __str__(self) -> str:
        return f"?{self.name}"
    
    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(frozen=True)
class IRI:
    """
    An Internationalized Resource Identifier.
    
    Can be a full IRI (<http://...>) or a prefixed name (foaf:name).
    """
    value: str
    
    def __str__(self) -> str:
        return f"<{self.value}>"
    
    def __hash__(self) -> int:
        return hash(self.value)


@dataclass(frozen=True)
class Literal:
    """
    An RDF Literal value.
    
    Can have an optional language tag (@en) or datatype (^^xsd:integer).
    """
    value: Any
    language: Optional[str] = None
    datatype: Optional[str] = None
    
    def __str__(self) -> str:
        base = f'"{self.value}"'
        if self.language:
            return f"{base}@{self.language}"
        if self.datatype:
            return f"{base}^^<{self.datatype}>"
        return base
    
    def __hash__(self) -> int:
        return hash((self.value, self.language, self.datatype))


@dataclass(frozen=True)
class BlankNode:
    """A blank node (anonymous resource)."""
    label: str
    
    def __str__(self) -> str:
        return f"_:{self.label}"


# Type alias for any term that can appear in a triple pattern
Term = Union[Variable, IRI, Literal, BlankNode, "QuotedTriplePattern"]


# =============================================================================
# Triple Patterns
# =============================================================================

@dataclass(frozen=True)
class TriplePattern:
    """
    A basic graph pattern matching triples in the store.
    
    Each position can be a variable (for matching) or a concrete term (for filtering).
    """
    subject: Term
    predicate: Term
    object: Term
    
    def __str__(self) -> str:
        return f"{self.subject} {self.predicate} {self.object} ."
    
    def get_variables(self) -> set[Variable]:
        """Return all variables in this pattern."""
        vars = set()
        for term in (self.subject, self.predicate, self.object):
            if isinstance(term, Variable):
                vars.add(term)
            elif isinstance(term, QuotedTriplePattern):
                vars.update(term.get_variables())
        return vars


@dataclass(frozen=True)
class QuotedTriplePattern:
    """
    An RDF-Star quoted triple pattern (<< s p o >>).
    
    This is the key innovation of SPARQL-Star - allows matching and 
    querying statements about statements.
    """
    subject: Term
    predicate: Term
    object: Term
    
    def __str__(self) -> str:
        return f"<< {self.subject} {self.predicate} {self.object} >>"
    
    def get_variables(self) -> set[Variable]:
        """Return all variables in this quoted pattern."""
        vars = set()
        for term in (self.subject, self.predicate, self.object):
            if isinstance(term, Variable):
                vars.add(term)
            elif isinstance(term, QuotedTriplePattern):
                vars.update(term.get_variables())
        return vars


# =============================================================================
# Filter Expressions
# =============================================================================

@dataclass
class Comparison:
    """A comparison expression (e.g., ?age > 30)."""
    left: Union[Variable, Literal, IRI, "FunctionCall"]
    operator: ComparisonOp
    right: Union[Variable, Literal, IRI, "FunctionCall"]
    
    def __str__(self) -> str:
        op_str = {
            ComparisonOp.EQ: "=", ComparisonOp.NE: "!=",
            ComparisonOp.LT: "<", ComparisonOp.LE: "<=",
            ComparisonOp.GT: ">", ComparisonOp.GE: ">=",
        }[self.operator]
        return f"{self.left} {op_str} {self.right}"


@dataclass
class LogicalExpression:
    """A logical combination of expressions (AND, OR, NOT)."""
    operator: LogicalOp
    operands: list[Union["Comparison", "LogicalExpression", "FunctionCall"]]
    
    def __str__(self) -> str:
        if self.operator == LogicalOp.NOT:
            return f"!({self.operands[0]})"
        op_str = " && " if self.operator == LogicalOp.AND else " || "
        return f"({op_str.join(str(o) for o in self.operands)})"


@dataclass
class FunctionCall:
    """A SPARQL function call (e.g., BOUND(?x), STR(?y))."""
    name: str
    arguments: list[Union[Variable, Literal, IRI, "FunctionCall"]]
    
    def __str__(self) -> str:
        args = ", ".join(str(a) for a in self.arguments)
        return f"{self.name}({args})"


@dataclass
class Filter:
    """A FILTER clause constraining query results."""
    expression: Union[Comparison, LogicalExpression, FunctionCall]
    
    def __str__(self) -> str:
        return f"FILTER({self.expression})"


# =============================================================================
# RDF-StarBase Extensions (Provenance Filters)
# =============================================================================

@dataclass
class ProvenanceFilter:
    """
    RDF-StarBase extension: filter by provenance metadata.
    
    Examples:
        FILTER_SOURCE(?source = "CRM")
        FILTER_CONFIDENCE(?conf > 0.8)
        FILTER_TIMESTAMP(?ts > "2026-01-01")
    """
    provenance_field: str  # source, confidence, timestamp, process
    expression: Comparison
    
    def __str__(self) -> str:
        return f"FILTER_{self.provenance_field.upper()}({self.expression})"


# =============================================================================
# Query Structure
# =============================================================================

@dataclass
class WhereClause:
    """The WHERE clause containing graph patterns and filters."""
    patterns: list[Union[TriplePattern, QuotedTriplePattern]] = field(default_factory=list)
    filters: list[Union[Filter, ProvenanceFilter]] = field(default_factory=list)
    
    def get_all_variables(self) -> set[Variable]:
        """Return all variables used in this WHERE clause."""
        vars = set()
        for pattern in self.patterns:
            vars.update(pattern.get_variables())
        return vars


@dataclass
class Query:
    """Base class for all SPARQL query types."""
    prefixes: dict[str, str] = field(default_factory=dict)


@dataclass
class SelectQuery(Query):
    """
    A SELECT query returning variable bindings.
    
    SELECT ?s ?p ?o
    WHERE { ?s ?p ?o }
    """
    variables: list[Variable] = field(default_factory=list)  # Empty list means SELECT *
    where: WhereClause = field(default_factory=WhereClause)
    distinct: bool = False
    limit: Optional[int] = None
    offset: Optional[int] = None
    order_by: list[tuple[Variable, bool]] = field(default_factory=list)  # (var, ascending)
    
    def is_select_all(self) -> bool:
        """Check if this is a SELECT * query."""
        return len(self.variables) == 0
    
    def __str__(self) -> str:
        parts = []
        
        # Prefixes
        for prefix, uri in self.prefixes.items():
            parts.append(f"PREFIX {prefix}: <{uri}>")

        
        # SELECT clause
        distinct_str = "DISTINCT " if self.distinct else ""
        if self.is_select_all():
            parts.append(f"SELECT {distinct_str}*")
        else:
            vars_str = " ".join(str(v) for v in self.variables)
            parts.append(f"SELECT {distinct_str}{vars_str}")
        
        # WHERE clause
        parts.append("WHERE {")
        for pattern in self.where.patterns:
            parts.append(f"  {pattern}")
        for filter in self.where.filters:
            parts.append(f"  {filter}")
        parts.append("}")
        
        # Modifiers
        if self.order_by:
            order_parts = []
            for var, asc in self.order_by:
                order_parts.append(str(var) if asc else f"DESC({var})")
            parts.append(f"ORDER BY {' '.join(order_parts)}")
        
        if self.limit:
            parts.append(f"LIMIT {self.limit}")
        
        if self.offset:
            parts.append(f"OFFSET {self.offset}")
        
        return "\n".join(parts)


@dataclass
class AskQuery(Query):
    """An ASK query returning boolean."""
    where: WhereClause = field(default_factory=WhereClause)


@dataclass 
class ConstructQuery(Query):
    """A CONSTRUCT query returning a new graph."""
    template: list[TriplePattern] = field(default_factory=list)
    where: WhereClause = field(default_factory=WhereClause)


@dataclass
class DescribeQuery(Query):
    """A DESCRIBE query returning information about resources."""
    resources: list[Union[Variable, IRI]] = field(default_factory=list)
    where: Optional[WhereClause] = None
