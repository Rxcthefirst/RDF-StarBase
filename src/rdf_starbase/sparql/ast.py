"""
Abstract Syntax Tree (AST) nodes for SPARQL-Star queries.

These classes represent the parsed structure of a SPARQL-Star query,
enabling type-safe query manipulation and execution.
"""

from dataclasses import dataclass, field
from typing import Union, Optional, Any
from enum import Enum, auto
from datetime import datetime


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


# =============================================================================
# Property Paths (SPARQL 1.1)
# =============================================================================

class PropertyPathModifier(Enum):
    """Modifiers for property path repetition."""
    ZERO_OR_MORE = auto()   # *
    ONE_OR_MORE = auto()    # +
    ZERO_OR_ONE = auto()    # ?
    FIXED_LENGTH = auto()   # {n,m}


@dataclass(frozen=True)
class PropertyPath:
    """
    Base class for property path expressions.
    
    Property paths allow navigation through RDF graphs:
    - foaf:knows/foaf:name  (sequence)
    - foaf:knows|foaf:friend (alternative)
    - foaf:knows* (zero or more)
    - foaf:knows+ (one or more)
    - foaf:knows? (zero or one)
    - ^foaf:knows (inverse)
    - !(foaf:knows|foaf:hates) (negated property set)
    """
    pass


@dataclass(frozen=True)
class PathIRI(PropertyPath):
    """A simple IRI in a property path."""
    iri: IRI
    
    def __str__(self) -> str:
        return str(self.iri)


@dataclass(frozen=True)
class PathSequence(PropertyPath):
    """A sequence of property paths (path1/path2/...)."""
    paths: tuple[PropertyPath, ...]
    
    def __str__(self) -> str:
        return "/".join(str(p) for p in self.paths)


@dataclass(frozen=True)
class PathAlternative(PropertyPath):
    """An alternative of property paths (path1|path2|...)."""
    paths: tuple[PropertyPath, ...]
    
    def __str__(self) -> str:
        return "|".join(str(p) for p in self.paths)


@dataclass(frozen=True)
class PathInverse(PropertyPath):
    """An inverse property path (^path)."""
    path: PropertyPath
    
    def __str__(self) -> str:
        return f"^{self.path}"


@dataclass(frozen=True)
class PathMod(PropertyPath):
    """A modified property path (path*, path+, path?)."""
    path: PropertyPath
    modifier: PropertyPathModifier
    
    def __str__(self) -> str:
        mod = {
            PropertyPathModifier.ZERO_OR_MORE: "*",
            PropertyPathModifier.ONE_OR_MORE: "+",
            PropertyPathModifier.ZERO_OR_ONE: "?",
        }[self.modifier]
        return f"{self.path}{mod}"


@dataclass(frozen=True)
class PathFixedLength(PropertyPath):
    """
    A fixed-length property path (path{n}, path{n,m}, path{n,}).
    
    Examples:
        foaf:knows{2}     - exactly 2 hops
        foaf:knows{2,4}   - 2 to 4 hops
        foaf:knows{2,}    - 2 or more hops
    """
    path: PropertyPath
    min_length: int
    max_length: Optional[int]  # None means unbounded
    
    def __str__(self) -> str:
        if self.max_length is None:
            return f"{self.path}{{{self.min_length},}}"
        elif self.min_length == self.max_length:
            return f"{self.path}{{{self.min_length}}}"
        else:
            return f"{self.path}{{{self.min_length},{self.max_length}}}"


@dataclass(frozen=True)
class PathNegatedPropertySet(PropertyPath):
    """A negated property set (!(iri1|iri2|...))."""
    iris: tuple[IRI, ...]
    
    def __str__(self) -> str:
        inner = "|".join(str(i) for i in self.iris)
        return f"!({inner})"


# Type alias for predicate which can be an IRI, Variable, or PropertyPath
PredicatePath = Union[IRI, Variable, PropertyPath]


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
    The predicate can also be a PropertyPath for path expressions.
    """
    subject: Term
    predicate: Union[Term, PropertyPath]  # Can be a property path
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
    
    def has_property_path(self) -> bool:
        """Check if predicate is a property path."""
        return isinstance(self.predicate, PropertyPath)


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
class ExistsExpression:
    """
    EXISTS or NOT EXISTS pattern expression.
    
    Used in FILTER to check if a graph pattern matches.
    
    Examples:
        FILTER EXISTS { ?person foaf:email ?email }
        FILTER NOT EXISTS { ?person foaf:deceased ?date }
    """
    pattern: "WhereClause"  # The graph pattern to check
    negated: bool = False  # True for NOT EXISTS
    
    def __str__(self) -> str:
        prefix = "NOT EXISTS" if self.negated else "EXISTS"
        return f"{prefix} {{ ... }}"


@dataclass
class AggregateExpression:
    """
    An aggregate function call (COUNT, SUM, AVG, MIN, MAX, GROUP_CONCAT, SAMPLE).
    
    Examples:
        COUNT(?x)
        COUNT(DISTINCT ?x)
        SUM(?price)
        GROUP_CONCAT(?name; separator=", ")
    """
    function: str  # COUNT, SUM, AVG, MIN, MAX, GROUP_CONCAT, SAMPLE
    argument: Optional[Union[Variable, Literal, IRI, "FunctionCall"]]  # None for COUNT(*)
    distinct: bool = False
    separator: Optional[str] = None  # For GROUP_CONCAT
    alias: Optional[Variable] = None  # AS ?varname
    
    def __str__(self) -> str:
        if self.argument is None:
            arg_str = "*"
        elif self.distinct:
            arg_str = f"DISTINCT {self.argument}"
        else:
            arg_str = str(self.argument)
        
        if self.separator and self.function == "GROUP_CONCAT":
            result = f'{self.function}({arg_str}; separator="{self.separator}")'
        else:
            result = f"{self.function}({arg_str})"
        
        if self.alias:
            result = f"({result} AS {self.alias})"
        return result


# Type alias for SELECT expressions (can be variables or aggregates)
SelectExpression = Union[Variable, AggregateExpression]


@dataclass
class Filter:
    """A FILTER clause constraining query results."""
    expression: Union[Comparison, LogicalExpression, FunctionCall, "ExistsExpression"]
    
    def __str__(self) -> str:
        return f"FILTER({self.expression})"


@dataclass
class ProvenanceFilter:
    """
    A FILTER clause for provenance-specific filtering.
    
    Used internally to optimize queries like:
        FILTER(?confidence > 0.8)
    when ?confidence is bound to a provenance column.
    """
    expression: Union[Comparison, LogicalExpression, FunctionCall]
    provenance_field: str  # e.g., "confidence", "source", "timestamp"
    
    def __str__(self) -> str:
        return f"FILTER_PROVENANCE({self.provenance_field}, {self.expression})"


@dataclass
class Bind:
    """
    A BIND clause assigning an expression to a variable.
    
    BIND(?price * ?quantity AS ?total)
    BIND("default" AS ?value)
    """
    expression: Union[Variable, Literal, IRI, Comparison, FunctionCall]
    variable: Variable
    
    def __str__(self) -> str:
        return f"BIND({self.expression} AS {self.variable})"


@dataclass
class ValuesClause:
    """
    A VALUES clause providing inline data.
    
    VALUES ?x { 1 2 3 }
    VALUES (?x ?y) { (1 2) (3 4) }
    """
    variables: list[Variable]
    bindings: list[list[Union[Literal, IRI, None]]]  # None for UNDEF
    
    def __str__(self) -> str:
        if len(self.variables) == 1:
            vals = " ".join(str(b[0]) if b[0] else "UNDEF" for b in self.bindings)
            return f"VALUES {self.variables[0]} {{ {vals} }}"
        else:
            vars_str = " ".join(str(v) for v in self.variables)
            rows = []
            for row in self.bindings:
                row_vals = " ".join(str(v) if v else "UNDEF" for v in row)
                rows.append(f"({row_vals})")
            return f"VALUES ({vars_str}) {{ {' '.join(rows)} }}"


# =============================================================================
# Graph Patterns (OPTIONAL, UNION, etc.)
# =============================================================================

@dataclass
class OptionalPattern:
    """
    An OPTIONAL graph pattern.
    
    OPTIONAL { ?s ?p ?o }
    OPTIONAL { ?s ?p ?o . BIND(?o AS ?val) }
    
    Results include rows even if the optional pattern doesn't match.
    """
    patterns: list[Union[TriplePattern, QuotedTriplePattern, "OptionalPattern", "UnionPattern"]] = field(default_factory=list)
    filters: list[Filter] = field(default_factory=list)
    binds: list["Bind"] = field(default_factory=list)
    
    def __str__(self) -> str:
        inner = " ".join(str(p) for p in self.patterns)
        return f"OPTIONAL {{ {inner} }}"
    
    def get_variables(self) -> set[Variable]:
        vars = set()
        for pattern in self.patterns:
            vars.update(pattern.get_variables())
        for bind in self.binds:
            vars.add(bind.variable)
        return vars


@dataclass
class UnionPattern:
    """
    A UNION of graph patterns.
    
    { ?s ?p ?o } UNION { ?s ?q ?r }
    """
    alternatives: list["GraphPattern"] = field(default_factory=list)
    
    def __str__(self) -> str:
        parts = [f"{{ {' '.join(str(p) for p in alt)} }}" for alt in self.alternatives]
        return " UNION ".join(parts)
    
    def get_variables(self) -> set[Variable]:
        vars = set()
        for alt in self.alternatives:
            for pattern in alt:
                if hasattr(pattern, 'get_variables'):
                    vars.update(pattern.get_variables())
        return vars


@dataclass
class GraphPattern:
    """A named graph pattern: GRAPH <uri> { ... }"""
    graph: Union[Variable, IRI]
    patterns: list[Union[TriplePattern, QuotedTriplePattern]] = field(default_factory=list)
    binds: list["Bind"] = field(default_factory=list)
    filters: list["Filter"] = field(default_factory=list)
    
    def get_variables(self) -> set[Variable]:
        vars = set()
        if isinstance(self.graph, Variable):
            vars.add(self.graph)
        for pattern in self.patterns:
            vars.update(pattern.get_variables())
        for bind in self.binds:
            vars.add(bind.variable)
        return vars


@dataclass
class MinusPattern:
    """
    A MINUS graph pattern for set difference.
    
    MINUS { ?s ?p ?o }
    
    Removes solutions where the MINUS pattern matches.
    """
    patterns: list[Union[TriplePattern, QuotedTriplePattern, "OptionalPattern", "UnionPattern"]] = field(default_factory=list)
    filters: list[Filter] = field(default_factory=list)
    
    def __str__(self) -> str:
        inner = " ".join(str(p) for p in self.patterns)
        return f"MINUS {{ {inner} }}"
    
    def get_variables(self) -> set[Variable]:
        vars = set()
        for pattern in self.patterns:
            vars.update(pattern.get_variables())
        return vars


# Type alias for patterns that can appear in WHERE
WherePattern = Union[TriplePattern, QuotedTriplePattern, OptionalPattern, UnionPattern, GraphPattern, Bind, ValuesClause, MinusPattern, "SubSelect"]


# =============================================================================
# Query Structure
# =============================================================================

@dataclass
class WhereClause:
    """The WHERE clause containing graph patterns and filters."""
    patterns: list[WherePattern] = field(default_factory=list)
    filters: list[Filter] = field(default_factory=list)
    optional_patterns: list[OptionalPattern] = field(default_factory=list)
    union_patterns: list[UnionPattern] = field(default_factory=list)
    minus_patterns: list[MinusPattern] = field(default_factory=list)
    binds: list[Bind] = field(default_factory=list)
    values: Optional[ValuesClause] = None
    graph_patterns: list[GraphPattern] = field(default_factory=list)
    subselects: list["SubSelect"] = field(default_factory=list)
    
    def get_all_variables(self) -> set[Variable]:
        """Return all variables used in this WHERE clause."""
        vars = set()
        for pattern in self.patterns:
            if hasattr(pattern, 'get_variables'):
                vars.update(pattern.get_variables())
        for opt in self.optional_patterns:
            vars.update(opt.get_variables())
        for union in self.union_patterns:
            vars.update(union.get_variables())
        for minus in self.minus_patterns:
            vars.update(minus.get_variables())
        for graph in self.graph_patterns:
            vars.update(graph.get_variables())
        for sub in self.subselects:
            vars.update(sub.get_variables())
        return vars


@dataclass
class SubSelect:
    """
    A subquery (nested SELECT) within a WHERE clause.
    
    Subqueries allow computing intermediate results within a larger query.
    The subquery is evaluated first, then joined with the outer query.
    
    Example:
        SELECT ?person ?avgAge WHERE {
            ?person a foaf:Person .
            {
                SELECT (AVG(?age) AS ?avgAge)
                WHERE { ?p foaf:age ?age }
            }
        }
    """
    variables: list[Union[Variable, "AggregateExpression"]]
    where: "WhereClause"
    distinct: bool = False
    group_by: list[Variable] = field(default_factory=list)
    having: Optional["Filter"] = None
    order_by: list["OrderCondition"] = field(default_factory=list)
    limit: Optional[int] = None
    offset: Optional[int] = None
    
    def get_variables(self) -> set[Variable]:
        """Return variables projected by this subquery."""
        vars = set()
        for v in self.variables:
            if isinstance(v, Variable):
                vars.add(v)
        return vars
    
    def __str__(self) -> str:
        var_str = " ".join(str(v) for v in self.variables)
        return f"{{ SELECT {var_str} WHERE {{ ... }} }}"


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
    
    SELECT ?s ?p ?o
    FROM <http://example.org/graph1>
    WHERE { ?s ?p ?o }
    
    SELECT ?s (COUNT(?p) AS ?count)
    WHERE { ?s ?p ?o }
    GROUP BY ?s
    HAVING (COUNT(?p) > 5)
    
    Time-travel query:
    SELECT ?s ?p ?o
    WHERE { ?s ?p ?o }
    AS OF "2025-01-15T00:00:00Z"
    """
    variables: list[SelectExpression] = field(default_factory=list)  # Empty list means SELECT *
    where: WhereClause = field(default_factory=WhereClause)
    distinct: bool = False
    limit: Optional[int] = None
    offset: Optional[int] = None
    order_by: list[tuple[Variable, bool]] = field(default_factory=list)  # (var, ascending)
    group_by: list[Variable] = field(default_factory=list)
    having: Optional[Union[Comparison, LogicalExpression, FunctionCall]] = None
    as_of: Optional[datetime] = None  # Time-travel: query as of this timestamp
    from_graphs: list[IRI] = field(default_factory=list)  # FROM <graph> clauses
    from_named_graphs: list[IRI] = field(default_factory=list)  # FROM NAMED <graph> clauses
    
    def is_select_all(self) -> bool:
        """Check if this is a SELECT * query."""
        return len(self.variables) == 0
    
    def has_aggregates(self) -> bool:
        """Check if any SELECT expression is an aggregate."""
        return any(isinstance(v, AggregateExpression) for v in self.variables)
    
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
    as_of: Optional[datetime] = None  # Time-travel: query as of this timestamp


@dataclass 
class ConstructQuery(Query):
    """A CONSTRUCT query returning a new graph."""
    template: list[TriplePattern] = field(default_factory=list)
    where: WhereClause = field(default_factory=WhereClause)
    as_of: Optional[datetime] = None  # Time-travel: query as of this timestamp


@dataclass
class DescribeQuery(Query):
    """A DESCRIBE query returning information about resources."""
    resources: list[Union[Variable, IRI]] = field(default_factory=list)
    where: Optional[WhereClause] = None
    as_of: Optional[datetime] = None  # Time-travel: query as of this timestamp


@dataclass
class InsertDataQuery(Query):
    """
    An INSERT DATA update operation.
    
    INSERT DATA { 
        <s1> <p1> <o1> .
        <s2> <p2> "literal" .
    }
    
    Note: INSERT DATA does not allow variables - all terms must be ground.
    """
    triples: list[TriplePattern] = field(default_factory=list)
    graph: Optional[IRI] = None  # Optional GRAPH clause
    
    def __str__(self) -> str:
        parts = []
        for prefix, uri in self.prefixes.items():
            parts.append(f"PREFIX {prefix}: <{uri}>")
        
        parts.append("INSERT DATA {")
        if self.graph:
            parts.append(f"  GRAPH <{self.graph.value}> {{")
            for triple in self.triples:
                parts.append(f"    {triple} .")
            parts.append("  }")
        else:
            for triple in self.triples:
                parts.append(f"  {triple} .")
        parts.append("}")
        return "\n".join(parts)


@dataclass
class DeleteDataQuery(Query):
    """
    A DELETE DATA update operation.
    
    DELETE DATA {
        <s1> <p1> <o1> .
    }
    """
    triples: list[TriplePattern] = field(default_factory=list)
    graph: Optional[IRI] = None


@dataclass
class UpdateQuery(Query):
    """Base class for SPARQL UPDATE operations."""
    pass


@dataclass
class DeleteWhereQuery(Query):
    """
    A DELETE WHERE update operation.
    
    DELETE WHERE {
        ?s ?p ?o .
    }
    
    The pattern in the WHERE clause is also used as the delete template.
    """
    where: WhereClause = field(default_factory=WhereClause)
    graph: Optional[IRI] = None
    
    def __str__(self) -> str:
        parts = []
        for prefix, uri in self.prefixes.items():
            parts.append(f"PREFIX {prefix}: <{uri}>")
        parts.append("DELETE WHERE {")
        for pattern in self.where.patterns:
            parts.append(f"  {pattern} .")
        parts.append("}")
        return "\n".join(parts)


@dataclass
class ModifyQuery(Query):
    """
    A DELETE/INSERT WHERE update operation.
    
    DELETE { <delete patterns> }
    INSERT { <insert patterns> }
    WHERE { <where patterns> }
    
    Can have either DELETE, INSERT, or both.
    Variables in the templates are bound from the WHERE clause.
    """
    delete_patterns: list[TriplePattern] = field(default_factory=list)
    insert_patterns: list[TriplePattern] = field(default_factory=list)
    where: WhereClause = field(default_factory=WhereClause)
    graph: Optional[IRI] = None
    
    def __str__(self) -> str:
        parts = []
        for prefix, uri in self.prefixes.items():
            parts.append(f"PREFIX {prefix}: <{uri}>")
        
        if self.delete_patterns:
            parts.append("DELETE {")
            for pattern in self.delete_patterns:
                parts.append(f"  {pattern} .")
            parts.append("}")
        
        if self.insert_patterns:
            parts.append("INSERT {")
            for pattern in self.insert_patterns:
                parts.append(f"  {pattern} .")
            parts.append("}")
        
        parts.append("WHERE {")
        for pattern in self.where.patterns:
            parts.append(f"  {pattern} .")
        parts.append("}")
        return "\n".join(parts)


# =============================================================================
# Graph Management Queries
# =============================================================================

@dataclass
class CreateGraphQuery(Query):
    """
    CREATE GRAPH <uri>
    
    Creates a new empty named graph.
    """
    graph_uri: IRI = None
    silent: bool = False
    
    def __str__(self) -> str:
        silent_str = "SILENT " if self.silent else ""
        return f"CREATE {silent_str}GRAPH <{self.graph_uri.value}>"


@dataclass
class DropGraphQuery(Query):
    """
    DROP GRAPH <uri>
    DROP DEFAULT
    DROP NAMED
    DROP ALL
    
    Removes a graph from the graph store.
    """
    graph_uri: Optional[IRI] = None
    target: str = "graph"  # "graph", "default", "named", "all"
    silent: bool = False
    
    def __str__(self) -> str:
        silent_str = "SILENT " if self.silent else ""
        if self.target == "default":
            return f"DROP {silent_str}DEFAULT"
        elif self.target == "named":
            return f"DROP {silent_str}NAMED"
        elif self.target == "all":
            return f"DROP {silent_str}ALL"
        else:
            return f"DROP {silent_str}GRAPH <{self.graph_uri.value}>"


@dataclass
class ClearGraphQuery(Query):
    """
    CLEAR GRAPH <uri>
    CLEAR DEFAULT
    CLEAR NAMED
    CLEAR ALL
    
    Removes all triples from a graph but keeps the graph.
    """
    graph_uri: Optional[IRI] = None
    target: str = "graph"  # "graph", "default", "named", "all"
    silent: bool = False
    
    def __str__(self) -> str:
        silent_str = "SILENT " if self.silent else ""
        if self.target == "default":
            return f"CLEAR {silent_str}DEFAULT"
        elif self.target == "named":
            return f"CLEAR {silent_str}NAMED"
        elif self.target == "all":
            return f"CLEAR {silent_str}ALL"
        else:
            return f"CLEAR {silent_str}GRAPH <{self.graph_uri.value}>"


@dataclass
class LoadQuery(Query):
    """
    LOAD <source> INTO GRAPH <dest>
    
    Loads RDF from a source URI into a graph.
    """
    source_uri: IRI = None
    graph_uri: Optional[IRI] = None  # None = default graph
    silent: bool = False
    
    def __str__(self) -> str:
        silent_str = "SILENT " if self.silent else ""
        if self.graph_uri:
            return f"LOAD {silent_str}<{self.source_uri.value}> INTO GRAPH <{self.graph_uri.value}>"
        return f"LOAD {silent_str}<{self.source_uri.value}>"


@dataclass
class CopyGraphQuery(Query):
    """
    COPY <source> TO <dest>
    
    Copies all triples from source to destination (clears dest first).
    """
    source_graph: Optional[IRI] = None  # None = DEFAULT
    dest_graph: IRI = None
    silent: bool = False
    source_is_default: bool = False
    
    def __str__(self) -> str:
        silent_str = "SILENT " if self.silent else ""
        src = "DEFAULT" if self.source_is_default else f"GRAPH <{self.source_graph.value}>"
        return f"COPY {silent_str}{src} TO GRAPH <{self.dest_graph.value}>"


@dataclass
class MoveGraphQuery(Query):
    """
    MOVE <source> TO <dest>
    
    Moves all triples from source to destination (clears both source and dest).
    """
    source_graph: Optional[IRI] = None  # None = DEFAULT
    dest_graph: IRI = None
    silent: bool = False
    source_is_default: bool = False
    
    def __str__(self) -> str:
        silent_str = "SILENT " if self.silent else ""
        src = "DEFAULT" if self.source_is_default else f"GRAPH <{self.source_graph.value}>"
        return f"MOVE {silent_str}{src} TO GRAPH <{self.dest_graph.value}>"


@dataclass
class AddGraphQuery(Query):
    """
    ADD <source> TO <dest>
    
    Adds all triples from source to destination (doesn't clear dest).
    """
    source_graph: Optional[IRI] = None  # None = DEFAULT
    dest_graph: IRI = None
    silent: bool = False
    source_is_default: bool = False
    
    def __str__(self) -> str:
        silent_str = "SILENT " if self.silent else ""
        src = "DEFAULT" if self.source_is_default else f"GRAPH <{self.source_graph.value}>"
        return f"ADD {silent_str}{src} TO GRAPH <{self.dest_graph.value}>"
