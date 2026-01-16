"""
SPARQL-Star Parser using pyparsing.

Implements parsing of SPARQL-Star queries following the W3C specification,
with RDF-StarBase extensions for provenance filtering.
"""

from typing import Any, Optional
import pyparsing as pp
from pyparsing import (
    Keyword, Literal as Lit, Word, Regex, QuotedString,
    Suppress, Group, Optional as Opt, ZeroOrMore, OneOrMore,
    Forward, alphas, alphanums, nums, pyparsing_common,
    CaselessKeyword, Combine,
    DelimitedList,
)

from rdf_starbase.sparql.ast import (
    Query, SelectQuery, AskQuery,
    TriplePattern, QuotedTriplePattern,
    Variable, IRI, Literal, BlankNode,
    Filter, Comparison, LogicalExpression, FunctionCall,
    ComparisonOp, LogicalOp,
    WhereClause, ProvenanceFilter,
    Term,
)


class SPARQLStarParser:
    """
    Parser for SPARQL-Star queries.
    
    Supports:
    - Standard SPARQL SELECT, ASK queries
    - RDF-Star quoted triple patterns (<< s p o >>)
    - FILTER expressions with comparisons and functions
    - RDF-StarBase provenance extensions
    """
    
    def __init__(self):
        self._build_grammar()
    
    def _build_grammar(self):
        """Build the pyparsing grammar for SPARQL-Star."""
        
        # Enable packrat parsing for performance
        pp.ParserElement.enable_packrat()
        
        # =================================================================
        # Lexical tokens
        # =================================================================
        
        # Keywords (case-insensitive)
        SELECT = CaselessKeyword("SELECT")
        ASK = CaselessKeyword("ASK")
        WHERE = CaselessKeyword("WHERE")
        FILTER = CaselessKeyword("FILTER")
        PREFIX = CaselessKeyword("PREFIX")
        DISTINCT = CaselessKeyword("DISTINCT")
        LIMIT = CaselessKeyword("LIMIT")
        OFFSET = CaselessKeyword("OFFSET")
        ORDER = CaselessKeyword("ORDER")
        BY = CaselessKeyword("BY")
        ASC = CaselessKeyword("ASC")
        DESC = CaselessKeyword("DESC")
        AND = CaselessKeyword("AND") | Lit("&&")
        OR = CaselessKeyword("OR") | Lit("||")
        NOT = CaselessKeyword("NOT") | Lit("!")
        BOUND = CaselessKeyword("BOUND")
        ISIRI = CaselessKeyword("ISIRI") | CaselessKeyword("ISURI")
        ISBLANK = CaselessKeyword("ISBLANK")
        ISLITERAL = CaselessKeyword("ISLITERAL")
        STR = CaselessKeyword("STR")
        LANG = CaselessKeyword("LANG")
        DATATYPE = CaselessKeyword("DATATYPE")
        
        # RDF-StarBase extensions
        FILTER_SOURCE = CaselessKeyword("FILTER_SOURCE")
        FILTER_CONFIDENCE = CaselessKeyword("FILTER_CONFIDENCE")
        FILTER_TIMESTAMP = CaselessKeyword("FILTER_TIMESTAMP")
        FILTER_PROCESS = CaselessKeyword("FILTER_PROCESS")
        
        # Punctuation
        LBRACE = Suppress(Lit("{"))
        RBRACE = Suppress(Lit("}"))
        LPAREN = Suppress(Lit("("))
        RPAREN = Suppress(Lit(")"))
        DOT = Suppress(Lit("."))
        COMMA = Suppress(Lit(","))
        STAR = Lit("*")
        LQUOTE = Suppress(Lit("<<"))
        RQUOTE = Suppress(Lit(">>"))
        
        # Comparison operators
        comp_op = (
            Lit("<=") | Lit(">=") | Lit("!=") | Lit("<>") |
            Lit("=") | Lit("<") | Lit(">")
        )
        
        # =================================================================
        # Terms
        # =================================================================
        
        # Variable: ?name or $name
        def make_variable(tokens):
            return Variable(tokens[0][1:])
        
        variable = Combine(
            (Lit("?") | Lit("$")) + Word(alphas + "_", alphanums + "_")
        ).set_parse_action(make_variable)
        
        # IRI: <http://...> or prefix:localname
        def make_full_iri(tokens):
            return IRI(tokens[0][1:-1])
        
        full_iri = Combine(
            Lit("<") + Regex(r'[^<>]+') + Lit(">")
        ).set_parse_action(make_full_iri)
        
        # Prefixed name: prefix:local
        pname_ns = Combine(Opt(Word(alphas, alphanums + "_")) + Lit(":"))
        pname_local = Word(alphanums + "_.-")
        
        def make_prefixed_name(tokens):
            return IRI(tokens[0])
        
        prefixed_name = Combine(pname_ns + Opt(pname_local)).set_parse_action(make_prefixed_name)
        
        iri = full_iri | prefixed_name
        
        # Literals
        string_literal = (
            QuotedString('"', esc_char='\\', multiline=True) |
            QuotedString("'", esc_char='\\', multiline=True)
        )
        
        # Language tag: @en, @en-US
        lang_tag = Combine(Lit("@") + Word(alphas + "-"))
        
        # Datatype: ^^<type> or ^^prefix:type
        datatype = Suppress(Lit("^^")) + iri
        
        # Full literal with optional language or datatype
        def make_literal(tokens):
            value = tokens[0]
            lang = None
            dtype = None
            if len(tokens) > 1:
                if isinstance(tokens[1], str) and tokens[1].startswith("@"):
                    lang = tokens[1][1:]
                elif isinstance(tokens[1], IRI):
                    dtype = tokens[1].value
            return Literal(value, language=lang, datatype=dtype)
        
        literal = (string_literal + Opt(lang_tag | datatype)).set_parse_action(make_literal)
        
        # Numeric literals
        def make_int_literal(tokens):
            return Literal(tokens[0], datatype="http://www.w3.org/2001/XMLSchema#integer")
        
        def make_float_literal(tokens):
            return Literal(tokens[0], datatype="http://www.w3.org/2001/XMLSchema#decimal")
        
        integer_literal = pyparsing_common.signed_integer.copy().set_parse_action(make_int_literal)
        float_literal = pyparsing_common.real.copy().set_parse_action(make_float_literal)
        
        # Boolean literals
        def make_true(tokens):
            return Literal(True)
        
        def make_false(tokens):
            return Literal(False)
        
        boolean_literal = (
            CaselessKeyword("true").set_parse_action(make_true) |
            CaselessKeyword("false").set_parse_action(make_false)
        )
        
        # Blank node
        def make_blank_node(tokens):
            return BlankNode(tokens[0][2:])
        
        blank_node = Combine(
            Lit("_:") + Word(alphanums + "_")
        ).set_parse_action(make_blank_node)
        
        # =================================================================
        # Quoted Triple Pattern (RDF-Star)
        # =================================================================
        
        # Forward declaration for recursive quoted triples
        quoted_triple = Forward()
        
        # Term that can appear in a triple (including nested quoted triples)
        graph_term = variable | iri | literal | float_literal | integer_literal | boolean_literal | blank_node | quoted_triple
        
        # Quoted triple: << subject predicate object >>
        def make_quoted_triple(tokens):
            return QuotedTriplePattern(
                subject=tokens[0],
                predicate=tokens[1],
                object=tokens[2]
            )
        
        quoted_triple <<= (
            LQUOTE + graph_term + graph_term + graph_term + RQUOTE
        ).set_parse_action(make_quoted_triple)
        
        # Term including quoted triples
        term = graph_term
        
        # =================================================================
        # Triple Patterns
        # =================================================================
        
        def make_triple_pattern(tokens):
            return TriplePattern(
                subject=tokens[0],
                predicate=tokens[1],
                object=tokens[2]
            )
        
        triple_pattern = (
            term + term + term + Opt(DOT)
        ).set_parse_action(make_triple_pattern)
        
        # =================================================================
        # FILTER Expressions
        # =================================================================
        
        # Expression forward declaration
        expression = Forward()
        
        # Function call
        func_name = (
            BOUND | ISIRI | ISBLANK | ISLITERAL | STR | LANG | DATATYPE |
            Word(alphas, alphanums + "_")
        )
        
        def make_function_call(tokens):
            return FunctionCall(name=str(tokens[0]).upper(), arguments=list(tokens[1:]))
        
        function_call = (
            func_name + LPAREN + Opt(DelimitedList(expression)) + RPAREN
        ).set_parse_action(make_function_call)
        
        # Primary expression
        primary_expr = (
            function_call |
            variable |
            literal |
            float_literal |
            integer_literal |
            boolean_literal |
            iri |
            (LPAREN + expression + RPAREN)
        )
        
        # Comparison expression
        def make_comparison(tokens):
            if len(tokens) == 3:
                return Comparison(
                    left=tokens[0],
                    operator=ComparisonOp.from_str(tokens[1]),
                    right=tokens[2]
                )
            return tokens[0]
        
        comparison_expr = (
            primary_expr + Opt(comp_op + primary_expr)
        ).set_parse_action(make_comparison)
        
        # NOT expression
        def make_not(tokens):
            if len(tokens) == 2:  # Has NOT
                return LogicalExpression(LogicalOp.NOT, [tokens[1]])
            return tokens[0]
        
        not_expr = (
            Opt(NOT) + comparison_expr
        ).set_parse_action(make_not)
        
        # AND expression
        def make_and(tokens):
            tokens = list(tokens)
            if len(tokens) == 1:
                return tokens[0]
            return LogicalExpression(LogicalOp.AND, tokens)
        
        and_expr = (
            not_expr + ZeroOrMore(Suppress(AND) + not_expr)
        ).set_parse_action(make_and)
        
        # OR expression (lowest precedence)
        def make_or(tokens):
            tokens = list(tokens)
            if len(tokens) == 1:
                return tokens[0]
            return LogicalExpression(LogicalOp.OR, tokens)
        
        expression <<= (
            and_expr + ZeroOrMore(Suppress(OR) + and_expr)
        ).set_parse_action(make_or)
        
        # Standard FILTER
        def make_filter(tokens):
            return Filter(expression=tokens[0])
        
        filter_clause = (
            Suppress(FILTER) + LPAREN + expression + RPAREN
        ).set_parse_action(make_filter)
        
        # =================================================================
        # RDF-StarBase Provenance Filters
        # =================================================================
        
        def make_prov_filter(field_name):
            def action(tokens):
                return ProvenanceFilter(
                    provenance_field=field_name,
                    expression=tokens[0]
                )
            return action
        
        source_filter = (
            Suppress(FILTER_SOURCE) + LPAREN + expression + RPAREN
        ).set_parse_action(make_prov_filter("source"))
        
        confidence_filter = (
            Suppress(FILTER_CONFIDENCE) + LPAREN + expression + RPAREN
        ).set_parse_action(make_prov_filter("confidence"))
        
        timestamp_filter = (
            Suppress(FILTER_TIMESTAMP) + LPAREN + expression + RPAREN
        ).set_parse_action(make_prov_filter("timestamp"))
        
        process_filter = (
            Suppress(FILTER_PROCESS) + LPAREN + expression + RPAREN
        ).set_parse_action(make_prov_filter("process"))
        
        provenance_filter = source_filter | confidence_filter | timestamp_filter | process_filter
        
        any_filter = filter_clause | provenance_filter
        
        # =================================================================
        # WHERE Clause
        # =================================================================
        
        where_pattern = triple_pattern | any_filter
        
        def make_where_clause(tokens):
            patterns = []
            filters = []
            for token in tokens:
                if isinstance(token, (TriplePattern, QuotedTriplePattern)):
                    patterns.append(token)
                elif isinstance(token, (Filter, ProvenanceFilter)):
                    filters.append(token)
            return WhereClause(patterns=patterns, filters=filters)
        
        where_clause = (
            Suppress(WHERE) + LBRACE + ZeroOrMore(where_pattern) + RBRACE
        ).set_parse_action(make_where_clause)
        
        # =================================================================
        # PREFIX Declarations
        # =================================================================
        
        def make_prefix(tokens):
            prefix = tokens[0][:-1]  # Remove trailing colon
            uri = tokens[1].value
            return (prefix, uri)
        
        prefix_decl = (
            Suppress(PREFIX) + pname_ns + full_iri
        ).set_parse_action(make_prefix)
        
        # =================================================================
        # SELECT Query
        # =================================================================
        
        # Use a fresh copy of variable for select to avoid parse action interference
        select_variable = Combine(
            (Lit("?") | Lit("$")) + Word(alphas + "_", alphanums + "_")
        ).set_parse_action(make_variable)
        
        # Variable list or *
        def make_star(tokens):
            return []
        
        select_vars = (
            STAR.set_parse_action(make_star) |
            OneOrMore(select_variable)
        )
        
        # ORDER BY clause - use fresh copy 
        order_variable = Combine(
            (Lit("?") | Lit("$")) + Word(alphas + "_", alphanums + "_")
        ).set_parse_action(make_variable)
        
        def make_order_desc(tokens):
            return (tokens[0], False)
        
        def make_order_asc(tokens):
            return (tokens[0], True)
        
        # Plain variable for order by (no ASC/DESC) needs special handling
        def make_plain_order(tokens):
            # tokens[0] is the raw string like "?name", need to convert to Variable
            var_name = tokens[0][1:]  # Remove the ? or $
            return (Variable(var_name), True)  # Default to ascending
        
        plain_order_var = Combine(
            (Lit("?") | Lit("$")) + Word(alphas + "_", alphanums + "_")
        ).set_parse_action(make_plain_order)
        
        order_condition = (
            (Suppress(DESC) + LPAREN + order_variable + RPAREN).set_parse_action(make_order_desc) |
            (Suppress(ASC) + LPAREN + order_variable + RPAREN).set_parse_action(make_order_asc) |
            plain_order_var
        )
        
        order_clause = Suppress(ORDER) + Suppress(BY) + OneOrMore(order_condition)
        
        # LIMIT and OFFSET
        limit_clause = Suppress(LIMIT) + pyparsing_common.integer
        offset_clause = Suppress(OFFSET) + pyparsing_common.integer
        
        def make_select_query(tokens):
            prefixes = {}
            variables = []
            distinct = False
            where = WhereClause()
            limit = None
            offset = None
            order_by = []
            
            for token in tokens:
                if isinstance(token, tuple) and len(token) == 2:
                    if isinstance(token[0], str) and isinstance(token[1], str):
                        # This is a prefix declaration
                        prefixes[token[0]] = token[1]
                    elif isinstance(token[0], Variable):
                        # This is an order by condition
                        order_by.append(token)
                elif token == "DISTINCT":
                    distinct = True
                elif isinstance(token, Variable):
                    variables.append(token)
                elif isinstance(token, pp.ParseResults) or isinstance(token, list):
                    # Check what's in the list
                    token_list = list(token)
                    if token_list and isinstance(token_list[0], Variable):
                        variables = token_list
                    elif token_list and isinstance(token_list[0], tuple):
                        order_by = token_list
                    elif token_list == []:
                        pass  # SELECT *
                elif isinstance(token, WhereClause):
                    where = token
                elif isinstance(token, int):
                    if limit is None:
                        limit = token
                    else:
                        offset = token
            
            return SelectQuery(
                prefixes=prefixes,
                variables=variables,
                where=where,
                distinct=distinct,
                limit=limit,
                offset=offset,
                order_by=order_by,
            )
        
        def make_distinct(tokens):
            return "DISTINCT"
        
        select_query = (
            ZeroOrMore(prefix_decl) +
            Suppress(SELECT) +
            Opt(DISTINCT.set_parse_action(make_distinct)) +
            Group(select_vars) +
            where_clause +
            Opt(Group(order_clause)) +
            Opt(limit_clause) +
            Opt(offset_clause)
        ).set_parse_action(make_select_query)
        
        # =================================================================
        # ASK Query
        # =================================================================
        
        def make_ask_query(tokens):
            prefixes = {}
            where = WhereClause()
            for token in tokens:
                if isinstance(token, tuple) and len(token) == 2 and isinstance(token[0], str):
                    prefixes[token[0]] = token[1]
                elif isinstance(token, WhereClause):
                    where = token
            return AskQuery(prefixes=prefixes, where=where)
        
        ask_query = (
            ZeroOrMore(prefix_decl) +
            Suppress(ASK) +
            where_clause
        ).set_parse_action(make_ask_query)
        
        # =================================================================
        # Top-level Query
        # =================================================================
        
        self.query = select_query | ask_query
        
        # Ignore comments
        self.query.ignore(pp.pythonStyleComment)
        self.query.ignore(Lit("#") + pp.restOfLine)
    
    def parse(self, query_string: str) -> Query:
        """
        Parse a SPARQL-Star query string into an AST.
        
        Args:
            query_string: The SPARQL-Star query to parse
            
        Returns:
            Parsed Query AST
            
        Raises:
            ParseException: If the query is malformed
        """
        result = self.query.parse_string(query_string, parse_all=True)
        return result[0]


# Module-level parser instance for convenience
_parser: Optional[SPARQLStarParser] = None


def parse_query(query_string: str) -> Query:
    """
    Parse a SPARQL-Star query string.
    
    This is a convenience function that uses a cached parser instance.
    
    Args:
        query_string: The SPARQL-Star query to parse
        
    Returns:
        Parsed Query AST
    """
    global _parser
    if _parser is None:
        _parser = SPARQLStarParser()
    return _parser.parse(query_string)
