"""
SPARQL-Star Parser using pyparsing.

Implements parsing of SPARQL-Star queries following the W3C specification.
"""

from functools import lru_cache
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
    Query, SelectQuery, AskQuery, InsertDataQuery, DeleteDataQuery,
    DeleteWhereQuery, ModifyQuery,
    DescribeQuery, ConstructQuery,
    TriplePattern, QuotedTriplePattern,
    OptionalPattern, UnionPattern, GraphPattern, MinusPattern,
    Variable, IRI, Literal, BlankNode,
    Filter, Comparison, LogicalExpression, FunctionCall,
    AggregateExpression, Bind, ValuesClause,
    ComparisonOp, LogicalOp,
    WhereClause,
    Term,
    ExistsExpression, SubSelect,
    # Property Path types
    PropertyPath, PathIRI, PathSequence, PathAlternative,
    PathInverse, PathMod, PathFixedLength, PathNegatedPropertySet,
    PropertyPathModifier,
    # Graph management
    CreateGraphQuery, DropGraphQuery, ClearGraphQuery,
    LoadQuery, CopyGraphQuery, MoveGraphQuery, AddGraphQuery,
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
        
        # Additional SPARQL functions
        COALESCE = CaselessKeyword("COALESCE")
        IF = CaselessKeyword("IF")
        EXISTS = CaselessKeyword("EXISTS")
        NOT_EXISTS = CaselessKeyword("NOT") + CaselessKeyword("EXISTS")
        STRLEN = CaselessKeyword("STRLEN")
        CONTAINS = CaselessKeyword("CONTAINS")
        STRSTARTS = CaselessKeyword("STRSTARTS")
        STRENDS = CaselessKeyword("STRENDS")
        LCASE = CaselessKeyword("LCASE")
        UCASE = CaselessKeyword("UCASE")
        CONCAT = CaselessKeyword("CONCAT")
        REPLACE = CaselessKeyword("REPLACE")
        ABS = CaselessKeyword("ABS")
        ROUND = CaselessKeyword("ROUND")
        CEIL = CaselessKeyword("CEIL")
        FLOOR = CaselessKeyword("FLOOR")
        REGEX = CaselessKeyword("REGEX")
        
        # SPARQL Update keywords
        INSERT = CaselessKeyword("INSERT")
        DELETE = CaselessKeyword("DELETE")
        DATA = CaselessKeyword("DATA")
        GRAPH = CaselessKeyword("GRAPH")
        
        # Graph management keywords
        CREATE = CaselessKeyword("CREATE")
        DROP = CaselessKeyword("DROP")
        CLEAR = CaselessKeyword("CLEAR")
        LOAD = CaselessKeyword("LOAD")
        COPY = CaselessKeyword("COPY")
        MOVE = CaselessKeyword("MOVE")
        ADD = CaselessKeyword("ADD")
        TO = CaselessKeyword("TO")
        INTO = CaselessKeyword("INTO")
        DEFAULT = CaselessKeyword("DEFAULT")
        NAMED = CaselessKeyword("NAMED")
        ALL = CaselessKeyword("ALL")
        SILENT = CaselessKeyword("SILENT")
        FROM = CaselessKeyword("FROM")
        
        # Additional SPARQL keywords
        OPTIONAL = CaselessKeyword("OPTIONAL")
        UNION = CaselessKeyword("UNION")
        MINUS = CaselessKeyword("MINUS")
        DESCRIBE = CaselessKeyword("DESCRIBE")
        CONSTRUCT = CaselessKeyword("CONSTRUCT")
        
        # GROUP BY and HAVING keywords
        GROUP = CaselessKeyword("GROUP")
        HAVING = CaselessKeyword("HAVING")
        AS = CaselessKeyword("AS")
        
        # BIND and VALUES keywords
        BIND = CaselessKeyword("BIND")
        VALUES = CaselessKeyword("VALUES")
        UNDEF = CaselessKeyword("UNDEF")
        
        # Time-travel keyword
        OF = CaselessKeyword("OF")  # AS is already defined, we combine AS + OF
        
        # Aggregate function keywords
        COUNT = CaselessKeyword("COUNT")
        SUM = CaselessKeyword("SUM")
        AVG = CaselessKeyword("AVG")
        MIN = CaselessKeyword("MIN")
        MAX = CaselessKeyword("MAX")
        GROUP_CONCAT = CaselessKeyword("GROUP_CONCAT")
        SAMPLE = CaselessKeyword("SAMPLE")
        SEPARATOR = CaselessKeyword("SEPARATOR")
        
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
        # Note: Forward slashes NOT allowed here (they are path separators in property paths)
        # Use full IRIs for path-like local names, e.g., <http://example.org/customer/123>
        pname_ns = Combine(Opt(Word(alphas, alphanums + "_")) + Lit(":"))
        pname_local = Word(alphanums + "_.-")
        
        def make_prefixed_name(tokens):
            return IRI(tokens[0])
        
        prefixed_name = Combine(pname_ns + Opt(pname_local)).set_parse_action(make_prefixed_name)
        
        iri = full_iri | prefixed_name
        
        # 'a' keyword as shorthand for rdf:type (SPARQL standard)
        RDF_TYPE_IRI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        
        def make_a_keyword(tokens):
            return IRI(RDF_TYPE_IRI)
        
        a_keyword = Keyword("a").set_parse_action(make_a_keyword)
        
        # IRI or 'a' keyword (for predicates)
        iri_or_a = iri | a_keyword

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
        
        # =================================================================
        # AS OF Clause (Time-travel queries)
        # =================================================================
        
        from datetime import datetime, timezone
        
        # ISO 8601 datetime string: "2025-01-15T00:00:00Z" or "2025-01-15"
        def parse_datetime(tokens):
            dt_str = tokens[0]
            # Try various ISO formats
            for fmt in [
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d",
            ]:
                try:
                    dt = datetime.strptime(dt_str, fmt)
                    # Ensure UTC if no timezone
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except ValueError:
                    continue
            raise ValueError(f"Cannot parse datetime: {dt_str}")
        
        datetime_literal = QuotedString('"', esc_char='\\').copy().set_parse_action(parse_datetime)
        
        as_of_clause = (Suppress(AS) + Suppress(OF) + datetime_literal)
        
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
        # Property Paths (SPARQL 1.1)
        # =================================================================
        # 
        # Property paths are recognized by explicit path operators:
        # - ^ (inverse) at the start
        # - ! (negated) at the start  
        # - *, +, ? after an IRI
        # - / between IRIs (sequence)
        # - | between paths (alternative)
        #
        # A plain IRI like foaf:knows is NOT a property path.
        
        # Path modifiers - must NOT be followed by alphanumeric (to avoid ?var collision)
        from pyparsing import NotAny, Regex as PpRegex
        
        PATH_STAR = Lit("*") + NotAny(Word(alphanums))
        PATH_PLUS = Lit("+") + NotAny(Word(alphanums))
        PATH_QUESTION = Lit("?") + NotAny(Word(alphanums + "_"))  # ?name is a variable, not a modifier
        PATH_CARET = Lit("^")
        PATH_SLASH = Lit("/")
        PATH_PIPE = Lit("|")
        PATH_EXCLAIM = Lit("!")
        
        # Forward declaration
        path_expression = Forward()
        
        def make_path_iri(iri_val):
            if isinstance(iri_val, IRI):
                return PathIRI(iri=iri_val)
            return PathIRI(iri=IRI(str(iri_val)))
        
        # Grouped path: ( path_expression )
        path_group = (Suppress(LPAREN) + path_expression + Suppress(RPAREN))
        
        # Inverse path: ^iri or ^(path)
        def make_path_inverse(tokens):
            inner = tokens[0]
            if isinstance(inner, IRI):
                inner = make_path_iri(inner)
            return PathInverse(path=inner)
        
        path_inverse = (
            Suppress(PATH_CARET) + (iri | path_group)
        ).set_parse_action(make_path_inverse)
        
        # Negated property set: !(iri|iri|...) or !iri
        def make_path_negated(tokens):
            iris = []
            for t in tokens:
                if isinstance(t, IRI):
                    iris.append(t)
                elif isinstance(t, PathIRI):
                    iris.append(t.iri)
            return PathNegatedPropertySet(iris=tuple(iris))
        
        path_negated = (
            Suppress(PATH_EXCLAIM) + 
            (
                (Suppress(LPAREN) + DelimitedList(iri, delim="|") + Suppress(RPAREN)) |
                iri
            )
        ).set_parse_action(make_path_negated)
        
        # Modified IRI: iri+ or iri* or iri? or iri{n,m}
        def make_path_mod(tokens):
            iri_val = tokens[0]
            mod_str = tokens[1]
            path = make_path_iri(iri_val)
            if mod_str == "*":
                return PathMod(path=path, modifier=PropertyPathModifier.ZERO_OR_MORE)
            elif mod_str == "+":
                return PathMod(path=path, modifier=PropertyPathModifier.ONE_OR_MORE)
            elif mod_str == "?":
                return PathMod(path=path, modifier=PropertyPathModifier.ZERO_OR_ONE)
            return path
        
        # Fixed length modifier: {n}, {n,m}, {n,}
        def make_fixed_length(tokens):
            iri_val = tokens[0]
            min_len = int(tokens[1])
            max_len = None
            # Check if there's a comma and maybe a max value
            if len(tokens) > 2:
                if tokens[2] == ",":
                    if len(tokens) > 3:
                        max_len = int(tokens[3])
                    # else max_len stays None (unbounded)
                else:
                    max_len = min_len  # {n} means exactly n
            else:
                max_len = min_len  # {n} means exactly n
            path = make_path_iri(iri_val)
            return PathFixedLength(path=path, min_length=min_len, max_length=max_len)
        
        # Fixed length path: iri{n} or iri{n,m} or iri{n,}
        fixed_length_modifier = (
            Suppress(Lit("{")) + 
            Word(nums) + 
            Opt(Lit(",") + Opt(Word(nums))) + 
            Suppress(Lit("}"))
        )
        
        path_iri_fixed = (
            iri + fixed_length_modifier
        ).set_parse_action(make_fixed_length)
        
        path_iri_modified = (
            iri + (PATH_STAR | PATH_PLUS | PATH_QUESTION)
        ).set_parse_action(make_path_mod)
        
        # A path element: inverse, negated, fixed-length, modified IRI, or grouped
        path_element = path_inverse | path_negated | path_iri_fixed | path_iri_modified | path_group
        
        # A path step (for sequences): path element or plain IRI
        def wrap_path_step(tokens):
            t = tokens[0]
            if isinstance(t, IRI):
                return make_path_iri(t)
            return t
        
        path_step = (path_element | iri.copy().set_parse_action(wrap_path_step))
        
        # Sequence path: path1/path2/... (requires at least one /)
        def make_path_sequence(tokens):
            paths = list(tokens)
            if len(paths) == 1:
                return paths[0]
            return PathSequence(paths=tuple(paths))
        
        path_sequence = (
            path_step + OneOrMore(Suppress(PATH_SLASH) + path_step)
        ).set_parse_action(make_path_sequence)
        
        # Alternative path: path1|path2|... (requires at least one |)
        def make_path_alternative(tokens):
            paths = list(tokens)
            if len(paths) == 1:
                return paths[0]
            return PathAlternative(paths=tuple(paths))
        
        # Atomic path for alternatives: sequence, element, or plain IRI wrapped
        # We include path_step here to allow plain IRIs in alternatives
        path_atomic = path_sequence | path_element | path_step
        
        path_alternative = (
            path_atomic + OneOrMore(Suppress(PATH_PIPE) + path_atomic)
        ).set_parse_action(make_path_alternative)
        
        # Complete path expression
        path_expression <<= path_alternative | path_atomic
        
        # Predicate: try 'a' keyword, path expression, or term
        predicate_path = a_keyword | path_expression | term
        
        # =================================================================
        # Triple Patterns (with property list and object list support)
        # =================================================================
        
        # SPARQL property lists use:
        # - ; (semicolon) = same subject, different predicate-object pair
        # - , (comma) = same subject and predicate, different object
        
        SEMICOLON = Suppress(Lit(";"))
        
        def make_single_triple(tokens):
            """Create a single triple pattern."""
            pred = tokens[1]
            if isinstance(pred, PathIRI):
                pred = pred.iri
            return TriplePattern(
                subject=tokens[0],
                predicate=pred,
                object=tokens[2]
            )
        
        # Simple triple without property/object lists
        simple_triple = (
            term + predicate_path + term
        ).set_parse_action(make_single_triple)
        
        # Object list: same subject and predicate, multiple objects
        # <s> <p> <o1> , <o2> , <o3>
        object_list = term + ZeroOrMore(COMMA + term)
        
        # Predicate-object pair: predicate followed by object(s)
        predicate_object = predicate_path + object_list
        
        # Predicate-object list: multiple predicate-object pairs separated by ;
        # <p1> <o1> ; <p2> <o2> ; <p3> <o3>
        predicate_object_list = predicate_object + ZeroOrMore(SEMICOLON + predicate_object)
        
        def is_predicate_type(token):
            """Check if a token is a valid predicate type."""
            # Path expressions are always predicates
            if isinstance(token, (PathIRI, PathSequence, PathAlternative, PathInverse, PathMod, PathNegatedPropertySet, PathFixedLength)):
                return True
            # Plain IRIs can be predicates
            if isinstance(token, IRI):
                return True
            # Variables can be predicates (e.g., ?s ?p ?o)
            if isinstance(token, Variable):
                return True
            return False
        
        def is_object_type(token):
            """Check if a token is a valid object type."""
            return isinstance(token, (IRI, Variable, Literal, BlankNode, QuotedTriplePattern))
        
        def normalize_predicate(pred):
            """Normalize predicate - unwrap PathIRI to IRI for simple cases."""
            if isinstance(pred, PathIRI):
                return pred.iri
            return pred
        
        def make_triple_block(tokens):
            """Parse a triple block with optional property/object lists.
            
            Tokens come as a flat list after semicolons/commas are suppressed:
            [subject, pred1, obj1, pred2, obj2, ...]
            
            For property paths like <foaf:knows>+, the predicate is a PathMod.
            
            Expands:
            - ?s <p1> <o1> ; <p2> <o2> . → [(?s, <p1>, <o1>), (?s, <p2>, <o2>)]
            - ?s <p> <o1> , <o2> . → [(?s, <p>, <o1>), (?s, <p>, <o2>)]
            """
            tokens_list = list(tokens)
            if not tokens_list:
                return []
            
            subject = tokens_list[0]
            triples = []
            
            # Process remaining tokens as alternating predicate-objects
            # Since semicolons are suppressed, we get: [pred1, obj1, pred2, obj2, ...]
            i = 1
            while i < len(tokens_list):
                # Get predicate - can be IRI, PathIRI, or other path expressions
                pred = tokens_list[i]
                
                if not is_predicate_type(pred):
                    # Skip non-predicates (shouldn't happen but defensive)
                    i += 1
                    continue
                
                # Normalize simple PathIRI to IRI
                pred = normalize_predicate(pred)
                
                i += 1
                if i >= len(tokens_list):
                    break
                
                # Get object(s) - handle object lists with comma
                while i < len(tokens_list):
                    obj = tokens_list[i]
                    
                    # If it's an object type, create triple
                    if is_object_type(obj):
                        triples.append(TriplePattern(
                            subject=subject,
                            predicate=pred,
                            object=obj
                        ))
                        i += 1
                        
                        # Check if next token is also an object (comma was suppressed)
                        if i < len(tokens_list):
                            next_tok = tokens_list[i]
                            # If next is a path expression (not just IRI), it's a predicate
                            if isinstance(next_tok, (PathIRI, PathSequence, PathAlternative, PathInverse, PathMod, PathNegatedPropertySet, PathFixedLength)):
                                break
                            # If next is IRI, need to peek further to determine if predicate or object
                            # Heuristic: if it's followed by something that could be an object, it's a predicate
                            if isinstance(next_tok, IRI):
                                # Look ahead to see if there's an object after this
                                if i + 1 < len(tokens_list) and is_object_type(tokens_list[i + 1]):
                                    break  # It's a predicate
                            # If next is still an object type (Variable, Literal, etc), continue object list
                            if isinstance(next_tok, (Variable, Literal, BlankNode)):
                                continue  # Continue in object list
                            break
                    else:
                        break
            
            return triples if triples else []
        
        # Full triple block: subject + predicate-object list + optional dot
        triple_block = (
            term + predicate_object_list + Opt(DOT)
        ).set_parse_action(make_triple_block)
        
        # triple_pattern now returns a list of TriplePatterns
        triple_pattern = triple_block
        
        # =================================================================
        # FILTER Expressions
        # =================================================================
        
        # Expression forward declaration
        expression = Forward()
        
        # Function call
        func_name = (
            BOUND | ISIRI | ISBLANK | ISLITERAL | STR | LANG | DATATYPE |
            COALESCE | IF | STRLEN | CONTAINS | STRSTARTS | STRENDS |
            LCASE | UCASE | CONCAT | REPLACE | ABS | ROUND | CEIL | FLOOR | REGEX |
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
        # OPTIONAL and UNION Patterns
        # =================================================================
        
        # Forward declaration for nested group patterns
        group_graph_pattern = Forward()
        
        # Forward declaration for bind_clause (defined later, used in nested patterns)
        bind_clause = Forward()
        
        # Forward declaration for subselect_pattern (nested SELECT)
        subselect_pattern = Forward()
        
        # EXISTS and NOT EXISTS (defined after group_graph_pattern)
        def make_exists_filter(tokens):
            """Create an ExistsExpression filter."""
            negated = str(tokens[0]).upper() == "NOT"
            # Find the WhereClause in the tokens
            patterns_start = 1 if negated else 0
            # The group_graph_pattern returns a list of patterns
            inner_patterns = []
            inner_filters = []
            for t in tokens[patterns_start + 1:]:  # Skip EXISTS keyword
                if isinstance(t, (TriplePattern, QuotedTriplePattern)):
                    inner_patterns.append(t)
                elif isinstance(t, list):
                    for item in t:
                        if isinstance(item, (TriplePattern, QuotedTriplePattern)):
                            inner_patterns.append(item)
                        elif isinstance(item, Filter):
                            inner_filters.append(item)
                elif isinstance(t, Filter):
                    inner_filters.append(t)
            
            inner_where = WhereClause(patterns=inner_patterns, filters=inner_filters)
            return Filter(expression=ExistsExpression(pattern=inner_where, negated=negated))
        
        # OPTIONAL { ... }
        def make_optional(tokens):
            patterns = []
            filters = []
            binds = []
            for token in tokens:
                if isinstance(token, (TriplePattern, QuotedTriplePattern)):
                    patterns.append(token)
                elif isinstance(token, list):
                    for item in token:
                        if isinstance(item, (TriplePattern, QuotedTriplePattern)):
                            patterns.append(item)
                elif isinstance(token, Filter):
                    filters.append(token)
                elif isinstance(token, OptionalPattern):
                    patterns.append(token)
                elif isinstance(token, Bind):
                    binds.append(token)
            return OptionalPattern(patterns=patterns, filters=filters, binds=binds)
        
        optional_pattern = (
            Suppress(OPTIONAL) + LBRACE + ZeroOrMore(triple_pattern | filter_clause | bind_clause) + RBRACE
        ).set_parse_action(make_optional)
        
        # UNION { ... } UNION { ... }
        # A group graph pattern that can participate in UNION
        def make_group_pattern(tokens):
            """Convert a list of patterns/filters/binds into a tuple for UNION alternatives."""
            patterns = []
            filters = []
            binds = []
            for token in tokens:
                if isinstance(token, (TriplePattern, QuotedTriplePattern)):
                    patterns.append(token)
                elif isinstance(token, list):
                    for item in token:
                        if isinstance(item, (TriplePattern, QuotedTriplePattern)):
                            patterns.append(item)
                elif isinstance(token, Filter):
                    filters.append(token)
                elif isinstance(token, OptionalPattern):
                    patterns.append(token)
                elif isinstance(token, Bind):
                    binds.append(token)
            return (patterns, filters, binds)
        
        union_alternative = (
            LBRACE + ZeroOrMore(triple_pattern | filter_clause | optional_pattern | bind_clause) + RBRACE
        ).set_parse_action(make_group_pattern)
        
        def make_union(tokens):
            """Combine UNION alternatives into UnionPattern."""
            alternatives = []
            for token in tokens:
                if isinstance(token, tuple) and len(token) == 3:
                    patterns, filters, binds = token
                    # Store as WhereClause-like structure for full support
                    alternatives.append({
                        'patterns': patterns,
                        'filters': filters,
                        'binds': binds
                    })
            return UnionPattern(alternatives=alternatives)
        
        union_pattern = (
            union_alternative + OneOrMore(Suppress(UNION) + union_alternative)
        ).set_parse_action(make_union)
        
        # =================================================================
        # MINUS Pattern
        # =================================================================
        
        def make_minus(tokens):
            """Create a MINUS pattern for set difference."""
            patterns = []
            filters = []
            for token in tokens:
                if isinstance(token, (TriplePattern, QuotedTriplePattern)):
                    patterns.append(token)
                elif isinstance(token, list):
                    for item in token:
                        if isinstance(item, (TriplePattern, QuotedTriplePattern)):
                            patterns.append(item)
                elif isinstance(token, Filter):
                    filters.append(token)
                elif isinstance(token, OptionalPattern):
                    patterns.append(token)
            return MinusPattern(patterns=patterns, filters=filters)
        
        minus_pattern = (
            Suppress(MINUS) + LBRACE + ZeroOrMore(triple_pattern | filter_clause | bind_clause) + RBRACE
        ).set_parse_action(make_minus)
        
        # =================================================================
        # EXISTS / NOT EXISTS Filter
        # =================================================================
        
        # EXISTS { patterns } or NOT EXISTS { patterns }
        exists_filter_clause = (
            Suppress(FILTER) + (
                (Suppress(NOT) + Suppress(EXISTS) + LBRACE + ZeroOrMore(triple_pattern | filter_clause | bind_clause) + RBRACE) |
                (Suppress(EXISTS) + LBRACE + ZeroOrMore(triple_pattern | filter_clause | bind_clause) + RBRACE)
            )
        ).set_parse_action(lambda t: make_exists_filter(["NOT"] + list(t) if len(list(t)) > 0 and str(t[0]).upper() == "NOT" else list(t)))
        
        def make_exists_filter_inner(tokens, negated=False):
            """Create an ExistsExpression filter from parsed tokens."""
            inner_patterns = []
            inner_filters = []
            for t in tokens:
                if isinstance(t, (TriplePattern, QuotedTriplePattern)):
                    inner_patterns.append(t)
                elif isinstance(t, list):
                    for item in t:
                        if isinstance(item, (TriplePattern, QuotedTriplePattern)):
                            inner_patterns.append(item)
                        elif isinstance(item, Filter):
                            inner_filters.append(item)
                elif isinstance(t, Filter):
                    inner_filters.append(t)
            
            inner_where = WhereClause(patterns=inner_patterns, filters=inner_filters)
            return Filter(expression=ExistsExpression(pattern=inner_where, negated=negated))
        
        not_exists_filter = (
            Suppress(FILTER) + Suppress(NOT) + Suppress(EXISTS) + 
            LBRACE + ZeroOrMore(triple_pattern | filter_clause) + RBRACE
        ).set_parse_action(lambda t: make_exists_filter_inner(t, negated=True))
        
        exists_filter = (
            Suppress(FILTER) + Suppress(EXISTS) + 
            LBRACE + ZeroOrMore(triple_pattern | filter_clause) + RBRACE
        ).set_parse_action(lambda t: make_exists_filter_inner(t, negated=False))
        
        # =================================================================
        # BIND Clause
        # =================================================================
        
        bind_variable = Combine(
            (Lit("?") | Lit("$")) + Word(alphas + "_", alphanums + "_")
        ).set_parse_action(make_variable)
        
        def make_bind(tokens):
            # BIND(expr AS ?var)
            expr = tokens[0]
            var = tokens[1]
            return Bind(expression=expr, variable=var)
        
        bind_clause <<= (
            Suppress(BIND) + LPAREN + 
            (expression | literal | float_literal | integer_literal | variable | iri) + 
            Suppress(AS) + bind_variable + 
            RPAREN
        ).set_parse_action(make_bind)
        
        # =================================================================
        # VALUES Clause
        # =================================================================
        
        values_variable = Combine(
            (Lit("?") | Lit("$")) + Word(alphas + "_", alphanums + "_")
        ).set_parse_action(make_variable)
        
        # Value term (can be UNDEF or a value)
        def make_undef(tokens):
            return None
        
        value_term = (
            UNDEF.set_parse_action(make_undef) |
            iri | literal | float_literal | integer_literal | boolean_literal
        )
        
        # Single variable VALUES: VALUES ?x { 1 2 3 }
        def make_single_values(tokens):
            var = tokens[0]
            bindings = [[v] for v in tokens[1:]]
            return ValuesClause(variables=[var], bindings=bindings)
        
        single_values = (
            Suppress(VALUES) + values_variable + LBRACE + ZeroOrMore(value_term) + RBRACE
        ).set_parse_action(make_single_values)
        
        # Multi-variable VALUES: VALUES (?x ?y) { (1 2) (3 4) }
        def make_value_row(tokens):
            return list(tokens)
        
        value_row = (LPAREN + ZeroOrMore(value_term) + RPAREN).set_parse_action(make_value_row)
        
        def make_multi_values(tokens):
            # First tokens are variables, rest are rows
            vars_list = []
            rows = []
            for token in tokens:
                if isinstance(token, Variable):
                    vars_list.append(token)
                elif isinstance(token, list):
                    rows.append(token)
            return ValuesClause(variables=vars_list, bindings=rows)
        
        multi_values = (
            Suppress(VALUES) + LPAREN + OneOrMore(values_variable) + RPAREN + 
            LBRACE + ZeroOrMore(value_row) + RBRACE
        ).set_parse_action(make_multi_values)
        
        values_clause = multi_values | single_values
        
        # =================================================================
        # GRAPH Pattern
        # =================================================================
        
        # Forward declaration for graph_pattern since where_pattern needs it
        graph_pattern = Forward()
        
        def make_graph_pattern(tokens):
            graph_ref = tokens[0]
            patterns = []
            binds = []
            filters = []
            for token in tokens[1:]:
                if isinstance(token, (TriplePattern, QuotedTriplePattern)):
                    patterns.append(token)
                elif isinstance(token, list):
                    for item in token:
                        if isinstance(item, (TriplePattern, QuotedTriplePattern)):
                            patterns.append(item)
                elif isinstance(token, Bind):
                    binds.append(token)
                elif isinstance(token, Filter):
                    filters.append(token)
            return GraphPattern(graph=graph_ref, patterns=patterns, binds=binds, filters=filters)
        
        graph_pattern <<= (
            Suppress(GRAPH) + (variable | iri) + LBRACE + ZeroOrMore(triple_pattern | filter_clause | bind_clause) + RBRACE
        ).set_parse_action(make_graph_pattern)
        
        # =================================================================
        # WHERE Clause
        # =================================================================
        
        where_pattern = triple_pattern | not_exists_filter | exists_filter | filter_clause | optional_pattern | union_pattern | minus_pattern | bind_clause | values_clause | graph_pattern | subselect_pattern
        
        def make_where_clause(tokens):
            patterns = []
            filters = []
            optional_patterns = []
            union_patterns = []
            minus_patterns = []
            binds = []
            values = None
            graph_patterns = []
            subselects = []
            for token in tokens:
                if isinstance(token, (TriplePattern, QuotedTriplePattern)):
                    patterns.append(token)
                elif isinstance(token, list):
                    # Handle expanded triple blocks (from property/object lists)
                    for item in token:
                        if isinstance(item, (TriplePattern, QuotedTriplePattern)):
                            patterns.append(item)
                elif isinstance(token, Filter):
                    filters.append(token)
                elif isinstance(token, OptionalPattern):
                    optional_patterns.append(token)
                elif isinstance(token, UnionPattern):
                    union_patterns.append(token)
                elif isinstance(token, MinusPattern):
                    minus_patterns.append(token)
                elif isinstance(token, Bind):
                    binds.append(token)
                elif isinstance(token, ValuesClause):
                    values = token
                elif isinstance(token, GraphPattern):
                    graph_patterns.append(token)
                elif isinstance(token, SubSelect):
                    subselects.append(token)
            return WhereClause(
                patterns=patterns, 
                filters=filters, 
                optional_patterns=optional_patterns,
                union_patterns=union_patterns,
                minus_patterns=minus_patterns,
                binds=binds,
                values=values,
                graph_patterns=graph_patterns,
                subselects=subselects
            )
        
        # WHERE clause with optional WHERE keyword (for ASK queries)
        where_clause = (
            Suppress(Opt(WHERE)) + LBRACE + ZeroOrMore(where_pattern) + RBRACE
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
        
        # Aggregate functions
        aggregate_name = COUNT | SUM | AVG | MIN | MAX | GROUP_CONCAT | SAMPLE
        
        # Separator for GROUP_CONCAT
        separator_clause = Suppress(Lit(";")) + Suppress(SEPARATOR) + Suppress(Lit("=")) + (
            QuotedString('"', esc_char='\\') | QuotedString("'", esc_char='\\')
        )
        
        def make_aggregate(tokens):
            func_name = str(tokens[0]).upper()
            distinct = False
            arg = None
            separator = None
            
            for i, t in enumerate(tokens[1:], 1):
                if str(t).upper() == "DISTINCT":
                    distinct = True
                elif t == "*":
                    arg = None  # COUNT(*)
                elif isinstance(t, Variable):
                    arg = t
                elif isinstance(t, str) and t not in ("DISTINCT", "*"):
                    separator = t
            
            return AggregateExpression(
                function=func_name,
                argument=arg,
                distinct=distinct,
                separator=separator
            )
        
        # COUNT(*) or COUNT(DISTINCT ?var) or COUNT(?var)
        aggregate_arg = (
            Opt(DISTINCT) + (STAR | select_variable) + Opt(separator_clause)
        )
        
        aggregate_expr = (
            aggregate_name + LPAREN + aggregate_arg + RPAREN
        ).set_parse_action(make_aggregate)
        
        # Aggregate with alias: (COUNT(?x) AS ?count)
        def make_aggregate_with_alias(tokens):
            agg = tokens[0]
            if len(tokens) > 1 and isinstance(tokens[1], Variable):
                agg.alias = tokens[1]
            return agg
        
        aliased_aggregate = (
            LPAREN + aggregate_expr + Suppress(AS) + select_variable + RPAREN
        ).set_parse_action(make_aggregate_with_alias)
        
        # Select expression: variable or (aggregate AS ?alias)
        select_expr = aliased_aggregate | aggregate_expr | select_variable
        
        # Variable list or *
        def make_star(tokens):
            return []
        
        select_vars = (
            STAR.set_parse_action(make_star) |
            OneOrMore(select_expr)
        )
        
        # GROUP BY clause
        group_by_variable = Combine(
            (Lit("?") | Lit("$")) + Word(alphas + "_", alphanums + "_")
        ).set_parse_action(make_variable)
        
        def make_group_by_marker(tokens):
            """Mark this as a GROUP BY list."""
            return ("GROUP_BY", list(tokens))
        
        group_by_clause = (
            Suppress(GROUP) + Suppress(BY) + OneOrMore(group_by_variable)
        ).set_parse_action(make_group_by_marker)
        
        # HAVING clause
        having_clause = Suppress(HAVING) + LPAREN + expression + RPAREN
        
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
        
        # FROM clause for dataset specification
        from_clause = Suppress(FROM) + iri
        from_named_clause = Suppress(FROM) + Suppress(NAMED) + iri
        
        def make_select_query(tokens):
            prefixes = {}
            variables = []
            distinct = False
            where = WhereClause()
            limit = None
            offset = None
            order_by = []
            group_by = []
            having = None
            as_of = None
            from_graphs = []
            from_named_graphs = []
            
            for token in tokens:
                if isinstance(token, datetime):
                    as_of = token
                elif isinstance(token, tuple) and len(token) == 2:
                    if token[0] == "GROUP_BY":
                        # This is a GROUP BY clause
                        group_by = token[1]
                    elif token[0] == "FROM":
                        # This is a FROM clause
                        from_graphs.append(token[1])
                    elif token[0] == "FROM_NAMED":
                        # This is a FROM NAMED clause
                        from_named_graphs.append(token[1])
                    elif isinstance(token[0], str) and isinstance(token[1], str):
                        # This is a prefix declaration
                        prefixes[token[0]] = token[1]
                    elif isinstance(token[0], Variable):
                        # This is an order by condition
                        order_by.append(token)
                elif token == "DISTINCT":
                    distinct = True
                elif isinstance(token, AggregateExpression):
                    variables.append(token)
                elif isinstance(token, Variable):
                    variables.append(token)
                elif isinstance(token, (Comparison, LogicalExpression, FunctionCall)):
                    # HAVING expression
                    having = token
                elif isinstance(token, pp.ParseResults) or isinstance(token, list):
                    # Check what's in the list
                    token_list = list(token)
                    if token_list and isinstance(token_list[0], (Variable, AggregateExpression)):
                        variables = token_list
                    elif token_list and isinstance(token_list[0], tuple):
                        # Could be order_by or group_by marker
                        if token_list[0][0] == "GROUP_BY":
                            group_by = token_list[0][1]
                        else:
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
                group_by=group_by,
                having=having,
                as_of=as_of,
                from_graphs=from_graphs,
                from_named_graphs=from_named_graphs,
            )
        
        def make_distinct(tokens):
            return "DISTINCT"
        
        def make_from_clause(tokens):
            return ("FROM", tokens[0])
        
        def make_from_named_clause(tokens):
            return ("FROM_NAMED", tokens[0])
        
        # =================================================================
        # SubSelect (nested SELECT in WHERE clause)
        # =================================================================
        
        def make_subselect(tokens):
            """Build a SubSelect AST node from parsed tokens."""
            variables = []
            where = WhereClause()
            distinct = False
            limit = None
            offset = None
            group_by = []
            having = None
            order_by = []
            
            for token in tokens:
                if isinstance(token, tuple) and len(token) == 2:
                    if token[0] == "GROUP_BY":
                        group_by = token[1]
                    elif isinstance(token[0], Variable):
                        # This is an order by condition
                        order_by.append(token)
                elif token == "DISTINCT":
                    distinct = True
                elif isinstance(token, AggregateExpression):
                    variables.append(token)
                elif isinstance(token, Variable):
                    variables.append(token)
                elif isinstance(token, (Comparison, LogicalExpression, FunctionCall)):
                    # HAVING expression
                    having = Filter(expression=token)
                elif isinstance(token, pp.ParseResults) or isinstance(token, list):
                    token_list = list(token)
                    if token_list and isinstance(token_list[0], (Variable, AggregateExpression)):
                        variables = token_list
                    elif token_list and isinstance(token_list[0], tuple):
                        if token_list[0][0] == "GROUP_BY":
                            group_by = token_list[0][1]
                        else:
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
            
            return SubSelect(
                variables=variables,
                where=where,
                distinct=distinct,
                group_by=group_by,
                having=having,
                order_by=[],  # Order by in subselect is complex, skip for now
                limit=limit,
                offset=offset
            )
        
        # SubSelect grammar - enclosed in braces within WHERE clause
        subselect_pattern <<= (
            LBRACE +
            Suppress(SELECT) +
            Opt(DISTINCT.set_parse_action(make_distinct)) +
            Group(select_vars) +
            where_clause +
            Opt(Group(group_by_clause)) +
            Opt(having_clause) +
            Opt(limit_clause) +
            Opt(offset_clause) +
            RBRACE
        ).set_parse_action(make_subselect)
        
        select_query = (
            ZeroOrMore(prefix_decl) +
            Suppress(SELECT) +
            Opt(DISTINCT.set_parse_action(make_distinct)) +
            Group(select_vars) +
            ZeroOrMore(from_named_clause.set_parse_action(make_from_named_clause) | from_clause.set_parse_action(make_from_clause)) +
            where_clause +
            Opt(Group(group_by_clause)) +
            Opt(having_clause) +
            Opt(Group(order_clause)) +
            Opt(limit_clause) +
            Opt(offset_clause) +
            Opt(as_of_clause)
        ).set_parse_action(make_select_query)
        
        # =================================================================
        # ASK Query
        # =================================================================
        
        def make_ask_query(tokens):
            prefixes = {}
            where = WhereClause()
            as_of = None
            for token in tokens:
                if isinstance(token, datetime):
                    as_of = token
                elif isinstance(token, tuple) and len(token) == 2 and isinstance(token[0], str):
                    prefixes[token[0]] = token[1]
                elif isinstance(token, WhereClause):
                    where = token
            return AskQuery(prefixes=prefixes, where=where, as_of=as_of)
        
        ask_query = (
            ZeroOrMore(prefix_decl) +
            Suppress(ASK) +
            where_clause +
            Opt(as_of_clause)
        ).set_parse_action(make_ask_query)
        
        # =================================================================
        # INSERT DATA Update (with full Turtle syntax support)
        # =================================================================
        
        # Special prefixed name for Turtle/INSERT DATA that allows slashes
        # (unlike the main prefixed_name which doesn't, to avoid conflicts with property paths)
        turtle_pname_local = Word(alphanums + "_.-/")
        
        def make_turtle_prefixed_name(tokens):
            return IRI(tokens[0])
        
        turtle_prefixed_name = Combine(pname_ns + Opt(turtle_pname_local)).set_parse_action(make_turtle_prefixed_name)
        turtle_iri = full_iri | turtle_prefixed_name
        turtle_iri_or_a = turtle_iri | a_keyword
        
        # Ground term for INSERT DATA (no variables, allows path-like prefixed names)
        # Also includes quoted_triple for RDF-Star annotation support
        ground_term = quoted_triple | turtle_iri | literal | float_literal | integer_literal | boolean_literal | blank_node
        
        # Turtle-style triple parsing with semicolons and commas
        # Semicolon (;) = same subject, new predicate-object pair
        # Comma (,) = same subject and predicate, new object
        # Dot (.) = end of triple block
        
        SEMICOLON = Suppress(Lit(";"))
        
        def parse_turtle_triples(tokens):
            """Parse Turtle-style triples into a list of TriplePattern objects.
            
            Handles:
            - Simple triples: <s> <p> <o> .
            - Property lists: <s> <p1> <o1> ; <p2> <o2> .
            - Object lists: <s> <p> <o1> , <o2> , <o3> .
            - Combined: <s> <p1> <o1> , <o2> ; <p2> <o3> .
            - RDF-Star: << s p o >> <annotation_pred> <value> .
            """
            triples = []
            token_list = list(tokens)
            
            i = 0
            current_subject = None
            current_predicate = None
            
            while i < len(token_list):
                token = token_list[i]
                
                # Skip punctuation strings if they slip through
                if isinstance(token, str) and token in '.;,':
                    i += 1
                    continue
                
                # If we have a ground term and no subject yet, or after a dot
                if current_subject is None:
                    if isinstance(token, (IRI, Literal, BlankNode, QuotedTriplePattern)):
                        current_subject = token
                        current_predicate = None
                        i += 1
                        continue
                
                # If we have subject but no predicate
                if current_subject is not None and current_predicate is None:
                    if isinstance(token, IRI):
                        current_predicate = token
                        i += 1
                        continue
                
                # If we have subject and predicate, next is object
                if current_subject is not None and current_predicate is not None:
                    if isinstance(token, (IRI, Literal, BlankNode, QuotedTriplePattern)):
                        triples.append(TriplePattern(
                            subject=current_subject,
                            predicate=current_predicate,
                            object=token
                        ))
                        i += 1
                        
                        # Check what comes next
                        if i < len(token_list):
                            next_token = token_list[i]
                            if isinstance(next_token, str):
                                if next_token == ',':
                                    # Same subject and predicate, new object
                                    i += 1
                                    continue
                                elif next_token == ';':
                                    # Same subject, new predicate
                                    current_predicate = None
                                    i += 1
                                    continue
                                elif next_token == '.':
                                    # End of this subject block
                                    current_subject = None
                                    current_predicate = None
                                    i += 1
                                    continue
                        continue
                
                i += 1
            
            return triples
        
        # Object list: <o1> , <o2> , <o3>
        turtle_object = ground_term
        turtle_object_list = turtle_object + ZeroOrMore(Lit(",") + turtle_object)
        
        # Predicate-object: <p> <o1> , <o2> (use turtle_iri_or_a for path-like prefixed names)
        turtle_predicate = turtle_iri_or_a
        turtle_predicate_object = turtle_predicate + turtle_object_list
        turtle_predicate_object_list = turtle_predicate_object + ZeroOrMore(Lit(";") + Opt(turtle_predicate_object))
        
        # Full triple block: <s> <p1> <o1> ; <p2> <o2> , <o3> .
        turtle_triple_block = ground_term + turtle_predicate_object_list + Opt(Lit("."))
        
        # Multiple triple blocks
        turtle_triples = ZeroOrMore(turtle_triple_block)
        turtle_triples.set_parse_action(parse_turtle_triples)
        
        # Ground triple for INSERT DATA (simple form - backward compatibility)
        def make_ground_triple(tokens):
            return TriplePattern(
                subject=tokens[0],
                predicate=tokens[1],
                object=tokens[2],
            )
        
        ground_triple = (
            ground_term + ground_term + ground_term + Opt(DOT)
        ).set_parse_action(make_ground_triple)
        
        # INSERT DATA { triples } - supports full Turtle syntax
        def make_insert_data_query(tokens):
            prefixes = {}
            triples = []
            graph = None
            
            for token in tokens:
                if isinstance(token, tuple) and len(token) == 2 and isinstance(token[0], str):
                    prefixes[token[0]] = token[1]
                elif isinstance(token, TriplePattern):
                    triples.append(token)
                elif isinstance(token, IRI):
                    graph = token
                elif isinstance(token, list) or isinstance(token, pp.ParseResults):
                    for item in token:
                        if isinstance(item, TriplePattern):
                            triples.append(item)
            
            return InsertDataQuery(prefixes=prefixes, triples=triples, graph=graph)
        
        # Use turtle_triples for full Turtle syntax support
        insert_data_body = LBRACE + turtle_triples + RBRACE
        
        insert_data_query = (
            ZeroOrMore(prefix_decl) +
            Suppress(INSERT) +
            Suppress(DATA) +
            insert_data_body
        ).set_parse_action(make_insert_data_query)
        
        # =================================================================
        # DELETE DATA Update
        # =================================================================
        
        def make_delete_data_query(tokens):
            prefixes = {}
            triples = []
            graph = None
            
            for token in tokens:
                if isinstance(token, tuple) and len(token) == 2 and isinstance(token[0], str):
                    prefixes[token[0]] = token[1]
                elif isinstance(token, TriplePattern):
                    triples.append(token)
                elif isinstance(token, list) or isinstance(token, pp.ParseResults):
                    for item in token:
                        if isinstance(item, TriplePattern):
                            triples.append(item)
            
            return DeleteDataQuery(prefixes=prefixes, triples=triples, graph=graph)
        
        delete_data_query = (
            ZeroOrMore(prefix_decl) +
            Suppress(DELETE) +
            Suppress(DATA) +
            insert_data_body
        ).set_parse_action(make_delete_data_query)
        
        # =================================================================
        # DELETE WHERE Update
        # =================================================================
        
        def make_delete_where_query(tokens):
            prefixes = {}
            where = WhereClause()
            graph = None
            
            for token in tokens:
                if isinstance(token, tuple) and len(token) == 2 and isinstance(token[0], str):
                    prefixes[token[0]] = token[1]
                elif isinstance(token, WhereClause):
                    where = token
            
            return DeleteWhereQuery(prefixes=prefixes, where=where, graph=graph)
        
        delete_where_query = (
            ZeroOrMore(prefix_decl) +
            Suppress(DELETE) +
            where_clause
        ).set_parse_action(make_delete_where_query)
        
        # =================================================================
        # DELETE/INSERT WHERE (Modify) Update
        # =================================================================
        
        # Template for DELETE/INSERT patterns (can contain variables)
        template_triple = (
            term + term + term + Opt(DOT)
        ).set_parse_action(make_ground_triple)
        
        # DELETE { patterns } clause (patterns in braces, not WHERE keyword)
        delete_template = LBRACE + ZeroOrMore(template_triple) + RBRACE
        
        # INSERT { patterns } clause
        insert_template = LBRACE + ZeroOrMore(template_triple) + RBRACE
        
        def make_modify_query(tokens):
            prefixes = {}
            delete_patterns = []
            insert_patterns = []
            where = WhereClause()
            
            # Track which section we're in
            # Tokens will be structured as: [prefixes...], [delete_patterns...], [insert_patterns...], WhereClause
            section = "prefixes"
            
            for token in tokens:
                if isinstance(token, tuple) and len(token) == 2 and isinstance(token[0], str):
                    prefixes[token[0]] = token[1]
                elif token == "DELETE_SECTION":
                    section = "delete"
                elif token == "INSERT_SECTION":
                    section = "insert"
                elif isinstance(token, WhereClause):
                    where = token
                elif isinstance(token, TriplePattern):
                    if section == "delete":
                        delete_patterns.append(token)
                    elif section == "insert":
                        insert_patterns.append(token)
                elif isinstance(token, pp.ParseResults):
                    for item in token:
                        if isinstance(item, TriplePattern):
                            if section == "delete":
                                delete_patterns.append(item)
                            elif section == "insert":
                                insert_patterns.append(item)
            
            return ModifyQuery(
                prefixes=prefixes,
                delete_patterns=delete_patterns,
                insert_patterns=insert_patterns,
                where=where
            )
        
        # DELETE { } INSERT { } WHERE { } - full modify query
        # We need markers to distinguish delete vs insert patterns
        delete_section = (
            Suppress(DELETE) + 
            pp.Literal("{").suppress().set_parse_action(lambda: "DELETE_SECTION") +
            ZeroOrMore(template_triple) + 
            Suppress(RBRACE)
        )
        
        insert_section = (
            Suppress(INSERT) + 
            pp.Literal("{").suppress().set_parse_action(lambda: "INSERT_SECTION") +
            ZeroOrMore(template_triple) + 
            Suppress(RBRACE)
        )
        
        # Modify query with both DELETE and INSERT (or just one)
        # Must have at least one of DELETE or INSERT followed by WHERE
        modify_query = (
            ZeroOrMore(prefix_decl) +
            Opt(delete_section) +
            Opt(insert_section) +
            where_clause
        ).set_parse_action(make_modify_query)
        
        # =================================================================
        # DESCRIBE Query
        # =================================================================
        
        describe_resource = iri | variable
        
        def make_describe_query(tokens):
            prefixes = {}
            resources = []
            where = None
            
            for token in tokens:
                if isinstance(token, tuple) and len(token) == 2 and isinstance(token[0], str):
                    prefixes[token[0]] = token[1]
                elif isinstance(token, (IRI, Variable)):
                    resources.append(token)
                elif isinstance(token, WhereClause):
                    where = token
            
            return DescribeQuery(prefixes=prefixes, resources=resources, where=where)
        
        describe_query = (
            ZeroOrMore(prefix_decl) +
            Suppress(DESCRIBE) +
            OneOrMore(describe_resource) +
            Opt(where_clause)
        ).set_parse_action(make_describe_query)
        
        # =================================================================
        # CONSTRUCT Query
        # =================================================================
        
        construct_template = LBRACE + ZeroOrMore(triple_pattern) + RBRACE
        
        def make_construct_query(tokens):
            prefixes = {}
            template = []
            where = WhereClause()
            
            for token in tokens:
                if isinstance(token, tuple) and len(token) == 2 and isinstance(token[0], str):
                    prefixes[token[0]] = token[1]
                elif isinstance(token, TriplePattern):
                    template.append(token)
                elif isinstance(token, WhereClause):
                    where = token
            
            return ConstructQuery(prefixes=prefixes, template=template, where=where)
        
        construct_query = (
            ZeroOrMore(prefix_decl) +
            Suppress(CONSTRUCT) +
            construct_template +
            where_clause
        ).set_parse_action(make_construct_query)
        
        # =================================================================
        # Graph Management Queries
        # =================================================================
        
        # CREATE [SILENT] GRAPH <uri>
        def make_create_graph(tokens):
            silent = False
            graph_uri = None
            for token in tokens:
                if token == "SILENT":
                    silent = True
                elif isinstance(token, IRI):
                    graph_uri = token
            return CreateGraphQuery(prefixes={}, graph_uri=graph_uri, silent=silent)
        
        create_graph_query = (
            Suppress(CREATE) + Opt(SILENT.set_parse_action(lambda: "SILENT")) +
            Suppress(GRAPH) + iri
        ).set_parse_action(make_create_graph)
        
        # DROP [SILENT] (GRAPH <uri> | DEFAULT | NAMED | ALL)
        def make_drop_graph(tokens):
            silent = False
            graph_uri = None
            target = "graph"
            for token in tokens:
                if token == "SILENT":
                    silent = True
                elif token == "DEFAULT":
                    target = "default"
                elif token == "NAMED":
                    target = "named"
                elif token == "ALL":
                    target = "all"
                elif isinstance(token, IRI):
                    graph_uri = token
            return DropGraphQuery(prefixes={}, graph_uri=graph_uri, target=target, silent=silent)
        
        drop_target = (
            (Suppress(GRAPH) + iri) |
            DEFAULT.set_parse_action(lambda: "DEFAULT") |
            NAMED.set_parse_action(lambda: "NAMED") |
            ALL.set_parse_action(lambda: "ALL")
        )
        drop_graph_query = (
            Suppress(DROP) + Opt(SILENT.set_parse_action(lambda: "SILENT")) + drop_target
        ).set_parse_action(make_drop_graph)
        
        # CLEAR [SILENT] (GRAPH <uri> | DEFAULT | NAMED | ALL)
        def make_clear_graph(tokens):
            silent = False
            graph_uri = None
            target = "graph"
            for token in tokens:
                if token == "SILENT":
                    silent = True
                elif token == "DEFAULT":
                    target = "default"
                elif token == "NAMED":
                    target = "named"
                elif token == "ALL":
                    target = "all"
                elif isinstance(token, IRI):
                    graph_uri = token
            return ClearGraphQuery(prefixes={}, graph_uri=graph_uri, target=target, silent=silent)
        
        clear_graph_query = (
            Suppress(CLEAR) + Opt(SILENT.set_parse_action(lambda: "SILENT")) + drop_target
        ).set_parse_action(make_clear_graph)
        
        # LOAD [SILENT] <source> [INTO GRAPH <dest>]
        def make_load(tokens):
            silent = False
            source_uri = None
            graph_uri = None
            for token in tokens:
                if token == "SILENT":
                    silent = True
                elif isinstance(token, IRI):
                    if source_uri is None:
                        source_uri = token
                    else:
                        graph_uri = token
            return LoadQuery(prefixes={}, source_uri=source_uri, graph_uri=graph_uri, silent=silent)
        
        load_query = (
            Suppress(LOAD) + Opt(SILENT.set_parse_action(lambda: "SILENT")) +
            iri + Opt(Suppress(INTO) + Suppress(GRAPH) + iri)
        ).set_parse_action(make_load)
        
        # COPY/MOVE/ADD [SILENT] (DEFAULT | GRAPH <uri>) TO (DEFAULT | GRAPH <uri>)
        def make_graph_transfer(operation):
            def action(tokens):
                silent = False
                source_graph = None
                dest_graph = None
                source_is_default = False
                
                token_list = list(tokens)
                i = 0
                while i < len(token_list):
                    token = token_list[i]
                    if token == "SILENT":
                        silent = True
                    elif token == "DEFAULT":
                        if source_graph is None and not source_is_default:
                            source_is_default = True
                        # dest_graph would be set via IRI
                    elif isinstance(token, IRI):
                        if source_graph is None and not source_is_default:
                            source_graph = token
                        else:
                            dest_graph = token
                    i += 1
                
                if operation == "COPY":
                    return CopyGraphQuery(
                        prefixes={}, source_graph=source_graph, dest_graph=dest_graph,
                        silent=silent, source_is_default=source_is_default
                    )
                elif operation == "MOVE":
                    return MoveGraphQuery(
                        prefixes={}, source_graph=source_graph, dest_graph=dest_graph,
                        silent=silent, source_is_default=source_is_default
                    )
                else:  # ADD
                    return AddGraphQuery(
                        prefixes={}, source_graph=source_graph, dest_graph=dest_graph,
                        silent=silent, source_is_default=source_is_default
                    )
            return action
        
        graph_ref = (
            DEFAULT.set_parse_action(lambda: "DEFAULT") |
            (Suppress(GRAPH) + iri)
        )
        
        copy_query = (
            Suppress(COPY) + Opt(SILENT.set_parse_action(lambda: "SILENT")) +
            graph_ref + Suppress(TO) + graph_ref
        ).set_parse_action(make_graph_transfer("COPY"))
        
        move_query = (
            Suppress(MOVE) + Opt(SILENT.set_parse_action(lambda: "SILENT")) +
            graph_ref + Suppress(TO) + graph_ref
        ).set_parse_action(make_graph_transfer("MOVE"))
        
        add_query = (
            Suppress(ADD) + Opt(SILENT.set_parse_action(lambda: "SILENT")) +
            graph_ref + Suppress(TO) + graph_ref
        ).set_parse_action(make_graph_transfer("ADD"))
        
        # =================================================================
        # Top-level Query
        # =================================================================
        
        # Note: Order matters - more specific patterns must come first
        # modify_query must come before delete_where_query because:
        #   DELETE { ... } WHERE { ... } should match modify_query
        #   DELETE WHERE { ... } should match delete_where_query
        # delete_data_query must come before delete_where_query (DATA keyword distinguishes)
        self.query = (
            select_query | ask_query | describe_query | construct_query | 
            insert_data_query | delete_data_query | modify_query | delete_where_query |
            create_graph_query | drop_graph_query | clear_graph_query |
            load_query | copy_query | move_query | add_query
        )
        
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


@lru_cache(maxsize=256)
def _cached_parse(query_string: str) -> Query:
    """
    Internal cached parse function.
    
    Uses LRU cache to avoid re-parsing identical query strings.
    Cache size of 256 balances memory usage with hit rate for typical workloads.
    """
    global _parser
    if _parser is None:
        _parser = SPARQLStarParser()
    return _parser.parse(query_string)


def parse_query(query_string: str) -> Query:
    """
    Parse a SPARQL-Star query string.
    
    This function uses LRU caching to avoid re-parsing identical queries.
    For repeated execution of the same query (common in benchmarks and
    production workloads), this provides significant speedup.
    
    Args:
        query_string: The SPARQL-Star query to parse
        
    Returns:
        Parsed Query AST
    """
    return _cached_parse(query_string)


def clear_query_cache():
    """Clear the query parsing cache."""
    _cached_parse.cache_clear()


def get_query_cache_info():
    """Get cache statistics (hits, misses, size, maxsize)."""
    return _cached_parse.cache_info()
