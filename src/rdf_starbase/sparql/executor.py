
"""
SPARQL-Star Query Executor using Polars.

Translates SPARQL-Star AST to Polars operations for blazingly fast execution.

Includes internal optimizations for provenance queries that map standard
SPARQL-Star patterns like << ?s ?p ?o >> prov:value ?conf to efficient
columnar access.

Supported provenance vocabularies:
- PROV-O: W3C Provenance Ontology (prov:wasAttributedTo, prov:value, etc.)
- DQV: Data Quality Vocabulary (dqv:hasQualityMeasurement)
- PAV: Provenance, Authoring and Versioning (pav:createdBy, pav:authoredBy)
- DCAT: Data Catalog Vocabulary (dcat:accessURL, etc.)

When inserting RDF-Star annotations like:
    << ex:s ex:p ex:o >> prov:wasAttributedTo "IMDb" .
    << ex:s ex:p ex:o >> prov:value 0.95 .

The executor recognizes these predicates and maps them to internal assertion
metadata (source, confidence) rather than creating separate triples.
"""

from typing import Any, Optional, Union, TYPE_CHECKING
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl

from rdf_starbase.sparql.ast import (
    Query, SelectQuery, AskQuery, InsertDataQuery, DeleteDataQuery,
    DeleteWhereQuery, ModifyQuery,
    DescribeQuery, ConstructQuery,
    CreateGraphQuery, DropGraphQuery, ClearGraphQuery,
    LoadQuery, CopyGraphQuery, MoveGraphQuery, AddGraphQuery,
    TriplePattern, QuotedTriplePattern,
    OptionalPattern, UnionPattern, GraphPattern,
    Variable, IRI, Literal, BlankNode,
    Filter, Comparison, LogicalExpression, FunctionCall,
    AggregateExpression, Bind, ValuesClause,
    ComparisonOp, LogicalOp,
    WhereClause,
    Term,
    ExistsExpression, SubSelect,
)
from rdf_starbase.models import ProvenanceContext

if TYPE_CHECKING:
    from rdf_starbase.store import TripleStore


# =============================================================================
# Provenance Predicate Mappings
# =============================================================================
# These predicates, when used in RDF-Star annotations, are recognized and
# mapped to internal assertion metadata fields rather than stored as
# separate triples.

# Maps predicate IRIs to internal field names
PROVENANCE_SOURCE_PREDICATES = {
    # PROV-O - W3C Provenance Ontology
    "http://www.w3.org/ns/prov#wasAttributedTo",
    "http://www.w3.org/ns/prov#wasDerivedFrom",
    "http://www.w3.org/ns/prov#wasGeneratedBy",
    "http://www.w3.org/ns/prov#hadPrimarySource",
    # PAV - Provenance, Authoring and Versioning
    "http://purl.org/pav/createdBy",
    "http://purl.org/pav/authoredBy", 
    "http://purl.org/pav/importedFrom",
    "http://purl.org/pav/retrievedFrom",
    "http://purl.org/pav/sourceAccessedAt",
    # Dublin Core
    "http://purl.org/dc/terms/source",
    "http://purl.org/dc/elements/1.1/source",
    # Schema.org
    "http://schema.org/isBasedOn",
    "http://schema.org/citation",
    # Custom RDF-StarBase
    "http://rdf-starbase.io/source",
    "source",  # Short form
}

PROVENANCE_CONFIDENCE_PREDICATES = {
    # PROV-O 
    "http://www.w3.org/ns/prov#value",
    # DQV - Data Quality Vocabulary
    "http://www.w3.org/ns/dqv#hasQualityMeasurement",
    "http://www.w3.org/ns/dqv#value",
    # Schema.org
    "http://schema.org/ratingValue",
    # Custom RDF-StarBase
    "http://rdf-starbase.io/confidence",
    "confidence",  # Short form
}

PROVENANCE_TIMESTAMP_PREDICATES = {
    # PROV-O
    "http://www.w3.org/ns/prov#generatedAtTime",
    "http://www.w3.org/ns/prov#invalidatedAtTime",
    # PAV
    "http://purl.org/pav/createdOn",
    "http://purl.org/pav/authoredOn",
    "http://purl.org/pav/lastRefreshedOn",
    # Dublin Core
    "http://purl.org/dc/terms/created",
    "http://purl.org/dc/terms/modified",
    # Custom
    "http://rdf-starbase.io/timestamp",
    "timestamp",
}

# Legacy map for query optimization (reading provenance)
PROV_PREDICATE_MAP = {
    "http://www.w3.org/ns/prov#value": "confidence",
    "http://www.w3.org/ns/prov#wasDerivedFrom": "source",
    "http://www.w3.org/ns/prov#generatedAtTime": "timestamp",
    "http://www.w3.org/ns/prov#wasGeneratedBy": "process",
    "prov:value": "confidence",
    "prov:wasDerivedFrom": "source",
    "prov:generatedAtTime": "timestamp",
    "prov:wasGeneratedBy": "process",
}


# Configuration for parallel execution
# Note: Parallel execution is primarily beneficial for I/O-bound operations
# (e.g., federated SERVICE queries). For local CPU-bound Polars operations,
# Python's GIL limits benefits and Polars already parallelizes internally.
_PARALLEL_THRESHOLD = 3  # Minimum patterns to trigger parallel execution
_MAX_WORKERS = 4  # Maximum parallel workers


class SPARQLExecutor:
    """
    Executes SPARQL-Star queries against a TripleStore.
    
    Translation strategy:
    - Each TriplePattern becomes a filtered view of the DataFrame
    - Variables become column selections
    - Joins are performed for patterns sharing variables
    - Filters become Polars filter expressions
    - Uses lazy evaluation for query optimization
    
    Performance features:
    - Query plan caching (via parser)
    - Short-circuit on non-existent terms
    - Parallel execution for independent patterns (opt-in, useful for federated queries)
    """
    
    def __init__(self, store: "TripleStore", parallel: bool = False):
        """
        Initialize executor with a triple store.
        
        Args:
            store: The TripleStore to query
            parallel: If True, execute independent patterns in parallel.
                     Default False (Polars already parallelizes internally).
                     Set True for federated/SERVICE queries.
        """
        self.store = store
        self._var_counter = 0
        self._parallel = parallel
    
    def _get_pattern_variables(self, pattern: TriplePattern) -> set[str]:
        """Extract variable names from a triple pattern."""
        vars_set = set()
        if isinstance(pattern.subject, Variable):
            vars_set.add(pattern.subject.name)
        if isinstance(pattern.predicate, Variable):
            vars_set.add(pattern.predicate.name)
        if isinstance(pattern.object, Variable):
            vars_set.add(pattern.object.name)
        elif isinstance(pattern.object, QuotedTriplePattern):
            # Handle quoted triple variables
            qt = pattern.object
            if isinstance(qt.subject, Variable):
                vars_set.add(qt.subject.name)
            if isinstance(qt.predicate, Variable):
                vars_set.add(qt.predicate.name)
            if isinstance(qt.object, Variable):
                vars_set.add(qt.object.name)
        return vars_set
    
    def _extract_explicit_bindings(
        self, 
        where: WhereClause, 
        prefixes: dict[str, str]
    ) -> dict[str, str]:
        """
        Extract variable-to-explicit-term bindings from WHERE patterns.
        
        This handles the case where a SELECT variable corresponds to an
        explicit IRI in the WHERE clause pattern positions.
        
        Example: SELECT ?s ?p ?o WHERE { ?s <http://example.org/worksFor> ?o }
        Returns: {"p": "http://example.org/worksFor"}
        
        Args:
            where: The WHERE clause to analyze
            prefixes: Prefix mappings for resolving prefixed IRIs
            
        Returns:
            Dict mapping variable names to their explicit IRI values
        """
        bindings = {}
        
        # Standard variable names for triple positions
        position_vars = {"s": "subject", "p": "predicate", "o": "object"}
        
        for pattern in where.patterns:
            if isinstance(pattern, TriplePattern):
                # Check each position for explicit terms
                for var_name, position in [("s", pattern.subject), 
                                            ("p", pattern.predicate), 
                                            ("o", pattern.object)]:
                    if isinstance(position, IRI):
                        # This position has an explicit IRI
                        resolved = self._resolve_term(position, prefixes)
                        bindings[var_name] = resolved
                    elif isinstance(position, Literal):
                        bindings[var_name] = str(position.value)
        
        return bindings
    
    def _find_independent_groups(
        self, patterns: list[tuple[int, TriplePattern, Any]]
    ) -> list[list[tuple[int, TriplePattern, Any]]]:
        """
        Group patterns into independent sets that can be executed in parallel.
        
        Two patterns are independent if they share no variables.
        Returns a list of groups where patterns within each group share variables.
        """
        if not patterns:
            return []
        
        # Build adjacency based on shared variables
        n = len(patterns)
        groups = []
        visited = [False] * n
        
        for i in range(n):
            if visited[i]:
                continue
            
            # Start a new group with this pattern
            group = [patterns[i]]
            visited[i] = True
            group_vars = self._get_pattern_variables(patterns[i][1])
            
            # Find all patterns that share variables with this group
            changed = True
            while changed:
                changed = False
                for j in range(n):
                    if visited[j]:
                        continue
                    pattern_vars = self._get_pattern_variables(patterns[j][1])
                    if group_vars & pattern_vars:  # Shared variables
                        group.append(patterns[j])
                        group_vars |= pattern_vars
                        visited[j] = True
                        changed = True
            
            groups.append(group)
        
        return groups
    
    def _execute_pattern_group(
        self,
        group: list[tuple[int, TriplePattern, Any]],
        prefixes: dict[str, str],
        as_of: Optional[datetime],
        from_graphs: Optional[list[str]],
    ) -> pl.DataFrame:
        """
        Execute a group of related patterns (patterns sharing variables).
        
        This is used by parallel execution to process independent pattern groups
        in separate threads.
        """
        result_df: Optional[pl.DataFrame] = None
        
        for i, pattern, prov_binding in group:
            pattern_df = self._execute_pattern(pattern, prefixes, i, as_of=as_of, from_graphs=from_graphs)
            
            # Apply provenance binding if present
            if prov_binding:
                obj_var_name, col_name, pred_var_name = prov_binding
                if col_name != "*":
                    prov_col = f"_prov_{i}_{col_name}"
                    if prov_col in pattern_df.columns:
                        pattern_df = pattern_df.with_columns(
                            pl.col(prov_col).alias(obj_var_name)
                        )
            
            if result_df is None:
                result_df = pattern_df
            else:
                # Join patterns that share variables
                shared_cols = set(result_df.columns) & set(pattern_df.columns)
                shared_cols -= {"_pattern_idx"}
                # Exclude internal columns from join keys:
                # - _prov_* columns are provenance metadata
                # - *_value columns are typed object values (can have nulls that break joins)
                shared_cols = {c for c in shared_cols 
                               if not c.startswith("_prov_") and not c.endswith("_value")}
                
                if shared_cols:
                    result_df = result_df.join(pattern_df, on=list(shared_cols), how="inner")
                else:
                    result_df = result_df.join(pattern_df, how="cross")
        
        return result_df if result_df is not None else pl.DataFrame()
    
    def execute(
        self, 
        query: Query, 
        provenance: Optional[ProvenanceContext] = None
    ) -> Union[pl.DataFrame, bool, dict]:
        """
        Execute a SPARQL-Star query.
        
        Args:
            query: Parsed Query AST
            provenance: Optional provenance context for INSERT/DELETE operations
            
        Returns:
            DataFrame for SELECT queries, bool for ASK queries,
            dict with count for INSERT/DELETE operations
        """
        if isinstance(query, SelectQuery):
            return self._execute_select(query)
        elif isinstance(query, AskQuery):
            return self._execute_ask(query)
        elif isinstance(query, DescribeQuery):
            return self._execute_describe(query)
        elif isinstance(query, ConstructQuery):
            return self._execute_construct(query)
        elif isinstance(query, InsertDataQuery):
            return self._execute_insert_data(query, provenance)
        elif isinstance(query, DeleteDataQuery):
            return self._execute_delete_data(query)
        elif isinstance(query, DeleteWhereQuery):
            return self._execute_delete_where(query)
        elif isinstance(query, ModifyQuery):
            return self._execute_modify(query, provenance)
        elif isinstance(query, CreateGraphQuery):
            return self._execute_create_graph(query)
        elif isinstance(query, DropGraphQuery):
            return self._execute_drop_graph(query)
        elif isinstance(query, ClearGraphQuery):
            return self._execute_clear_graph(query)
        elif isinstance(query, LoadQuery):
            return self._execute_load(query, provenance)
        elif isinstance(query, CopyGraphQuery):
            return self._execute_copy_graph(query)
        elif isinstance(query, MoveGraphQuery):
            return self._execute_move_graph(query)
        elif isinstance(query, AddGraphQuery):
            return self._execute_add_graph(query)
        else:
            raise NotImplementedError(f"Query type {type(query)} not yet supported")
    
    def explain(self, query: Query) -> "ExplainPlan":
        """
        Generate an execution plan for a query without executing it.
        
        Args:
            query: Parsed Query AST
            
        Returns:
            ExplainPlan object with query plan details
        """
        from rdf_starbase.storage.query_context import ExplainPlan
        
        if isinstance(query, SelectQuery):
            return self._explain_select(query)
        elif isinstance(query, AskQuery):
            return self._explain_ask(query)
        elif isinstance(query, (ConstructQuery, DescribeQuery)):
            return self._explain_construct_or_describe(query)
        else:
            return ExplainPlan(
                query_type=type(query).__name__,
                patterns=[],
                estimated_cost=0.0,
            )
    
    def _explain_select(self, query: SelectQuery) -> "ExplainPlan":
        """Generate execution plan for SELECT query."""
        from rdf_starbase.storage.query_context import ExplainPlan
        
        patterns = []
        filters = []
        joins = []
        total_cost = 0.0
        total_facts = len(self.store._df)
        
        # Analyze WHERE clause patterns
        if query.where:
            for i, pattern in enumerate(query.where.patterns):
                pattern_info = self._analyze_pattern(pattern, query.prefixes, i, total_facts)
                patterns.append(pattern_info)
                total_cost += pattern_info.get("estimated_rows", 0)
                
                if i > 0:
                    joins.append({
                        "type": "inner",
                        "columns": pattern_info.get("join_columns", []),
                        "pattern_idx": i,
                    })
            
            # Analyze OPTIONAL patterns
            for opt in query.where.optional_patterns:
                for j, opt_pattern in enumerate(opt.patterns):
                    pattern_info = self._analyze_pattern(opt_pattern, query.prefixes, len(patterns), total_facts)
                    pattern_info["optional"] = True
                    patterns.append(pattern_info)
                    joins.append({
                        "type": "left_outer",
                        "columns": pattern_info.get("join_columns", []),
                        "pattern_idx": len(patterns) - 1,
                    })
            
            # Analyze FILTER clauses
            for f in query.where.filters:
                if isinstance(f, Filter):
                    filters.append(str(f.expression))
        
        # Analyze ORDER BY
        order_by = []
        if query.order_by:
            for var, asc in query.order_by:
                order_by.append(f"{var.name} {'ASC' if asc else 'DESC'}")
        
        return ExplainPlan(
            query_type="SELECT",
            patterns=patterns,
            filters=filters,
            joins=joins,
            order_by=order_by,
            limit=query.limit,
            offset=query.offset,
            distinct=query.distinct,
            estimated_cost=total_cost,
        )
    
    def _explain_ask(self, query: AskQuery) -> "ExplainPlan":
        """Generate execution plan for ASK query."""
        from rdf_starbase.storage.query_context import ExplainPlan
        
        patterns = []
        total_facts = len(self.store._df)
        
        if query.where:
            for i, pattern in enumerate(query.where.patterns):
                patterns.append(self._analyze_pattern(pattern, query.prefixes, i, total_facts))
        
        return ExplainPlan(
            query_type="ASK",
            patterns=patterns,
            limit=1,
            estimated_cost=sum(p.get("estimated_rows", 0) for p in patterns),
        )
    
    def _explain_construct_or_describe(self, query: Query) -> "ExplainPlan":
        """Generate execution plan for CONSTRUCT or DESCRIBE query."""
        from rdf_starbase.storage.query_context import ExplainPlan
        
        patterns = []
        total_facts = len(self.store._df)
        
        if hasattr(query, 'where') and query.where:
            for i, pattern in enumerate(query.where.patterns):
                patterns.append(self._analyze_pattern(pattern, query.prefixes, i, total_facts))
        
        return ExplainPlan(
            query_type=type(query).__name__.replace("Query", "").upper(),
            patterns=patterns,
            estimated_cost=sum(p.get("estimated_rows", 0) for p in patterns),
        )
    
    def _analyze_pattern(
        self,
        pattern: TriplePattern,
        prefixes: dict[str, str],
        pattern_idx: int,
        total_facts: int
    ) -> dict:
        """Analyze a triple pattern for EXPLAIN output."""
        # Build description
        s_str = pattern.subject.name if isinstance(pattern.subject, Variable) else str(pattern.subject)
        p_str = pattern.predicate.name if isinstance(pattern.predicate, Variable) else str(pattern.predicate)
        o_str = pattern.object.name if isinstance(pattern.object, Variable) else str(pattern.object)
        
        # Handle IRI with prefixes
        if isinstance(pattern.subject, IRI):
            s_str = self._compact_iri(pattern.subject, prefixes)
        if isinstance(pattern.predicate, IRI):
            p_str = self._compact_iri(pattern.predicate, prefixes)
        if isinstance(pattern.object, IRI):
            o_str = self._compact_iri(pattern.object, prefixes)
        
        description = f"?{s_str} {p_str} ?{o_str}" if isinstance(pattern.subject, Variable) else f"{s_str} {p_str} {o_str}"
        
        # Estimate selectivity
        selectivity = 1.0
        join_columns = []
        
        if not isinstance(pattern.subject, Variable):
            selectivity *= 0.001
        else:
            join_columns.append(pattern.subject.name)
        
        if not isinstance(pattern.predicate, Variable):
            selectivity *= 0.1
        else:
            join_columns.append(pattern.predicate.name)
        
        if not isinstance(pattern.object, Variable):
            selectivity *= 0.01
        else:
            join_columns.append(pattern.object.name)
        
        estimated_rows = max(1, int(total_facts * selectivity))
        
        return {
            "type": "TriplePattern",
            "description": description,
            "selectivity": selectivity,
            "estimated_rows": estimated_rows,
            "join_columns": join_columns,
            "pattern_idx": pattern_idx,
        }
    
    def _compact_iri(self, iri: IRI, prefixes: dict[str, str]) -> str:
        """Compact an IRI using prefixes if possible."""
        for prefix, ns in prefixes.items():
            if iri.value.startswith(ns):
                return f"{prefix}:{iri.value[len(ns):]}"
        return f"<{iri.value}>"

    def _execute_select(self, query: SelectQuery) -> pl.DataFrame:
        """Execute a SELECT query."""
        # Handle FROM clause - restrict to specified graphs
        from_graphs = None
        if query.from_graphs:
            # Merge all FROM graphs into default graph behavior
            from_graphs = [g.value for g in query.from_graphs]
        
        # Start with lazy frame for optimization
        df = self._execute_where(
            query.where, 
            query.prefixes, 
            as_of=query.as_of,
            from_graphs=from_graphs
        )
        
        # Bind provenance variables if requested (source, confidence, timestamp, process)
        # These are special variable names that map to assertion metadata
        provenance_var_mapping = {
            "source": "source",
            "confidence": "confidence", 
            "timestamp": "timestamp",
            "process": "process",
        }
        
        for var in query.variables:
            if isinstance(var, Variable) and var.name in provenance_var_mapping:
                prov_col = provenance_var_mapping[var.name]
                # Find the first pattern's provenance column
                for col in df.columns:
                    if col.startswith("_prov_") and col.endswith(f"_{prov_col}"):
                        df = df.with_columns(pl.col(col).alias(var.name))
                        break
        
        # Determine columns to select before DISTINCT (DISTINCT should only apply to output columns)
        select_cols = None
        if not query.is_select_all():
            select_cols = []
            for v in query.variables:
                if isinstance(v, Variable) and v.name in df.columns:
                    select_cols.append(v.name)
                elif isinstance(v, AggregateExpression) and v.alias and v.alias.name in df.columns:
                    select_cols.append(v.alias.name)
        
        # Handle GROUP BY and aggregates
        if query.group_by or query.has_aggregates():
            df = self._apply_group_by_aggregates(df, query)
        else:
            # Apply DISTINCT if requested (non-aggregate)
            # Must apply DISTINCT only on the projected columns, not internal _prov_* columns
            if query.distinct:
                if select_cols:
                    df = df.unique(subset=select_cols)
                else:
                    # SELECT * - apply unique to all non-internal columns
                    non_internal = [c for c in df.columns if not c.startswith("_prov_")]
                    df = df.unique(subset=non_internal if non_internal else None)
        
        # Apply HAVING (filter after grouping)
        if query.having:
            df = self._apply_filter(df, Filter(expression=query.having))
        
        # Apply ORDER BY
        if query.order_by:
            order_cols = []
            descending = []
            for var, asc in query.order_by:
                if var.name in df.columns:
                    order_cols.append(var.name)
                    descending.append(not asc)
            if order_cols:
                df = df.sort(order_cols, descending=descending)
        
        # Apply LIMIT and OFFSET
        if query.offset:
            df = df.slice(query.offset, query.limit or len(df))
        elif query.limit:
            df = df.head(query.limit)
        
        # Handle unbound SELECT variables that correspond to explicit terms in WHERE
        # Example: SELECT ?s ?p ?o WHERE { ?s <http://example.org/worksFor> ?o }
        # Here ?p is in SELECT but bound to explicit IRI in WHERE - fill it with that value
        if not query.is_select_all() and len(df) > 0:
            explicit_bindings = self._extract_explicit_bindings(query.where, query.prefixes)
            for v in query.variables:
                if isinstance(v, Variable) and v.name not in df.columns:
                    if v.name in explicit_bindings:
                        # Add column with the explicit value
                        df = df.with_columns(
                            pl.lit(explicit_bindings[v.name]).alias(v.name)
                        )
        
        # Select only requested variables (or all if SELECT *)
        if not query.is_select_all():
            select_cols = []
            for v in query.variables:
                if isinstance(v, Variable) and v.name in df.columns:
                    select_cols.append(v.name)
                elif isinstance(v, AggregateExpression) and v.alias and v.alias.name in df.columns:
                    select_cols.append(v.alias.name)
            if select_cols:
                df = df.select(select_cols)
        
        return df
    
    def _apply_group_by_aggregates(
        self,
        df: pl.DataFrame,
        query: SelectQuery
    ) -> pl.DataFrame:
        """
        Apply GROUP BY and aggregate functions to a DataFrame.
        
        Supports: COUNT, SUM, AVG, MIN, MAX, GROUP_CONCAT, SAMPLE
        """
        if len(df) == 0:
            return df
        
        # Build aggregation expressions
        agg_exprs = []
        
        for var in query.variables:
            if isinstance(var, AggregateExpression):
                agg_expr = self._build_aggregate_expr(var)
                if agg_expr is not None:
                    agg_exprs.append(agg_expr)
        
        # If we have GROUP BY, use it; otherwise aggregate entire result
        if query.group_by:
            group_cols = [v.name for v in query.group_by if v.name in df.columns]
            if group_cols and agg_exprs:
                df = df.group_by(group_cols).agg(agg_exprs)
            elif group_cols:
                # GROUP BY without aggregates - just unique combinations
                df = df.select(group_cols).unique()
        elif agg_exprs:
            # Aggregates without GROUP BY - aggregate entire result
            df = df.select(agg_exprs)
        
        return df
    
    def _build_aggregate_expr(self, agg: AggregateExpression) -> Optional[pl.Expr]:
        """Build a Polars aggregation expression from an AggregateExpression AST."""
        # Get the column to aggregate
        if agg.argument is None:
            # COUNT(*) - count all rows
            col_name = None
        elif isinstance(agg.argument, Variable):
            col_name = agg.argument.name
        else:
            return None
        
        # Determine alias
        alias = agg.alias.name if agg.alias else f"{agg.function.lower()}"
        
        # Build the aggregation
        if agg.function == "COUNT":
            if col_name is None:
                expr = pl.len().alias(alias)
            elif agg.distinct:
                expr = pl.col(col_name).n_unique().alias(alias)
            else:
                expr = pl.col(col_name).count().alias(alias)
        elif agg.function == "SUM":
            if col_name:
                expr = pl.col(col_name).cast(pl.Float64).sum().alias(alias)
            else:
                return None
        elif agg.function == "AVG":
            if col_name:
                expr = pl.col(col_name).cast(pl.Float64).mean().alias(alias)
            else:
                return None
        elif agg.function == "MIN":
            if col_name:
                expr = pl.col(col_name).min().alias(alias)
            else:
                return None
        elif agg.function == "MAX":
            if col_name:
                expr = pl.col(col_name).max().alias(alias)
            else:
                return None
        elif agg.function == "GROUP_CONCAT":
            if col_name:
                sep = agg.separator or " "
                expr = pl.col(col_name).cast(pl.Utf8).str.concat(sep).alias(alias)
            else:
                return None
        elif agg.function == "SAMPLE":
            if col_name:
                expr = pl.col(col_name).first().alias(alias)
            else:
                return None
        else:
            return None
        
        return expr
    
    def _execute_ask(self, query: AskQuery) -> bool:
        """Execute an ASK query."""
        df = self._execute_where(query.where, query.prefixes, as_of=query.as_of)
        return len(df) > 0
    
    def _execute_describe(self, query: DescribeQuery) -> pl.DataFrame:
        """
        Execute a DESCRIBE query.
        
        Returns all triples where the resource appears as subject or object.
        """
        prefixes = query.prefixes
        
        # Get resource URIs to describe
        if query.where:
            # Execute WHERE clause to get bindings
            bindings = self._execute_where(query.where, prefixes, as_of=query.as_of)
            resources = set()
            for resource in query.resources:
                if isinstance(resource, Variable) and resource.name in bindings.columns:
                    resources.update(bindings[resource.name].unique().to_list())
                elif isinstance(resource, IRI):
                    resources.add(self._expand_iri(resource.value, prefixes))
        else:
            resources = {
                self._expand_iri(r.value, prefixes) if isinstance(r, IRI) else str(r)
                for r in query.resources
            }
        
        # Get all triples where resource is subject or object
        df = self.store._df
        
        # Apply time-travel filter if specified
        if query.as_of:
            df = df.filter(pl.col("timestamp") <= query.as_of)
        
        if len(df) == 0:
            return df
        
        resource_list = list(resources)
        result = df.filter(
            pl.col("subject").is_in(resource_list) | 
            pl.col("object").is_in(resource_list)
        )
        
        return result
    
    def _execute_construct(self, query: ConstructQuery) -> pl.DataFrame:
        """
        Execute a CONSTRUCT query.
        
        Returns triples constructed from the template using WHERE bindings.
        """
        prefixes = query.prefixes
        bindings = self._execute_where(query.where, prefixes, as_of=query.as_of)
        
        if len(bindings) == 0:
            return pl.DataFrame({"subject": [], "predicate": [], "object": []})
        
        # Build result triples from template
        result_triples = []
        
        for row in bindings.iter_rows(named=True):
            for pattern in query.template:
                # Substitute variables with bound values
                subject = self._substitute_term(pattern.subject, row, prefixes)
                predicate = self._substitute_term(pattern.predicate, row, prefixes)
                obj = self._substitute_term(pattern.object, row, prefixes)
                
                if subject is not None and predicate is not None and obj is not None:
                    result_triples.append({
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                    })
        
        return pl.DataFrame(result_triples) if result_triples else pl.DataFrame({"subject": [], "predicate": [], "object": []})
    
    def _substitute_term(self, term: Term, row: dict, prefixes: dict) -> Optional[str]:
        """Substitute a term with a value from bindings."""
        if isinstance(term, Variable):
            return row.get(term.name)
        elif isinstance(term, IRI):
            return self._expand_iri(term.value, prefixes)
        elif isinstance(term, Literal):
            return term.value
        elif isinstance(term, BlankNode):
            return f"_:{term.label}"
        return str(term)
    
    def _expand_iri(self, iri: str, prefixes: dict) -> str:
        """Expand a prefixed IRI using prefix declarations."""
        if ":" in iri and not iri.startswith("http"):
            parts = iri.split(":", 1)
            if len(parts) == 2 and parts[0] in prefixes:
                return prefixes[parts[0]] + parts[1]
        return iri
    
    def _try_optimize_provenance_pattern(
        self,
        pattern: TriplePattern,
        prefixes: dict[str, str],
        pattern_idx: int
    ) -> Optional[tuple[str, str, QuotedTriplePattern, Optional[str]]]:
        """
        Try to optimize a provenance pattern to direct column access.
        
        Detects patterns like:
            << ?s ?p ?o >> prov:value ?conf        (specific predicate)
            << ?s ?p ?o >> ?mp ?mo                  (variable predicate - get ALL)
        
        And maps them to the corresponding columnar provenance data
        (confidence, source, timestamp, process).
        
        Returns:
            Tuple of (object_var_name, column_name_or_"*", inner_pattern, predicate_var_name)
            - column_name is "*" when predicate is a variable (return all provenance)
            - predicate_var_name is set when predicate is a variable
            None if not a provenance pattern.
        """
        # Must be a triple pattern with a quoted triple as subject
        if not isinstance(pattern.subject, QuotedTriplePattern):
            return None
        
        # Object must be a variable to bind the provenance value
        if not isinstance(pattern.object, Variable):
            return None
        
        # Check if predicate is a variable - if so, return ALL provenance
        if isinstance(pattern.predicate, Variable):
            return (pattern.object.name, "*", pattern.subject, pattern.predicate.name)
        
        # Predicate must be a known provenance predicate IRI
        if not isinstance(pattern.predicate, IRI):
            return None
        
        pred_iri = self._expand_iri(pattern.predicate.value, prefixes)
        
        # Check if it's a provenance predicate we can optimize
        column_name = PROV_PREDICATE_MAP.get(pred_iri)
        if not column_name:
            # Also check without expansion
            column_name = PROV_PREDICATE_MAP.get(pattern.predicate.value)
        
        if not column_name:
            return None
        
        return (pattern.object.name, column_name, pattern.subject, None)
    
    def _execute_where(
        self,
        where: WhereClause,
        prefixes: dict[str, str],
        as_of: Optional[datetime] = None,
        from_graphs: Optional[list[str]] = None,
    ) -> pl.DataFrame:
        """
        Execute a WHERE clause and return matching bindings.
        
        Args:
            where: The WHERE clause to execute
            prefixes: Prefix mappings
            as_of: Optional timestamp for time-travel queries
            from_graphs: Optional list of graph URIs to restrict query to
        
        Includes internal optimization for provenance patterns:
        When detecting patterns like << ?s ?p ?o >> prov:value ?conf,
        we map directly to the confidence column instead of doing a join.
        Also handles << ?s ?p ?o >> ?mp ?mo to return ALL provenance.
        """
        # Handle case where UNION is the only pattern
        if not where.patterns and not where.union_patterns and not where.graph_patterns:
            return pl.DataFrame()
        
        # Separate regular patterns from optimizable provenance patterns
        # For provenance patterns, we execute the inner pattern and bind provenance columns
        patterns_to_execute = []  # List of (idx, pattern, prov_bindings)
        
        for i, pattern in enumerate(where.patterns):
            opt_result = self._try_optimize_provenance_pattern(pattern, prefixes, i)
            if opt_result:
                # This is a provenance pattern - execute inner pattern and bind column
                obj_var_name, col_name, inner_pattern, pred_var_name = opt_result
                # Create a TriplePattern from the inner QuotedTriplePattern
                inner_triple = TriplePattern(
                    subject=inner_pattern.subject,
                    predicate=inner_pattern.predicate,
                    object=inner_pattern.object
                )
                patterns_to_execute.append((i, inner_triple, (obj_var_name, col_name, pred_var_name)))
            else:
                patterns_to_execute.append((i, pattern, None))
        
        # Execute patterns and join results
        # Check if we can parallelize independent pattern groups
        result_df: Optional[pl.DataFrame] = None
        
        if self._parallel and len(patterns_to_execute) >= _PARALLEL_THRESHOLD:
            # Group patterns by shared variables for parallel execution
            groups = self._find_independent_groups(patterns_to_execute)
            
            if len(groups) > 1:
                # Execute independent groups in parallel
                with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(groups))) as executor:
                    futures = {}
                    for group in groups:
                        future = executor.submit(
                            self._execute_pattern_group,
                            group, prefixes, as_of, from_graphs
                        )
                        futures[future] = group
                    
                    # Collect results and cross-join independent groups
                    group_results = []
                    for future in as_completed(futures):
                        group_df = future.result()
                        if len(group_df) > 0:
                            group_results.append(group_df)
                    
                    # Cross-join the independent groups
                    for group_df in group_results:
                        if result_df is None:
                            result_df = group_df
                        else:
                            result_df = result_df.join(group_df, how="cross")
                
                # Skip the sequential loop since we processed everything
                patterns_to_execute = []
        
        for i, pattern, prov_binding in patterns_to_execute:
            pattern_df = self._execute_pattern(pattern, prefixes, i, as_of=as_of, from_graphs=from_graphs)
            
            # If this pattern has a provenance binding, add it as a column alias
            if prov_binding:
                obj_var_name, col_name, pred_var_name = prov_binding
                
                if col_name == "*":
                    # Variable predicate - unpivot ALL provenance columns into rows
                    # Map column names to their prov predicates
                    prov_col_to_pred = {
                        "source": "<http://www.w3.org/ns/prov#wasDerivedFrom>",
                        "confidence": "<http://www.w3.org/ns/prov#value>",
                        "timestamp": "<http://www.w3.org/ns/prov#generatedAtTime>",
                        "process": "<http://www.w3.org/ns/prov#wasGeneratedBy>",
                    }
                    
                    # Find all _prov_ columns for this pattern
                    prov_cols = [c for c in pattern_df.columns if c.startswith(f"_prov_{i}_")]
                    
                    if prov_cols:
                        # Build unpivoted dataframe - one row per provenance value
                        unpivoted_dfs = []
                        base_cols = [c for c in pattern_df.columns if not c.startswith("_prov_")]
                        
                        for prov_col in prov_cols:
                            # Extract column type from _prov_{idx}_{type}
                            col_type = prov_col.split("_")[-1]  # e.g., "source", "confidence"
                            pred_uri = prov_col_to_pred.get(col_type)
                            
                            if pred_uri:
                                # Create a df with this provenance column as the object
                                row_df = pattern_df.select(base_cols + [prov_col])
                                # Filter out nulls
                                row_df = row_df.filter(pl.col(prov_col).is_not_null())
                                
                                if len(row_df) > 0:
                                    # Add predicate and rename object column
                                    row_df = row_df.with_columns([
                                        pl.lit(pred_uri).alias(pred_var_name),
                                        pl.col(prov_col).cast(pl.Utf8).alias(obj_var_name)
                                    ]).drop(prov_col)
                                    unpivoted_dfs.append(row_df)
                        
                        if unpivoted_dfs:
                            pattern_df = pl.concat(unpivoted_dfs)
                        else:
                            # No provenance data - return empty with correct columns
                            pattern_df = pattern_df.select(base_cols).with_columns([
                                pl.lit(None).cast(pl.Utf8).alias(pred_var_name),
                                pl.lit(None).cast(pl.Utf8).alias(obj_var_name)
                            ]).head(0)
                else:
                    # Specific predicate - just alias the column
                    prov_col = f"_prov_{i}_{col_name}"
                    if prov_col in pattern_df.columns:
                        pattern_df = pattern_df.with_columns(
                            pl.col(prov_col).alias(obj_var_name)
                        )
            
            if result_df is None:
                result_df = pattern_df
            else:
                # Find shared variables to join on
                shared_cols = set(result_df.columns) & set(pattern_df.columns)
                shared_cols -= {"_pattern_idx"}  # Don't join on internal columns
                # Exclude internal columns from join keys:
                # - _prov_* columns are provenance metadata
                # - *_value columns are typed object values (can have nulls that break joins)
                shared_cols = {c for c in shared_cols 
                               if not c.startswith("_prov_") and not c.endswith("_value")}
                
                if shared_cols:
                    result_df = result_df.join(
                        pattern_df,
                        on=list(shared_cols),
                        how="inner"
                    )
                else:
                    # Cross join if no shared variables
                    result_df = result_df.join(pattern_df, how="cross")
        
        # Handle GRAPH patterns
        if where.graph_patterns:
            for graph_pattern in where.graph_patterns:
                graph_df = self._execute_graph_pattern(graph_pattern, prefixes, as_of=as_of)
                if result_df is None:
                    result_df = graph_df
                elif len(graph_df) > 0:
                    # Join with existing results
                    shared_cols = set(result_df.columns) & set(graph_df.columns)
                    shared_cols -= {"_pattern_idx"}
                    # Exclude internal columns from join keys
                    shared_cols = {c for c in shared_cols 
                                   if not c.startswith("_prov_") and not c.endswith("_value")}
                    
                    if shared_cols:
                        result_df = result_df.join(graph_df, on=list(shared_cols), how="inner")
                    else:
                        result_df = result_df.join(graph_df, how="cross")
        
        # Handle UNION patterns - these can be standalone or combined with other patterns
        if where.union_patterns:
            for union in where.union_patterns:
                if result_df is None or len(result_df) == 0:
                    # UNION is the primary pattern - execute it directly
                    result_df = self._execute_union_standalone(union, prefixes)
                else:
                    # Combine UNION results with existing patterns
                    result_df = self._apply_union(result_df, union, prefixes)
        
        if result_df is None:
            return pl.DataFrame()
        
        # Apply OPTIONAL patterns with left outer joins
        # (MUST come before FILTER since FILTER may reference optional variables)
        for optional in where.optional_patterns:
            result_df = self._apply_optional(result_df, optional, prefixes)
        
        # Apply standard FILTER clauses (after OPTIONAL so variables are available)
        for filter_clause in where.filters:
            result_df = self._apply_filter(result_df, filter_clause, prefixes)
        
        # Apply BIND clauses - add new columns with computed values
        for bind in where.binds:
            result_df = self._apply_bind(result_df, bind, prefixes)
        
        # Apply SubSelect (nested SELECT) clauses
        for subselect in where.subselects:
            result_df = self._apply_subselect(result_df, subselect, prefixes)
        
        # Apply VALUES clause - filter/join with inline data
        if where.values:
            result_df = self._apply_values(result_df, where.values, prefixes)
        
        # Check if we have matches before removing internal columns
        has_matches = len(result_df) > 0
        
        # Remove internal columns EXCEPT provenance columns (keep _prov_*)
        internal_cols = [c for c in result_df.columns if c.startswith("_") and not c.startswith("_prov_")]
        if internal_cols:
            result_df = result_df.drop(internal_cols)
        
        # If we had matches but now have no columns (all terms were concrete),
        # return a DataFrame with a single row to indicate a match exists
        if has_matches and len(result_df.columns) == 0:
            result_df = pl.DataFrame({"_matched": [True] * has_matches})
            # Actually just need count, not the values
            result_df = pl.DataFrame({"_matched": [True]})
        
        return result_df
    
    def _execute_pattern(
        self,
        pattern: TriplePattern,
        prefixes: dict[str, str],
        pattern_idx: int,
        as_of: Optional[datetime] = None,
        from_graphs: Optional[list[str]] = None,
    ) -> pl.DataFrame:
        """
        Execute a single triple pattern against the store.
        
        Uses integer-level filter pushdown for performance when possible.
        Falls back to string-level filtering for complex patterns.
        
        Args:
            pattern: The triple pattern to match
            prefixes: Prefix mappings
            pattern_idx: Index of this pattern (for internal column naming)
            as_of: Optional timestamp for time-travel queries
            from_graphs: Optional list of graph URIs to restrict query to
        
        Returns a DataFrame with columns for each variable in the pattern.
        """
        # Try integer-level pushdown for concrete terms
        # This avoids materializing the full DataFrame when we can filter at int level
        s_id = None
        p_id = None
        o_id = None
        
        term_dict = self.store._term_dict
        
        # Look up term IDs for concrete pattern elements
        # This is primarily for short-circuit optimization - if a term doesn't
        # exist in the store, we can return empty immediately without scanning
        if not isinstance(pattern.subject, Variable):
            s_value = self._resolve_term(pattern.subject, prefixes)
            s_id = term_dict.get_iri_id(s_value)
            if s_id is None:
                # Subject term not in store - no matches possible
                return self._empty_pattern_result(pattern)
        
        if not isinstance(pattern.predicate, Variable):
            p_value = self._resolve_term(pattern.predicate, prefixes)
            p_id = term_dict.get_iri_id(p_value)
            if p_id is None:
                # Predicate term not in store - no matches possible
                return self._empty_pattern_result(pattern)
        
        if not isinstance(pattern.object, (Variable, QuotedTriplePattern)):
            o_value = str(self._resolve_term(pattern.object, prefixes))
            # Object could be IRI or literal - check both caches
            o_id = term_dict.get_iri_id(o_value)
            if o_id is None:
                o_id = term_dict.get_literal_id(o_value)
            if o_id is None:
                # Object term not in store - no matches possible
                return self._empty_pattern_result(pattern)
        
        # Use the full scan path - the cached _df plus Polars lazy filter
        # is very fast, and avoids the overhead of rebuilding term lookups
        return self._execute_pattern_full_scan(
            pattern, prefixes, pattern_idx, as_of, from_graphs
        )
    
    def _empty_pattern_result(self, pattern: TriplePattern) -> pl.DataFrame:
        """Create an empty DataFrame with the correct columns for a pattern."""
        cols = {}
        if isinstance(pattern.subject, Variable):
            cols[pattern.subject.name] = pl.Series([], dtype=pl.Utf8)
        if isinstance(pattern.predicate, Variable):
            cols[pattern.predicate.name] = pl.Series([], dtype=pl.Utf8)
        if isinstance(pattern.object, Variable):
            cols[pattern.object.name] = pl.Series([], dtype=pl.Utf8)
            cols[f"{pattern.object.name}_value"] = pl.Series([], dtype=pl.Float64)
        return pl.DataFrame(cols) if cols else pl.DataFrame({"_match": []})
    
    def _execute_pattern_full_scan(
        self,
        pattern: TriplePattern,
        prefixes: dict[str, str],
        pattern_idx: int,
        as_of: Optional[datetime],
        from_graphs: Optional[list[str]],
    ) -> pl.DataFrame:
        """
        Execute pattern with full DataFrame scan (fallback for all-variable patterns).
        """
        # Start with all assertions - use lazy for predicate pushdown
        df = self.store._df.lazy()
        
        # Collect all filter conditions for pushdown
        filters = []
        
        # Exclude deprecated by default (pushdown)
        filters.append(~pl.col("deprecated"))
        
        # Apply time-travel filter if specified
        if as_of is not None:
            filters.append(pl.col("timestamp") <= as_of)
        
        # Apply FROM graph restriction
        if from_graphs is not None:
            # Match triples in specified graphs (None for default graph)
            graph_conditions = []
            for g in from_graphs:
                if g is None or g == "":
                    graph_conditions.append(pl.col("graph").is_null())
                else:
                    graph_conditions.append(pl.col("graph") == g)
            if graph_conditions:
                combined = graph_conditions[0]
                for cond in graph_conditions[1:]:
                    combined = combined | cond
                filters.append(combined)
        
        # Apply filters for concrete terms - pushdown to lazy evaluation
        if not isinstance(pattern.subject, Variable):
            value = self._resolve_term(pattern.subject, prefixes)
            filters.append(pl.col("subject") == value)
        
        if not isinstance(pattern.predicate, Variable):
            value = self._resolve_term(pattern.predicate, prefixes)
            filters.append(pl.col("predicate") == value)
        
        if not isinstance(pattern.object, (Variable, QuotedTriplePattern)):
            value = self._resolve_term(pattern.object, prefixes)
            str_value = str(value)
            filters.append(pl.col("object") == str_value)
        
        # Apply all filters at once for optimal pushdown
        if filters:
            combined_filter = filters[0]
            for f in filters[1:]:
                combined_filter = combined_filter & f
            df = df.filter(combined_filter)
        
        # Collect results
        result = df.collect()
        
        # Rename columns to variable names and select relevant columns
        renames = {}
        select_cols = []
        
        if isinstance(pattern.subject, Variable):
            renames["subject"] = pattern.subject.name
            select_cols.append("subject")
        
        if isinstance(pattern.predicate, Variable):
            renames["predicate"] = pattern.predicate.name
            select_cols.append("predicate")
        
        if isinstance(pattern.object, Variable):
            renames["object"] = pattern.object.name
            select_cols.append("object")
            # Also include typed object_value for numeric FILTER comparisons
            # Rename it to the variable name with "_value" suffix
            if "object_value" in result.columns:
                renames["object_value"] = f"{pattern.object.name}_value"
                select_cols.append("object_value")
        
        # Always include provenance columns for provenance filters
        provenance_cols = ["source", "confidence", "timestamp", "process"]
        for col in provenance_cols:
            if col in result.columns:
                renames[col] = f"_prov_{pattern_idx}_{col}"
                select_cols.append(col)
        
        # Select and rename
        if select_cols:
            result = result.select(select_cols)
            result = result.rename(renames)
        else:
            # Pattern has no variables - just return count
            result = pl.DataFrame({"_match": [True] * len(result)})
        
        return result
    
    def _execute_graph_pattern(
        self,
        graph_pattern: "GraphPattern",
        prefixes: dict[str, str],
        as_of: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """
        Execute a GRAPH pattern: GRAPH <uri> { patterns }.
        
        Args:
            graph_pattern: The GRAPH pattern to execute
            prefixes: Prefix mappings
            as_of: Optional timestamp for time-travel queries
            
        Returns:
            DataFrame with matching bindings from the specified graph
        """
        # Resolve the graph reference
        if isinstance(graph_pattern.graph, IRI):
            graph_uri = self._resolve_term(graph_pattern.graph, prefixes)
            graph_filter = [graph_uri]
        elif isinstance(graph_pattern.graph, Variable):
            # Variable graph - match all named graphs and bind the variable
            graph_filter = None  # Will filter manually
            graph_var_name = graph_pattern.graph.name
        else:
            return pl.DataFrame()
        
        # Execute each pattern in the graph
        result_df: Optional[pl.DataFrame] = None
        
        for i, pattern in enumerate(graph_pattern.patterns):
            pattern_df = self._execute_pattern(
                pattern, 
                prefixes, 
                1000 + i,  # Use high pattern idx to avoid conflicts
                as_of=as_of,
                from_graphs=graph_filter
            )
            
            # If graph is a variable, add the graph column as a binding
            if isinstance(graph_pattern.graph, Variable):
                # Need to also get graph column from store
                df = self.store._df.lazy()
                if as_of is not None:
                    df = df.filter(pl.col("timestamp") <= as_of)
                df = df.filter(~pl.col("deprecated"))
                df = df.filter(pl.col("graph").is_not_null())  # Only named graphs
                
                # Re-execute pattern with graph column
                graph_df = self._execute_pattern_with_graph(
                    pattern, prefixes, 1000 + i, as_of=as_of
                )
                if graph_var_name not in graph_df.columns and "graph" in graph_df.columns:
                    graph_df = graph_df.rename({"graph": graph_var_name})
                pattern_df = graph_df
            
            if result_df is None:
                result_df = pattern_df
            else:
                # Join on shared variables
                shared_cols = set(result_df.columns) & set(pattern_df.columns)
                # Exclude internal columns from join keys
                shared_cols = {c for c in shared_cols 
                               if not c.startswith("_prov_") and not c.endswith("_value")}
                if shared_cols:
                    result_df = result_df.join(pattern_df, on=list(shared_cols), how="inner")
                else:
                    result_df = result_df.join(pattern_df, how="cross")
        
        return result_df if result_df is not None else pl.DataFrame()
    
    def _execute_pattern_with_graph(
        self,
        pattern: TriplePattern,
        prefixes: dict[str, str],
        pattern_idx: int,
        as_of: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """Execute a pattern and include the graph column in results."""
        # Start with all assertions
        df = self.store._df.lazy()
        
        if as_of is not None:
            df = df.filter(pl.col("timestamp") <= as_of)
        
        # Only named graphs
        df = df.filter(pl.col("graph").is_not_null())
        
        # Apply filters for concrete terms
        if not isinstance(pattern.subject, Variable):
            value = self._resolve_term(pattern.subject, prefixes)
            if value.startswith("http"):
                df = df.filter(
                    (pl.col("subject") == value) | 
                    (pl.col("subject") == f"<{value}>")
                )
            else:
                df = df.filter(pl.col("subject") == value)
        
        if not isinstance(pattern.predicate, Variable):
            value = self._resolve_term(pattern.predicate, prefixes)
            if value.startswith("http"):
                df = df.filter(
                    (pl.col("predicate") == value) | 
                    (pl.col("predicate") == f"<{value}>")
                )
            else:
                df = df.filter(pl.col("predicate") == value)
        
        if not isinstance(pattern.object, (Variable, QuotedTriplePattern)):
            value = self._resolve_term(pattern.object, prefixes)
            str_value = str(value)
            if str_value.startswith("http"):
                df = df.filter(
                    (pl.col("object") == str_value) | 
                    (pl.col("object") == f"<{str_value}>")
                )
            else:
                df = df.filter(pl.col("object") == str_value)
        
        df = df.filter(~pl.col("deprecated"))
        result = df.collect()
        
        # Rename and select columns
        renames = {}
        select_cols = ["graph"]  # Always include graph
        
        if isinstance(pattern.subject, Variable):
            renames["subject"] = pattern.subject.name
            select_cols.append("subject")
        
        if isinstance(pattern.predicate, Variable):
            renames["predicate"] = pattern.predicate.name
            select_cols.append("predicate")
        
        if isinstance(pattern.object, Variable):
            renames["object"] = pattern.object.name
            select_cols.append("object")
        
        if select_cols:
            result = result.select(select_cols)
            result = result.rename(renames)
        
        return result

    def _resolve_term(self, term: Term, prefixes: dict[str, str]) -> str:
        """Resolve a term to its string value for matching against store."""
        if isinstance(term, IRI):
            value = term.value
            # Expand prefixed names
            if ":" in value and not value.startswith("http"):
                prefix, local = value.split(":", 1)
                if prefix in prefixes:
                    value = prefixes[prefix] + local
            # Return without angle brackets - store has mixed formats
            # The _execute_pattern will try both with/without brackets
            return value
        elif isinstance(term, Literal):
            return str(term.value)
        elif isinstance(term, BlankNode):
            return f"_:{term.label}"
        else:
            return str(term)
    
    def _apply_filter(self, df: pl.DataFrame, filter_clause: Filter, prefixes: dict[str, str] = None) -> pl.DataFrame:
        """Apply a standard FILTER to the DataFrame."""
        if prefixes is None:
            prefixes = {}
        # Handle EXISTS/NOT EXISTS specially
        if isinstance(filter_clause.expression, ExistsExpression):
            return self._apply_exists_filter(df, filter_clause.expression, prefixes)
        
        expr = self._build_filter_expression(filter_clause.expression, df)
        if expr is not None:
            return df.filter(expr)
        return df
    
    def _apply_exists_filter(
        self,
        df: pl.DataFrame,
        exists_expr: ExistsExpression,
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """
        Apply EXISTS or NOT EXISTS filter.
        
        EXISTS { pattern } keeps rows where pattern matches.
        NOT EXISTS { pattern } keeps rows where pattern does NOT match.
        """
        # Execute the inner pattern with the same prefixes
        pattern_df = self._execute_where(exists_expr.pattern, prefixes)
        
        if pattern_df is None or len(pattern_df) == 0:
            # No matches in pattern
            if exists_expr.negated:
                # NOT EXISTS with no matches -> keep all rows
                return df
            else:
                # EXISTS with no matches -> keep no rows
                return df.head(0)
        
        # Find shared variables between outer query and EXISTS pattern
        shared_cols = set(df.columns) & set(pattern_df.columns)
        # Remove internal columns
        shared_cols = {c for c in shared_cols if not c.startswith("_")}
        
        if not shared_cols:
            # No shared variables - EXISTS is either true or false for all rows
            if exists_expr.negated:
                # NOT EXISTS with matches but no join -> keep no rows
                return df.head(0)
            else:
                # EXISTS with matches but no join -> keep all rows
                return df
        
        # Use anti-join for NOT EXISTS, semi-join for EXISTS
        if exists_expr.negated:
            # NOT EXISTS: keep rows that DON'T have a match
            return df.join(pattern_df.select(list(shared_cols)).unique(), 
                          on=list(shared_cols), how="anti")
        else:
            # EXISTS: keep rows that DO have a match
            return df.join(pattern_df.select(list(shared_cols)).unique(), 
                          on=list(shared_cols), how="semi")
    
    def _apply_optional(
        self,
        df: pl.DataFrame,
        optional: OptionalPattern,
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """
        Apply an OPTIONAL pattern using left outer join.
        
        OPTIONAL { ... } patterns add bindings when matched but keep
        rows even when no match exists (with NULL for optional columns).
        """
        # Collect all variables that will be bound by the optional pattern
        optional_variables = set()
        for pattern in optional.patterns:
            if hasattr(pattern, 'get_variables'):
                for var in pattern.get_variables():
                    optional_variables.add(var.name)
        for bind in optional.binds:
            optional_variables.add(bind.variable.name)
        
        # Execute the optional patterns
        optional_df: Optional[pl.DataFrame] = None
        
        for i, pattern in enumerate(optional.patterns):
            if isinstance(pattern, (TriplePattern, QuotedTriplePattern)):
                pattern_df = self._execute_pattern(pattern, prefixes, 1000 + i)
                
                if optional_df is None:
                    optional_df = pattern_df
                else:
                    shared_cols = set(optional_df.columns) & set(pattern_df.columns)
                    shared_cols -= {"_pattern_idx"}
                    
                    if shared_cols:
                        optional_df = optional_df.join(pattern_df, on=list(shared_cols), how="inner")
                    else:
                        optional_df = optional_df.join(pattern_df, how="cross")
        
        # Handle case where optional pattern has no matches
        if optional_df is None or len(optional_df) == 0:
            # Add null columns for all optional variables that aren't already in df
            for var_name in optional_variables:
                if var_name not in df.columns:
                    df = df.with_columns(pl.lit(None).alias(var_name))
            return df
        
        # Apply filters within the optional block
        for filter_clause in optional.filters:
            optional_df = self._apply_filter(optional_df, filter_clause, prefixes)
        
        # Apply binds within the optional block
        for bind in optional.binds:
            optional_df = self._apply_bind(optional_df, bind, prefixes)
        
        # Remove internal columns from optional_df
        internal_cols = [c for c in optional_df.columns if c.startswith("_")]
        if internal_cols:
            optional_df = optional_df.drop(internal_cols)
        
        # Find shared columns for the join
        shared_cols = set(df.columns) & set(optional_df.columns)
        
        if shared_cols:
            # Left outer join - keep all rows from df, add optional columns where matched
            result = df.join(optional_df, on=list(shared_cols), how="left")
            
            # Ensure all optional variables exist in result (may be null for non-matches)
            for var_name in optional_variables:
                if var_name not in result.columns:
                    result = result.with_columns(pl.lit(None).alias(var_name))
            
            return result
        else:
            # No shared columns - add null columns for optional variables
            for var_name in optional_variables:
                if var_name not in df.columns:
                    df = df.with_columns(pl.lit(None).alias(var_name))
            return df
    
    def _apply_union(
        self,
        df: pl.DataFrame,
        union: UnionPattern,
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """
        Apply a UNION pattern by combining results from alternatives.
        
        UNION combines results from multiple pattern groups:
        { ?s ?p ?o } UNION { ?s ?q ?r }
        
        Returns all rows matching ANY of the alternatives.
        """
        union_results = []
        
        for i, alternative in enumerate(union.alternatives):
            # Execute each alternative as a mini WHERE clause
            alt_where = WhereClause(patterns=alternative)
            alt_df = self._execute_where(alt_where, prefixes)
            
            if len(alt_df) > 0:
                union_results.append(alt_df)
        
        if not union_results:
            return df
        
        # Combine all union results
        if len(union_results) == 1:
            union_df = union_results[0]
        else:
            # Align schemas - add missing columns with null values
            all_columns = set()
            for r in union_results:
                all_columns.update(r.columns)
            
            aligned_results = []
            for r in union_results:
                missing = all_columns - set(r.columns)
                if missing:
                    for col in missing:
                        r = r.with_columns(pl.lit(None).alias(col))
                aligned_results.append(r.select(sorted(all_columns)))
            
            union_df = pl.concat(aligned_results, how="vertical")
        
        # If we have existing results, join with them
        if len(df) > 0 and len(df.columns) > 0:
            shared_cols = set(df.columns) & set(union_df.columns)
            if shared_cols:
                return df.join(union_df, on=list(shared_cols), how="inner")
            else:
                return df.join(union_df, how="cross")
        
        return union_df
    
    def _execute_union_standalone(
        self,
        union: UnionPattern,
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """
        Execute a UNION pattern as a standalone query (no prior patterns).
        
        Returns combined results from all alternatives.
        Parallelizes execution when multiple alternatives exist.
        """
        alternatives = union.alternatives
        
        def build_where_clause(alternative):
            """Build a WhereClause from a UNION alternative."""
            if isinstance(alternative, dict):
                # New format: dict with patterns, filters, binds
                return WhereClause(
                    patterns=alternative.get('patterns', []),
                    filters=alternative.get('filters', []),
                    binds=alternative.get('binds', [])
                )
            else:
                # Legacy format: list of patterns
                return WhereClause(patterns=alternative)
        
        # Parallel execution for multiple UNION branches
        if self._parallel and len(alternatives) >= 2:
            with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(alternatives))) as executor:
                futures = []
                for alternative in alternatives:
                    alt_where = build_where_clause(alternative)
                    future = executor.submit(self._execute_where, alt_where, prefixes)
                    futures.append(future)
                
                union_results = []
                for future in as_completed(futures):
                    alt_df = future.result()
                    if len(alt_df) > 0:
                        union_results.append(alt_df)
        else:
            # Sequential execution
            union_results = []
            for alternative in alternatives:
                alt_where = build_where_clause(alternative)
                alt_df = self._execute_where(alt_where, prefixes)
                if len(alt_df) > 0:
                    union_results.append(alt_df)
        
        if not union_results:
            return pl.DataFrame()
        
        # Combine all union results
        if len(union_results) == 1:
            return union_results[0]
        
        # Align schemas - add missing columns with null values
        all_columns = set()
        for r in union_results:
            all_columns.update(r.columns)
        
        aligned_results = []
        for r in union_results:
            missing = all_columns - set(r.columns)
            if missing:
                for col in missing:
                    r = r.with_columns(pl.lit(None).alias(col))
            aligned_results.append(r.select(sorted(all_columns)))
        
        return pl.concat(aligned_results, how="vertical")
    
    def _apply_bind(
        self,
        df: pl.DataFrame,
        bind: Bind,
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """
        Apply a BIND clause, adding a new column with the computed value.
        
        BIND(?price * 1.1 AS ?taxed_price)
        BIND("default" AS ?label)
        """
        var_name = bind.variable.name
        
        # Handle different expression types
        if isinstance(bind.expression, Variable):
            # BIND(?x AS ?y) - copy column
            src_name = bind.expression.name
            if src_name in df.columns:
                df = df.with_columns(pl.col(src_name).alias(var_name))
        elif isinstance(bind.expression, Literal):
            # BIND("value" AS ?var) - add constant
            df = df.with_columns(pl.lit(bind.expression.value).alias(var_name))
        elif isinstance(bind.expression, IRI):
            # BIND(<uri> AS ?var) - add constant IRI
            value = self._resolve_term(bind.expression, prefixes)
            df = df.with_columns(pl.lit(value).alias(var_name))
        elif isinstance(bind.expression, Comparison):
            # BIND(?x > 5 AS ?flag) - boolean expression
            expr = self._build_filter_expression(bind.expression)
            if expr is not None:
                df = df.with_columns(expr.alias(var_name))
        elif isinstance(bind.expression, FunctionCall):
            # BIND(CONCAT(?a, ?b) AS ?c) - function call
            expr = self._build_function_call(bind.expression)
            if expr is not None:
                df = df.with_columns(expr.alias(var_name))
        
        return df
    
    def _apply_values(
        self,
        df: pl.DataFrame,
        values: ValuesClause,
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """
        Apply a VALUES clause, joining with inline data.
        
        VALUES ?x { 1 2 3 }
        VALUES (?x ?y) { (1 2) (3 4) }
        """
        # Build a DataFrame from the VALUES data
        var_names = [v.name for v in values.variables]
        
        # Convert bindings to column data
        columns = {name: [] for name in var_names}
        
        for row in values.bindings:
            for i, val in enumerate(row):
                if i < len(var_names):
                    if val is None:
                        columns[var_names[i]].append(None)
                    elif isinstance(val, Literal):
                        columns[var_names[i]].append(val.value)
                    elif isinstance(val, IRI):
                        columns[var_names[i]].append(self._resolve_term(val, prefixes))
                    else:
                        columns[var_names[i]].append(str(val))
        
        values_df = pl.DataFrame(columns)
        
        if len(df) == 0 or len(df.columns) == 0:
            # VALUES is the only source - return it directly
            return values_df
        
        # Join with existing results
        shared_cols = set(df.columns) & set(values_df.columns)
        
        if shared_cols:
            # Inner join on shared columns - filter to matching values
            return df.join(values_df, on=list(shared_cols), how="inner")
        else:
            # Cross join - add all value combinations
            return df.join(values_df, how="cross")
    
    def _apply_subselect(
        self,
        df: pl.DataFrame,
        subselect: SubSelect,
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """
        Apply a subquery (nested SELECT) to the current results.
        
        The subquery is executed independently and then joined with the
        outer query on any shared variables.
        
        Example:
            SELECT ?person ?avgAge WHERE {
                ?person a foaf:Person .
                {
                    SELECT (AVG(?age) AS ?avgAge)
                    WHERE { ?p foaf:age ?age }
                }
            }
        """
        # Execute the subquery's WHERE clause
        subquery_df = self._execute_where(subselect.where, prefixes)
        
        if len(subquery_df) == 0:
            return df
        
        # Apply GROUP BY if present
        if subselect.group_by:
            group_cols = [v.name for v in subselect.group_by]
            
            # Build aggregation expressions
            agg_exprs = []
            select_vars = []
            
            for var in subselect.variables:
                if isinstance(var, Variable):
                    select_vars.append(var.name)
                elif isinstance(var, AggregateExpression):
                    agg_expr = self._build_aggregate_expr(var)
                    if agg_expr is not None:
                        alias = var.alias.name if var.alias else f"_agg_{len(agg_exprs)}"
                        agg_exprs.append(agg_expr.alias(alias))
                        select_vars.append(alias)
            
            if group_cols:
                subquery_df = subquery_df.group_by(group_cols).agg(agg_exprs)
            elif agg_exprs:
                # No GROUP BY but has aggregates - aggregate over entire result
                subquery_df = subquery_df.select(agg_exprs)
        else:
            # No GROUP BY - just project the selected variables
            select_vars = []
            for var in subselect.variables:
                if isinstance(var, Variable):
                    select_vars.append(var.name)
                elif isinstance(var, AggregateExpression):
                    # Aggregate without GROUP BY
                    agg_expr = self._build_aggregate_expr(var)
                    if agg_expr is not None:
                        alias = var.alias.name if var.alias else f"_agg"
                        subquery_df = subquery_df.select(agg_expr.alias(alias))
                        select_vars.append(alias)
        
        # Apply HAVING if present
        if subselect.having:
            subquery_df = self._apply_filter(subquery_df, subselect.having, prefixes)
        
        # Apply ORDER BY if present
        if subselect.order_by:
            order_cols = []
            descending = []
            for cond in subselect.order_by:
                if isinstance(cond.expression, Variable):
                    order_cols.append(cond.expression.name)
                    descending.append(cond.descending)
            if order_cols:
                subquery_df = subquery_df.sort(order_cols, descending=descending)
        
        # Apply LIMIT/OFFSET
        if subselect.offset:
            subquery_df = subquery_df.slice(subselect.offset)
        if subselect.limit:
            subquery_df = subquery_df.head(subselect.limit)
        
        # Project only selected variables
        available_cols = set(subquery_df.columns)
        project_cols = [c for c in select_vars if c in available_cols]
        if project_cols:
            subquery_df = subquery_df.select(project_cols)
        
        # Join with outer query
        if len(df) == 0 or len(df.columns) == 0:
            return subquery_df
        
        shared_cols = set(df.columns) & set(subquery_df.columns)
        
        if shared_cols:
            return df.join(subquery_df, on=list(shared_cols), how="inner")
        else:
            return df.join(subquery_df, how="cross")
    
    def _build_filter_expression(
        self,
        expr: Union[Comparison, LogicalExpression, FunctionCall, ExistsExpression],
        current_df: Optional[pl.DataFrame] = None
    ) -> Optional[pl.Expr]:
        """Build a Polars filter expression from SPARQL filter AST."""
        
        if isinstance(expr, Comparison):
            # Handle type coercion for variable vs literal comparisons
            left, right = self._build_comparison_operands(expr.left, expr.right)
            
            if left is None or right is None:
                return None
            
            op_map = {
                ComparisonOp.EQ: lambda l, r: l == r,
                ComparisonOp.NE: lambda l, r: l != r,
                ComparisonOp.LT: lambda l, r: l < r,
                ComparisonOp.LE: lambda l, r: l <= r,
                ComparisonOp.GT: lambda l, r: l > r,
                ComparisonOp.GE: lambda l, r: l >= r,
            }
            
            return op_map[expr.operator](left, right)
        
        elif isinstance(expr, LogicalExpression):
            operand_exprs = [
                self._build_filter_expression(op, current_df) for op in expr.operands
            ]
            operand_exprs = [e for e in operand_exprs if e is not None]
            
            if not operand_exprs:
                return None
            
            if expr.operator == LogicalOp.NOT:
                return ~operand_exprs[0]
            elif expr.operator == LogicalOp.AND:
                result = operand_exprs[0]
                for e in operand_exprs[1:]:
                    result = result & e
                return result
            elif expr.operator == LogicalOp.OR:
                result = operand_exprs[0]
                for e in operand_exprs[1:]:
                    result = result | e
                return result
        
        elif isinstance(expr, FunctionCall):
            return self._build_function_call(expr)
        
        elif isinstance(expr, ExistsExpression):
            # EXISTS/NOT EXISTS is handled specially in _apply_filter
            # Return a placeholder that will be evaluated there
            return None
        
        return None
    
    def _build_comparison_operands(
        self,
        left_term: Union[Variable, Literal, IRI, FunctionCall],
        right_term: Union[Variable, Literal, IRI, FunctionCall]
    ) -> tuple[Optional[pl.Expr], Optional[pl.Expr]]:
        """
        Build comparison operands with proper type coercion.
        
        When comparing a variable (column) with a typed literal, uses the
        pre-computed typed value column (e.g., age_value) if available.
        """
        left = self._term_to_expr(left_term)
        right = self._term_to_expr(right_term)
        
        if left is None or right is None:
            return left, right
        
        # Use typed _value column for numeric comparisons with variables
        if isinstance(left_term, Variable) and isinstance(right_term, Literal):
            if right_term.datatype and self._is_numeric_datatype(right_term.datatype):
                # Use the pre-computed typed value column
                left = pl.col(f"{left_term.name}_value")
        elif isinstance(right_term, Variable) and isinstance(left_term, Literal):
            if left_term.datatype and self._is_numeric_datatype(left_term.datatype):
                # Use the pre-computed typed value column
                right = pl.col(f"{right_term.name}_value")
        
        return left, right
    
    def _is_numeric_datatype(self, datatype: str) -> bool:
        """Check if a datatype is numeric (integer, decimal, double, float, boolean)."""
        numeric_indicators = ["integer", "int", "decimal", "float", "double", "boolean"]
        datatype_lower = datatype.lower()
        return any(ind in datatype_lower for ind in numeric_indicators)
    
    def _cast_column_for_comparison(self, col_expr: pl.Expr, datatype: str) -> pl.Expr:
        """Cast a column expression based on the datatype of the comparison literal."""
        if "integer" in datatype or "int" in datatype:
            return col_expr.cast(pl.Int64, strict=False)
        elif "decimal" in datatype or "float" in datatype or "double" in datatype:
            return col_expr.cast(pl.Float64, strict=False)
        elif "boolean" in datatype:
            return col_expr.cast(pl.Boolean, strict=False)
        return col_expr

    def _term_to_expr(
        self,
        term: Union[Variable, Literal, IRI, FunctionCall]
    ) -> Optional[pl.Expr]:
        """Convert a term to a Polars expression."""
        if isinstance(term, Variable):
            return pl.col(term.name)
        elif isinstance(term, Literal):
            # Convert typed literals to appropriate Python types
            value = term.value
            if term.datatype:
                value = self._convert_typed_value(value, term.datatype)
            return pl.lit(value)
        elif isinstance(term, IRI):
            return pl.lit(term.value)
        elif isinstance(term, FunctionCall):
            return self._build_function_call(term)
        return None
    
    def _convert_typed_value(self, value: Any, datatype: str) -> Any:
        """Convert a literal value based on its XSD datatype."""
        if isinstance(value, (int, float, bool)):
            return value  # Already native type
        
        # XSD numeric types
        if "integer" in datatype or "int" in datatype:
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        elif "decimal" in datatype or "float" in datatype or "double" in datatype:
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        elif "boolean" in datatype:
            if isinstance(value, str):
                return value.lower() == "true"
            return bool(value)
        
        return value
    
    def _build_function_call(self, func: FunctionCall) -> Optional[pl.Expr]:
        """Build a Polars expression for a SPARQL function."""
        name = func.name.upper()
        
        if name == "BOUND":
            if func.arguments and isinstance(func.arguments[0], Variable):
                return pl.col(func.arguments[0].name).is_not_null()
        
        elif name in ("ISIRI", "ISURI"):
            if func.arguments and isinstance(func.arguments[0], Variable):
                col = pl.col(func.arguments[0].name)
                return col.str.starts_with("http")
        
        elif name == "ISLITERAL":
            if func.arguments and isinstance(func.arguments[0], Variable):
                col = pl.col(func.arguments[0].name)
                return ~col.str.starts_with("http") & ~col.str.starts_with("_:")
        
        elif name == "ISBLANK":
            if func.arguments and isinstance(func.arguments[0], Variable):
                col = pl.col(func.arguments[0].name)
                return col.str.starts_with("_:")
        
        elif name == "STR":
            if func.arguments and isinstance(func.arguments[0], Variable):
                return pl.col(func.arguments[0].name).cast(pl.Utf8)
        
        elif name == "COALESCE":
            # COALESCE returns the first non-null argument
            if func.arguments:
                exprs = []
                for arg in func.arguments:
                    if isinstance(arg, Variable):
                        exprs.append(pl.col(arg.name))
                    elif isinstance(arg, Literal):
                        exprs.append(pl.lit(arg.value))
                if exprs:
                    return pl.coalesce(exprs)
        
        elif name == "IF":
            # IF(condition, then_value, else_value)
            if len(func.arguments) >= 3:
                cond_expr = self._build_filter_expression(func.arguments[0])
                then_expr = self._arg_to_expr(func.arguments[1])
                else_expr = self._arg_to_expr(func.arguments[2])
                if cond_expr is not None and then_expr is not None and else_expr is not None:
                    return pl.when(cond_expr).then(then_expr).otherwise(else_expr)
        
        elif name == "LANG":
            # Return language tag of a literal (simplified - returns empty string)
            if func.arguments and isinstance(func.arguments[0], Variable):
                # For now, literals don't carry language tags in our storage
                return pl.lit("")
        
        elif name == "DATATYPE":
            # Return datatype IRI of a literal
            if func.arguments and isinstance(func.arguments[0], Variable):
                col = pl.col(func.arguments[0].name)
                # Heuristic: detect datatype from value pattern
                return pl.when(col.str.contains(r"^\d+$")).then(pl.lit("http://www.w3.org/2001/XMLSchema#integer")) \
                    .when(col.str.contains(r"^\d+\.\d+$")).then(pl.lit("http://www.w3.org/2001/XMLSchema#decimal")) \
                    .when(col.str.to_lowercase().is_in(["true", "false"])).then(pl.lit("http://www.w3.org/2001/XMLSchema#boolean")) \
                    .otherwise(pl.lit("http://www.w3.org/2001/XMLSchema#string"))
        
        elif name == "STRLEN":
            if func.arguments and isinstance(func.arguments[0], Variable):
                return pl.col(func.arguments[0].name).str.len_chars()
        
        elif name == "CONTAINS":
            if len(func.arguments) >= 2:
                str_expr = self._arg_to_expr(func.arguments[0])
                pattern = self._arg_to_literal_value(func.arguments[1])
                if str_expr is not None and pattern is not None:
                    return str_expr.str.contains(pattern, literal=True)
        
        elif name == "STRSTARTS":
            if len(func.arguments) >= 2:
                str_expr = self._arg_to_expr(func.arguments[0])
                prefix = self._arg_to_literal_value(func.arguments[1])
                if str_expr is not None and prefix is not None:
                    return str_expr.str.starts_with(prefix)
        
        elif name == "STRENDS":
            if len(func.arguments) >= 2:
                str_expr = self._arg_to_expr(func.arguments[0])
                suffix = self._arg_to_literal_value(func.arguments[1])
                if str_expr is not None and suffix is not None:
                    return str_expr.str.ends_with(suffix)
        
        elif name == "LCASE":
            if func.arguments and isinstance(func.arguments[0], Variable):
                return pl.col(func.arguments[0].name).str.to_lowercase()
        
        elif name == "UCASE":
            if func.arguments and isinstance(func.arguments[0], Variable):
                return pl.col(func.arguments[0].name).str.to_uppercase()
        
        elif name == "CONCAT":
            if func.arguments:
                exprs = [self._arg_to_expr(arg) for arg in func.arguments]
                if all(e is not None for e in exprs):
                    return pl.concat_str(exprs)
        
        elif name == "REPLACE":
            # REPLACE(str, pattern, replacement)
            if len(func.arguments) >= 3:
                str_expr = self._arg_to_expr(func.arguments[0])
                pattern = self._arg_to_literal_value(func.arguments[1])
                replacement = self._arg_to_literal_value(func.arguments[2])
                if str_expr is not None and pattern and replacement is not None:
                    return str_expr.str.replace_all(pattern, replacement)
        
        elif name == "ABS":
            if func.arguments:
                expr = self._arg_to_expr(func.arguments[0])
                if expr is not None:
                    return expr.abs()
        
        elif name == "ROUND":
            if func.arguments:
                expr = self._arg_to_expr(func.arguments[0])
                if expr is not None:
                    return expr.round(0)
        
        elif name == "CEIL":
            if func.arguments:
                expr = self._arg_to_expr(func.arguments[0])
                if expr is not None:
                    return expr.ceil()
        
        elif name == "FLOOR":
            if func.arguments:
                expr = self._arg_to_expr(func.arguments[0])
                if expr is not None:
                    return expr.floor()
        
        return None
    
    def _arg_to_expr(self, arg) -> Optional[pl.Expr]:
        """Convert a function argument to a Polars expression."""
        if isinstance(arg, Variable):
            return pl.col(arg.name)
        elif isinstance(arg, Literal):
            return pl.lit(arg.value)
        elif isinstance(arg, FunctionCall):
            return self._build_function_call(arg)
        elif isinstance(arg, Comparison):
            return self._build_comparison(arg)
        return None
    
    def _arg_to_literal_value(self, arg) -> Optional[str]:
        """Extract a literal string value from an argument."""
        if isinstance(arg, Literal):
            return arg.value
        return None
    
    def _execute_insert_data(
        self, 
        query: InsertDataQuery,
        provenance: Optional[ProvenanceContext] = None
    ) -> dict:
        """
        Execute an INSERT DATA query with RDF-Star provenance recognition.
        
        This method intelligently handles RDF-Star annotations:
        - Regular triples are inserted with default provenance
        - Quoted triple annotations like << s p o >> prov:wasAttributedTo "source"
          are recognized and applied to the base triple's metadata
        
        Args:
            query: The InsertDataQuery AST
            provenance: Optional default provenance context
            
        Returns:
            Dict with 'count' of inserted triples
        """
        if provenance is None:
            provenance = ProvenanceContext(source="SPARQL_INSERT", confidence=1.0)
        
        prefixes = query.prefixes
        
        # First pass: collect provenance annotations for quoted triples
        # Key: (subject, predicate, object) tuple of the base triple
        # Value: dict with 'source', 'confidence', 'timestamp' overrides
        provenance_annotations: dict[tuple[str, str, str], dict[str, Any]] = {}
        
        # Separate regular triples from provenance annotations
        regular_triples = []
        
        for triple in query.triples:
            # Check if this is a provenance annotation (subject is a quoted triple)
            if isinstance(triple.subject, QuotedTriplePattern):
                # This is an RDF-Star annotation like:
                # << ex:s ex:p ex:o >> prov:wasAttributedTo "IMDb" .
                quoted = triple.subject
                predicate_iri = self._resolve_term_value(triple.predicate, prefixes)
                obj_value = self._resolve_term_value(triple.object, prefixes)
                
                # Get the base triple key
                base_s = self._resolve_term_value(quoted.subject, prefixes)
                base_p = self._resolve_term_value(quoted.predicate, prefixes)
                base_o = self._resolve_term_value(quoted.object, prefixes)
                base_key = (base_s, base_p, base_o)
                
                # Initialize annotations dict for this triple if needed
                if base_key not in provenance_annotations:
                    provenance_annotations[base_key] = {}
                
                # Check if this predicate maps to a provenance field
                if predicate_iri in PROVENANCE_SOURCE_PREDICATES:
                    provenance_annotations[base_key]['source'] = str(obj_value)
                elif predicate_iri in PROVENANCE_CONFIDENCE_PREDICATES:
                    try:
                        conf_val = float(obj_value)
                        provenance_annotations[base_key]['confidence'] = conf_val
                    except (ValueError, TypeError):
                        # If can't parse as float, store as-is (will be ignored)
                        pass
                elif predicate_iri in PROVENANCE_TIMESTAMP_PREDICATES:
                    provenance_annotations[base_key]['timestamp'] = str(obj_value)
                else:
                    # Not a recognized provenance predicate - treat as regular triple
                    # (This creates an actual RDF-Star triple about the quoted triple)
                    regular_triples.append(triple)
            else:
                # Regular triple
                regular_triples.append(triple)
        
        # Second pass: insert regular triples with their provenance
        count = 0
        
        for triple in regular_triples:
            subject = self._resolve_term_value(triple.subject, prefixes)
            predicate = self._resolve_term_value(triple.predicate, prefixes)
            obj = self._resolve_term_value(triple.object, prefixes)
            
            # Check if we have provenance annotations for this triple
            triple_key = (subject, predicate, obj)
            if triple_key in provenance_annotations:
                annotations = provenance_annotations[triple_key]
                # Create provenance context with overrides
                triple_prov = ProvenanceContext(
                    source=annotations.get('source', provenance.source),
                    confidence=annotations.get('confidence', provenance.confidence),
                    timestamp=provenance.timestamp,
                )
            else:
                triple_prov = provenance
            
            self.store.add_triple(subject, predicate, obj, triple_prov)
            count += 1
        
        # Also insert any base triples that only had annotations (no regular triple)
        # This handles the case where annotations come first:
        # << ex:s ex:p ex:o >> prov:wasAttributedTo "source" .
        # (but no explicit ex:s ex:p ex:o . triple)
        inserted_keys = {
            (self._resolve_term_value(t.subject, prefixes),
             self._resolve_term_value(t.predicate, prefixes),
             self._resolve_term_value(t.object, prefixes))
            for t in regular_triples
            if not isinstance(t.subject, QuotedTriplePattern)
        }
        
        for base_key, annotations in provenance_annotations.items():
            if base_key not in inserted_keys:
                # This triple was only defined via annotations, insert it
                subject, predicate, obj = base_key
                triple_prov = ProvenanceContext(
                    source=annotations.get('source', provenance.source),
                    confidence=annotations.get('confidence', provenance.confidence),
                    timestamp=provenance.timestamp,
                )
                self.store.add_triple(subject, predicate, obj, triple_prov)
                count += 1
        
        return {"count": count, "operation": "INSERT DATA"}
    
    def _execute_delete_data(self, query: DeleteDataQuery) -> dict:
        """
        Execute a DELETE DATA query.
        
        DELETE DATA {
            <subject> <predicate> <object> .
        }
        
        Deletes the specified concrete triples from the store.
        """
        prefixes = query.prefixes
        count = 0
        
        for triple in query.triples:
            subject = self._resolve_term_value(triple.subject, prefixes)
            predicate = self._resolve_term_value(triple.predicate, prefixes)
            obj = self._resolve_term_value(triple.object, prefixes)
            
            # Mark the triple as deleted
            deleted = self.store.mark_deleted(s=subject, p=predicate, o=obj)
            count += deleted
        
        return {"count": count, "operation": "DELETE DATA"}
    
    def _execute_delete_where(self, query: DeleteWhereQuery) -> dict:
        """
        Execute a DELETE WHERE query.
        
        DELETE WHERE { ?s ?p ?o }
        
        Finds all matching triples and deletes them.
        """
        # First, execute the WHERE clause to find matching bindings
        where = query.where
        prefixes = query.prefixes
        
        if not where.patterns:
            return {"count": 0, "operation": "DELETE WHERE", "error": "No patterns in WHERE clause"}
        
        # Execute WHERE to get bindings
        bindings = self._execute_where(where, prefixes)
        
        if bindings is None or bindings.height == 0:
            return {"count": 0, "operation": "DELETE WHERE"}
        
        # Build delete patterns from WHERE patterns
        count = 0
        for i in range(bindings.height):
            row = bindings.row(i, named=True)
            for pattern in where.patterns:
                if isinstance(pattern, TriplePattern):
                    # Resolve each component using bindings
                    subject = self._resolve_pattern_term(pattern.subject, row, query.prefixes)
                    predicate = self._resolve_pattern_term(pattern.predicate, row, query.prefixes)
                    obj = self._resolve_pattern_term(pattern.object, row, query.prefixes)
                    
                    if subject and predicate and obj:
                        # Mark as deleted
                        deleted = self.store.mark_deleted(s=subject, p=predicate, o=obj)
                        count += deleted
        
        return {"count": count, "operation": "DELETE WHERE"}
    
    def _execute_modify(
        self, 
        query: ModifyQuery, 
        provenance: Optional[ProvenanceContext] = None
    ) -> dict:
        """
        Execute a DELETE/INSERT WHERE (modify) query.
        
        DELETE { <patterns> }
        INSERT { <patterns> }
        WHERE { <patterns> }
        
        1. Execute WHERE to get variable bindings
        2. For each binding, delete matching patterns from DELETE clause
        3. For each binding, insert patterns from INSERT clause
        """
        where = query.where
        prefixes = query.prefixes
        
        # Execute WHERE to get bindings
        bindings = self._execute_where(where, prefixes)
        
        if bindings is None or bindings.height == 0:
            # No matches - nothing to delete or insert
            return {
                "deleted": 0, 
                "inserted": 0, 
                "operation": "MODIFY"
            }
        
        deleted_count = 0
        inserted_count = 0
        
        # Process each row of bindings
        for i in range(bindings.height):
            row = bindings.row(i, named=True)
            
            # Delete patterns
            for pattern in query.delete_patterns:
                subject = self._resolve_pattern_term(pattern.subject, row, query.prefixes)
                predicate = self._resolve_pattern_term(pattern.predicate, row, query.prefixes)
                obj = self._resolve_pattern_term(pattern.object, row, query.prefixes)
                
                if subject and predicate and obj:
                    deleted = self.store.mark_deleted(s=subject, p=predicate, o=obj)
                    deleted_count += deleted
            
            # Insert patterns
            for pattern in query.insert_patterns:
                subject = self._resolve_pattern_term(pattern.subject, row, query.prefixes)
                predicate = self._resolve_pattern_term(pattern.predicate, row, query.prefixes)
                obj = self._resolve_pattern_term(pattern.object, row, query.prefixes)
                
                if subject and predicate and obj:
                    prov = provenance or ProvenanceContext(source="SPARQL_UPDATE", confidence=1.0)
                    self.store.add_triple(subject, predicate, obj, prov)
                    inserted_count += 1
        
        return {
            "deleted": deleted_count, 
            "inserted": inserted_count, 
            "operation": "MODIFY"
        }
    
    # =================================================================
    # Graph Management Execution Methods
    # =================================================================
    
    def _execute_create_graph(self, query: CreateGraphQuery) -> dict:
        """Execute a CREATE GRAPH query."""
        graph_uri = self._resolve_term_value(query.graph_uri, query.prefixes)
        try:
            self.store.create_graph(graph_uri)
            return {"operation": "CREATE GRAPH", "graph": graph_uri, "success": True}
        except ValueError as e:
            if query.silent:
                return {"operation": "CREATE GRAPH", "graph": graph_uri, "success": False, "reason": str(e)}
            raise
    
    def _execute_drop_graph(self, query: DropGraphQuery) -> dict:
        """Execute a DROP GRAPH query."""
        if query.target == "default":
            # Drop the default graph (clear triples with empty graph)
            self.store.clear_graph(None, silent=query.silent)
            return {"operation": "DROP", "target": "DEFAULT", "success": True}
        elif query.target == "named":
            # Drop all named graphs
            graphs = self.store.list_graphs()
            for g in graphs:
                if g:  # Skip default graph
                    self.store.drop_graph(g, silent=query.silent)
            return {"operation": "DROP", "target": "NAMED", "graphs_dropped": len([g for g in graphs if g]), "success": True}
        elif query.target == "all":
            # Drop all graphs including default
            graphs = self.store.list_graphs()
            for g in graphs:
                if g:
                    self.store.drop_graph(g, silent=query.silent)
            self.store.clear_graph(None, silent=query.silent)
            return {"operation": "DROP", "target": "ALL", "success": True}
        else:
            # Drop specific graph
            graph_uri = self._resolve_term_value(query.graph_uri, query.prefixes)
            try:
                self.store.drop_graph(graph_uri, silent=query.silent)
                return {"operation": "DROP GRAPH", "graph": graph_uri, "success": True}
            except ValueError as e:
                if query.silent:
                    return {"operation": "DROP GRAPH", "graph": graph_uri, "success": False, "reason": str(e)}
                raise
    
    def _execute_clear_graph(self, query: ClearGraphQuery) -> dict:
        """Execute a CLEAR GRAPH query."""
        if query.target == "default":
            count = self.store.clear_graph(None, silent=query.silent)
            return {"operation": "CLEAR", "target": "DEFAULT", "triples_cleared": count, "success": True}
        elif query.target == "named":
            total_cleared = 0
            graphs = self.store.list_graphs()
            for g in graphs:
                if g:  # Skip default graph
                    count = self.store.clear_graph(g, silent=query.silent)
                    total_cleared += count
            return {"operation": "CLEAR", "target": "NAMED", "triples_cleared": total_cleared, "success": True}
        elif query.target == "all":
            total_cleared = 0
            graphs = self.store.list_graphs()
            for g in graphs:
                count = self.store.clear_graph(g if g else None, silent=query.silent)
                total_cleared += count
            return {"operation": "CLEAR", "target": "ALL", "triples_cleared": total_cleared, "success": True}
        else:
            # Clear specific graph
            graph_uri = self._resolve_term_value(query.graph_uri, query.prefixes)
            try:
                count = self.store.clear_graph(graph_uri, silent=query.silent)
                return {"operation": "CLEAR GRAPH", "graph": graph_uri, "triples_cleared": count, "success": True}
            except ValueError as e:
                if query.silent:
                    return {"operation": "CLEAR GRAPH", "graph": graph_uri, "success": False, "reason": str(e)}
                raise
    
    def _execute_load(self, query: LoadQuery, provenance: Optional[ProvenanceContext] = None) -> dict:
        """Execute a LOAD query."""
        source_uri = self._resolve_term_value(query.source_uri, query.prefixes)
        graph_uri = None
        if query.graph_uri:
            graph_uri = self._resolve_term_value(query.graph_uri, query.prefixes)
        
        try:
            count = self.store.load_graph(source_uri, graph_uri, silent=query.silent)
            return {
                "operation": "LOAD", 
                "source": source_uri, 
                "graph": graph_uri,
                "triples_loaded": count, 
                "success": True
            }
        except Exception as e:
            if query.silent:
                return {
                    "operation": "LOAD",
                    "source": source_uri,
                    "graph": graph_uri,
                    "success": False,
                    "reason": str(e)
                }
            raise
    
    def _execute_copy_graph(self, query: CopyGraphQuery) -> dict:
        """Execute a COPY graph query."""
        source = None
        if not query.source_is_default and query.source_graph:
            source = self._resolve_term_value(query.source_graph, query.prefixes)
        
        dest = None
        if query.dest_graph:
            dest = self._resolve_term_value(query.dest_graph, query.prefixes)
        
        try:
            count = self.store.copy_graph(source, dest, silent=query.silent)
            return {
                "operation": "COPY",
                "source": source or "DEFAULT",
                "destination": dest or "DEFAULT",
                "triples_copied": count,
                "success": True
            }
        except ValueError as e:
            if query.silent:
                return {
                    "operation": "COPY",
                    "source": source or "DEFAULT",
                    "destination": dest or "DEFAULT",
                    "success": False,
                    "reason": str(e)
                }
            raise
    
    def _execute_move_graph(self, query: MoveGraphQuery) -> dict:
        """Execute a MOVE graph query."""
        source = None
        if not query.source_is_default and query.source_graph:
            source = self._resolve_term_value(query.source_graph, query.prefixes)
        
        dest = None
        if query.dest_graph:
            dest = self._resolve_term_value(query.dest_graph, query.prefixes)
        
        try:
            count = self.store.move_graph(source, dest, silent=query.silent)
            return {
                "operation": "MOVE",
                "source": source or "DEFAULT",
                "destination": dest or "DEFAULT",
                "triples_moved": count,
                "success": True
            }
        except ValueError as e:
            if query.silent:
                return {
                    "operation": "MOVE",
                    "source": source or "DEFAULT",
                    "destination": dest or "DEFAULT",
                    "success": False,
                    "reason": str(e)
                }
            raise
    
    def _execute_add_graph(self, query: AddGraphQuery) -> dict:
        """Execute an ADD graph query."""
        source = None
        if not query.source_is_default and query.source_graph:
            source = self._resolve_term_value(query.source_graph, query.prefixes)
        
        dest = None
        if query.dest_graph:
            dest = self._resolve_term_value(query.dest_graph, query.prefixes)
        
        try:
            count = self.store.add_graph(source, dest, silent=query.silent)
            return {
                "operation": "ADD",
                "source": source or "DEFAULT",
                "destination": dest or "DEFAULT",
                "triples_added": count,
                "success": True
            }
        except ValueError as e:
            if query.silent:
                return {
                    "operation": "ADD",
                    "source": source or "DEFAULT",
                    "destination": dest or "DEFAULT",
                    "success": False,
                    "reason": str(e)
                }
            raise
    
    def _resolve_pattern_term(
        self, 
        term: Term, 
        bindings: dict[str, Any], 
        prefixes: dict[str, str]
    ) -> Optional[str]:
        """
        Resolve a pattern term using variable bindings.
        
        Args:
            term: The term (Variable, IRI, Literal, etc.)
            bindings: Variable bindings from WHERE execution
            prefixes: Prefix mappings
            
        Returns:
            The resolved value or None if variable not bound
        """
        if isinstance(term, Variable):
            value = bindings.get(term.name)
            if value is None:
                return None
            return str(value)
        else:
            return self._resolve_term_value(term, prefixes)
    
    def _resolve_term_value(self, term: Term, prefixes: dict[str, str]) -> Any:
        """Resolve a term to its actual value, expanding prefixes."""
        if isinstance(term, IRI):
            iri = term.value
            # Check if it's a prefixed name
            if ":" in iri and not iri.startswith("http"):
                prefix, local = iri.split(":", 1)
                if prefix in prefixes:
                    return prefixes[prefix] + local
            return iri
        elif isinstance(term, Literal):
            return term.value
        elif isinstance(term, BlankNode):
            return f"_:{term.id}"
        else:
            return str(term)


def execute_sparql(
    store: "TripleStore", 
    query_string: str,
    provenance: Optional[ProvenanceContext] = None
) -> Union[pl.DataFrame, bool, dict]:
    """
    Convenience function to parse and execute a SPARQL-Star query.
    
    Args:
        store: The TripleStore to query
        query_string: SPARQL-Star query string
        provenance: Optional provenance for INSERT/DELETE operations
        
    Returns:
        Query results (DataFrame for SELECT, bool for ASK, dict for UPDATE)
    """
    from rdf_starbase.sparql.parser import parse_query
    
    query = parse_query(query_string)
    executor = SPARQLExecutor(store)
    return executor.execute(query, provenance)
