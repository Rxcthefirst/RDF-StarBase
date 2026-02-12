"""
SPARQL★ Executor for the new dictionary-encoded storage layer.

Implements efficient query execution using integer-only operations
with expansion patterns for RDF-Star metadata queries (Q6-Q12).
"""

from __future__ import annotations
from typing import Optional, Union, Any
from datetime import datetime

import polars as pl

from rdf_starbase.sparql.ast import (
    Query, SelectQuery, AskQuery, ConstructQuery, DescribeQuery,
    TriplePattern, QuotedTriplePattern,
    Variable, IRI, Literal, BlankNode,
    Filter, Comparison, LogicalExpression, FunctionCall,
    ComparisonOp, LogicalOp,
    WhereClause, ProvenanceFilter,
    Term,
    # Property path types
    PropertyPath, PathIRI, PathSequence, PathAlternative,
    PathInverse, PathMod, PathNegatedPropertySet, PathFixedLength, PropertyPathModifier,
    # Pattern types
    MinusPattern, OptionalPattern, UnionPattern,
    # Aggregate types
    AggregateExpression,
)
from rdf_starbase.storage.terms import TermDict, TermId, TermKind
from rdf_starbase.storage.quoted_triples import QtDict
from rdf_starbase.storage.facts import FactStore


class StorageExecutor:
    """
    Executes SPARQL★ queries against the new dictionary-encoded storage.
    
    Key optimizations:
    - All comparisons use integer IDs (no string comparisons in hot path)
    - Quoted triple expansion via efficient joins
    - Predicate partitioning for scan pruning
    - Lazy evaluation for query optimization
    """
    
    def __init__(
        self,
        term_dict: TermDict,
        qt_dict: QtDict,
        fact_store: FactStore
    ):
        """
        Initialize executor with storage components.
        
        Args:
            term_dict: Dictionary mapping terms to integer IDs
            qt_dict: Dictionary mapping quoted triples to IDs
            fact_store: Integer-based fact storage
        """
        self.term_dict = term_dict
        self.qt_dict = qt_dict
        self.fact_store = fact_store
        self._var_counter = 0
    
    def execute(self, query: Query) -> Union[pl.DataFrame, bool, list[tuple[str, str, str]]]:
        """
        Execute a SPARQL★ query.
        
        Args:
            query: Parsed Query AST
            
        Returns:
            DataFrame for SELECT queries, bool for ASK queries,
            list of triples for CONSTRUCT/DESCRIBE queries
        """
        if isinstance(query, SelectQuery):
            return self._execute_select(query)
        elif isinstance(query, AskQuery):
            return self._execute_ask(query)
        elif isinstance(query, ConstructQuery):
            return self._execute_construct(query)
        elif isinstance(query, DescribeQuery):
            return self._execute_describe(query)
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
        
        # Determine query type
        if isinstance(query, SelectQuery):
            return self._explain_select(query)
        elif isinstance(query, AskQuery):
            return self._explain_ask(query)
        elif isinstance(query, ConstructQuery):
            return self._explain_construct(query)
        elif isinstance(query, DescribeQuery):
            return self._explain_describe(query)
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
        
        # Analyze WHERE clause patterns
        if query.where:
            for i, pattern in enumerate(query.where.patterns):
                pattern_info = self._analyze_pattern(pattern, query.prefixes, i)
                patterns.append(pattern_info)
                total_cost += pattern_info.get("estimated_rows", 0)
                
                # Each pattern after the first requires a join
                if i > 0:
                    joins.append({
                        "type": "inner",
                        "columns": pattern_info.get("join_columns", []),
                        "pattern_idx": i,
                    })
            
            # Analyze OPTIONAL patterns
            for opt in query.where.optional_patterns:
                for j, opt_pattern in enumerate(opt.patterns):
                    pattern_info = self._analyze_pattern(opt_pattern, query.prefixes, len(patterns))
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
                    filters.append(str(f.condition))
        
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
        if query.where:
            for i, pattern in enumerate(query.where.patterns):
                patterns.append(self._analyze_pattern(pattern, query.prefixes, i))
        
        return ExplainPlan(
            query_type="ASK",
            patterns=patterns,
            limit=1,  # ASK only needs one result
            estimated_cost=sum(p.get("estimated_rows", 0) for p in patterns),
        )
    
    def _explain_construct(self, query: ConstructQuery) -> "ExplainPlan":
        """Generate execution plan for CONSTRUCT query."""
        from rdf_starbase.storage.query_context import ExplainPlan
        
        patterns = []
        if query.where:
            for i, pattern in enumerate(query.where.patterns):
                patterns.append(self._analyze_pattern(pattern, query.prefixes, i))
        
        return ExplainPlan(
            query_type="CONSTRUCT",
            patterns=patterns,
            estimated_cost=sum(p.get("estimated_rows", 0) for p in patterns),
        )
    
    def _explain_describe(self, query: DescribeQuery) -> "ExplainPlan":
        """Generate execution plan for DESCRIBE query."""
        from rdf_starbase.storage.query_context import ExplainPlan
        
        patterns = []
        if query.where:
            for i, pattern in enumerate(query.where.patterns):
                patterns.append(self._analyze_pattern(pattern, query.prefixes, i))
        
        return ExplainPlan(
            query_type="DESCRIBE",
            patterns=patterns,
            estimated_cost=sum(p.get("estimated_rows", 0) for p in patterns),
        )
    
    def _analyze_pattern(
        self,
        pattern: TriplePattern,
        prefixes: dict[str, str],
        pattern_idx: int
    ) -> dict:
        """
        Analyze a triple pattern for EXPLAIN output.
        
        Returns pattern info with:
        - type: Pattern type
        - description: Human-readable pattern
        - selectivity: Estimated selectivity (0-1)
        - estimated_rows: Estimated result rows
        - join_columns: Columns used for join with previous patterns
        """
        total_facts = len(self.fact_store._df)
        
        # Build description
        s_str = pattern.subject.name if isinstance(pattern.subject, Variable) else str(pattern.subject)
        p_str = pattern.predicate.name if isinstance(pattern.predicate, Variable) else str(pattern.predicate)
        o_str = pattern.object.name if isinstance(pattern.object, Variable) else str(pattern.object)
        
        description = f"{s_str} {p_str} {o_str}"
        
        # Estimate selectivity based on bound vs variable positions
        selectivity = 1.0
        bound_positions = 0
        join_columns = []
        
        if not isinstance(pattern.subject, Variable):
            bound_positions += 1
            selectivity *= 0.001  # Very selective on subject
        else:
            join_columns.append(pattern.subject.name)
        
        if not isinstance(pattern.predicate, Variable):
            bound_positions += 1
            selectivity *= 0.1  # Moderately selective on predicate
        else:
            join_columns.append(pattern.predicate.name)
        
        if not isinstance(pattern.object, Variable):
            bound_positions += 1
            selectivity *= 0.01  # Selective on object
        else:
            join_columns.append(pattern.object.name)
        
        # Estimate rows
        estimated_rows = max(1, int(total_facts * selectivity))
        
        return {
            "type": "TriplePattern" if not pattern.has_property_path() else "PropertyPath",
            "description": description,
            "selectivity": selectivity,
            "estimated_rows": estimated_rows,
            "bound_positions": bound_positions,
            "join_columns": join_columns,
            "pattern_idx": pattern_idx,
        }

    def _execute_select(self, query: SelectQuery) -> pl.DataFrame:
        """Execute a SELECT query."""
        # Execute WHERE clause with integer IDs
        df = self._execute_where(query.where, query.prefixes)
        
        # Check if we have aggregates
        has_aggregates = any(isinstance(v, AggregateExpression) for v in query.variables)
        
        if has_aggregates or query.group_by:
            # Handle GROUP BY and aggregates (decode happens inside _apply_aggregates)
            df = self._apply_aggregates(df, query)
            
            # Apply DISTINCT if requested
            if query.distinct:
                df = df.unique()
            
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
        else:
            # Non-aggregate path: apply LIMIT/OFFSET BEFORE decode
            # when ORDER BY and DISTINCT don't require full data.
            needs_full_data = bool(query.order_by) or query.distinct
            
            if not needs_full_data:
                # Safe to LIMIT before decode — massive speedup for large datasets
                if query.offset:
                    df = df.slice(query.offset, query.limit or len(df))
                elif query.limit:
                    df = df.head(query.limit)
            
            # Decode term IDs back to lexical forms for output
            df = self._decode_result(df)
            
            if needs_full_data:
                # Apply DISTINCT if requested
                if query.distinct:
                    df = df.unique()
                
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
        
        # Select only requested variables (or all if SELECT *)
        if not query.is_select_all():
            select_cols = []
            for v in query.variables:
                if isinstance(v, Variable):
                    if v.name in df.columns:
                        select_cols.append(v.name)
                elif isinstance(v, AggregateExpression):
                    # Use the alias name
                    if v.alias and v.alias.name in df.columns:
                        select_cols.append(v.alias.name)
            if select_cols:
                df = df.select(select_cols)
        
        return df
    
    def _apply_aggregates(self, df: pl.DataFrame, query: SelectQuery) -> pl.DataFrame:
        """Apply GROUP BY and aggregate functions.
        
        Groups on integer term IDs when possible, decoding only
        columns needed for value-sensitive aggregates (SUM, AVG, etc.)
        before aggregation, then decoding GROUP BY columns after.
        """
        if len(df) == 0:
            # Create empty result with correct columns
            result_cols = {}
            for v in query.variables:
                if isinstance(v, Variable):
                    result_cols[v.name] = pl.Series([], dtype=pl.Utf8)
                elif isinstance(v, AggregateExpression) and v.alias:
                    result_cols[v.alias.name] = pl.Series([], dtype=pl.Int64)
            return pl.DataFrame(result_cols)
        
        # Determine which columns need pre-decode for value-sensitive aggs
        VALUE_AGGS = {"SUM", "AVG", "MIN", "MAX", "GROUP_CONCAT"}
        cols_needing_decode: set[str] = set()
        for v in query.variables:
            if isinstance(v, AggregateExpression):
                if v.function.upper() in VALUE_AGGS and v.argument is not None:
                    if isinstance(v.argument, Variable) and v.argument.name in df.columns:
                        if df.schema[v.argument.name] == pl.UInt64:
                            cols_needing_decode.add(v.argument.name)
        
        # Decode only the columns that need it (before aggregation)
        if cols_needing_decode:
            # Collect unique IDs from columns needing decode
            all_ids: set[int] = set()
            for col in cols_needing_decode:
                all_ids.update(
                    tid for tid in df[col].drop_nulls().unique().to_list()
                )
            if all_ids:
                id_to_lex = self.term_dict.get_lex_batch(all_ids)
                lookup_df = pl.DataFrame({
                    "_tid": pl.Series(list(id_to_lex.keys()), dtype=pl.UInt64),
                    "_lex": pl.Series(list(id_to_lex.values()), dtype=pl.Utf8),
                })
                for col in cols_needing_decode:
                    df = (
                        df
                        .join(
                            lookup_df.rename({"_tid": col, "_lex": f"_dec_{col}"}),
                            on=col,
                            how="left",
                        )
                        .drop(col)
                        .rename({f"_dec_{col}": col})
                    )
        
        # Build the aggregate expressions
        agg_exprs = []
        for v in query.variables:
            if isinstance(v, AggregateExpression):
                agg_expr = self._build_aggregate_expr(v, df)
                if agg_expr is not None:
                    agg_exprs.append(agg_expr)
        
        if not agg_exprs:
            return self._decode_result(df)
        
        # Apply GROUP BY — group columns stay as integer IDs for speed
        if query.group_by:
            group_cols = [g.name for g in query.group_by if isinstance(g, Variable) and g.name in df.columns]
            if group_cols:
                result = df.group_by(group_cols).agg(agg_exprs)
            else:
                result = df.select(agg_exprs)
        else:
            result = df.select(agg_exprs)
        
        # Decode the grouped result (tiny — often < 100 rows)
        result = self._decode_result(result)
        return result
    
    def _build_aggregate_expr(self, agg: AggregateExpression, df: pl.DataFrame) -> Optional[pl.Expr]:
        """Build a Polars aggregate expression from an AggregateExpression."""
        func = agg.function.upper()
        alias = agg.alias.name if agg.alias else f"_{func}"
        
        if agg.argument is None:
            # COUNT(*) - count all rows
            if func == "COUNT":
                return pl.len().alias(alias)
            return None
        
        if isinstance(agg.argument, Variable):
            col_name = agg.argument.name
            if col_name not in df.columns:
                return None
            
            col = pl.col(col_name)
            
            if func == "COUNT":
                if agg.distinct:
                    return col.n_unique().alias(alias)
                else:
                    return col.count().alias(alias)
            elif func == "SUM":
                # Need to convert to numeric first
                return col.cast(pl.Float64, strict=False).sum().alias(alias)
            elif func == "AVG":
                return col.cast(pl.Float64, strict=False).mean().alias(alias)
            elif func == "MIN":
                return col.min().alias(alias)
            elif func == "MAX":
                return col.max().alias(alias)
            elif func == "GROUP_CONCAT":
                sep = agg.separator or " "
                return col.str.concat(sep).alias(alias)
            elif func == "SAMPLE":
                return col.first().alias(alias)
        
        return None
    
    def _execute_ask(self, query: AskQuery) -> bool:
        """Execute an ASK query."""
        df = self._execute_where(query.where, query.prefixes)
        return len(df) > 0
    
    def _execute_construct(self, query: ConstructQuery) -> list[tuple[str, str, str]]:
        """
        Execute a CONSTRUCT query.
        
        Returns a list of triples (s, p, o) as strings, with template variables
        substituted from the WHERE clause results.
        """
        # Execute WHERE clause to get bindings
        df = self._execute_where(query.where, query.prefixes)
        
        if len(df) == 0:
            return []
        
        # Decode term IDs to strings
        df = self._decode_result(df)
        
        # Generate triples by substituting template with each binding
        triples = []
        for row in df.iter_rows(named=True):
            for pattern in query.template:
                s = self._substitute_term(pattern.subject, row, query.prefixes)
                p = self._substitute_term(pattern.predicate, row, query.prefixes)
                o = self._substitute_term(pattern.object, row, query.prefixes)
                
                if s is not None and p is not None and o is not None:
                    triples.append((s, p, o))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_triples = []
        for t in triples:
            if t not in seen:
                seen.add(t)
                unique_triples.append(t)
        
        return unique_triples
    
    def _substitute_term(
        self,
        term: Term,
        bindings: dict[str, Any],
        prefixes: dict[str, str]
    ) -> Optional[str]:
        """Substitute a term using variable bindings."""
        if isinstance(term, Variable):
            value = bindings.get(term.name)
            return value if value is not None else None
        elif isinstance(term, IRI):
            return self._expand_iri(term.value, prefixes)
        elif isinstance(term, Literal):
            if term.language:
                return f'"{term.value}"@{term.language}'
            elif term.datatype:
                return f'"{term.value}"^^<{term.datatype}>'
            else:
                return f'"{term.value}"'
        elif isinstance(term, BlankNode):
            return f"_:{term.label}"
        return None
    
    def _execute_describe(self, query: DescribeQuery) -> list[tuple[str, str, str]]:
        """
        Execute a DESCRIBE query.
        
        Returns all triples where the described resources appear as subject or object.
        """
        # Collect resources to describe
        resources_to_describe = set()
        
        if query.where:
            # Execute WHERE to get variable bindings
            df = self._execute_where(query.where, query.prefixes)
            df = self._decode_result(df)
            
            for resource in query.resources:
                if isinstance(resource, Variable):
                    # Get all values for this variable
                    if resource.name in df.columns:
                        resources_to_describe.update(df[resource.name].to_list())
                elif isinstance(resource, IRI):
                    resources_to_describe.add(self._expand_iri(resource.value, query.prefixes))
        else:
            # No WHERE clause - just describe the listed resources
            for resource in query.resources:
                if isinstance(resource, IRI):
                    resources_to_describe.add(self._expand_iri(resource.value, query.prefixes))
        
        if not resources_to_describe:
            return []
        
        # Get all triples about these resources
        triples = []
        df = self.fact_store.scan_facts()
        
        for resource in resources_to_describe:
            resource_id = self.term_dict.lookup_iri(resource)
            if resource_id is None:
                continue
            
            # As subject
            subj_df = df.filter(pl.col("s") == resource_id)
            for row in subj_df.iter_rows(named=True):
                s_lex = self.term_dict.get_lex(row["s"]) or resource
                p_lex = self.term_dict.get_lex(row["p"]) or f"<unknown:{row['p']}>"
                o_lex = self.term_dict.get_lex(row["o"]) or f"<unknown:{row['o']}>"
                triples.append((s_lex, p_lex, o_lex))
            
            # As object
            obj_df = df.filter(pl.col("o") == resource_id)
            for row in obj_df.iter_rows(named=True):
                s_lex = self.term_dict.get_lex(row["s"]) or f"<unknown:{row['s']}>"
                p_lex = self.term_dict.get_lex(row["p"]) or f"<unknown:{row['p']}>"
                o_lex = self.term_dict.get_lex(row["o"]) or resource
                triples.append((s_lex, p_lex, o_lex))
        
        # Remove duplicates
        return list(set(triples))
    
    def _execute_where(
        self,
        where: WhereClause,
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """Execute WHERE clause, returning DataFrame with integer term IDs."""
        # Check if there's no work to do at all
        if not where.patterns and not where.union_patterns and not where.optional_patterns:
            return pl.DataFrame()
        
        result_df: Optional[pl.DataFrame] = None
        
        # Process basic triple patterns
        for i, pattern in enumerate(where.patterns):
            if isinstance(pattern, QuotedTriplePattern):
                # Handle quoted triple patterns (Q6 style)
                pattern_df = self._execute_quoted_pattern(pattern, prefixes, i)
            else:
                pattern_df = self._execute_pattern(pattern, prefixes, i)
            
            if result_df is None:
                result_df = pattern_df
            else:
                # Join on shared variables
                shared_cols = set(result_df.columns) & set(pattern_df.columns)
                shared_cols -= {"_pattern_idx"}
                
                if shared_cols:
                    result_df = result_df.join(
                        pattern_df,
                        on=list(shared_cols),
                        how="inner"
                    )
                else:
                    result_df = result_df.join(pattern_df, how="cross")
        
        # If we only have UNION patterns (no basic patterns), process UNION first
        if result_df is None and where.union_patterns:
            # Process first UNION to establish result_df
            first_union = where.union_patterns[0]
            result_df = self._apply_union_standalone(first_union, prefixes)
            
            # Process remaining UNION patterns
            for union in where.union_patterns[1:]:
                result_df = self._apply_union(result_df, union, prefixes)
        elif result_df is None:
            return pl.DataFrame()
        
        # Apply OPTIONAL patterns (left outer join) - must come before FILTER
        # so that FILTER can reference optional variables
        for optional in where.optional_patterns:
            result_df = self._apply_optional(result_df, optional, prefixes)
        
        # Apply MINUS patterns (anti-join)
        for minus in where.minus_patterns:
            result_df = self._apply_minus(result_df, minus, prefixes)
        
        # Apply FILTER clauses - after OPTIONAL so all variables are available
        for filter_clause in where.filters:
            if isinstance(filter_clause, Filter):
                result_df = self._apply_filter(result_df, filter_clause, prefixes)
            elif isinstance(filter_clause, ProvenanceFilter):
                result_df = self._apply_provenance_filter(result_df, filter_clause)
        
        # Apply UNION patterns
        for union in where.union_patterns:
            result_df = self._apply_union(result_df, union, prefixes)
        
        # Remove internal columns
        internal_cols = [c for c in result_df.columns if c.startswith("_")]
        if internal_cols:
            result_df = result_df.drop(internal_cols)
        
        return result_df
    
    def _execute_pattern(
        self,
        pattern: TriplePattern,
        prefixes: dict[str, str],
        pattern_idx: int
    ) -> pl.DataFrame:
        """Execute a triple pattern using integer comparisons."""
        # Check if this pattern has a property path predicate
        if pattern.has_property_path():
            return self._execute_property_path_pattern(pattern, prefixes, pattern_idx)
        
        # Get the facts DataFrame
        df = self.fact_store.scan_facts()
        
        # Apply filters for concrete terms using integer IDs
        if not isinstance(pattern.subject, Variable):
            term_id = self._resolve_term_id(pattern.subject, prefixes)
            if term_id is None:
                return pl.DataFrame()  # Term not in store
            df = df.filter(pl.col("s") == term_id)
        
        if not isinstance(pattern.predicate, Variable):
            term_id = self._resolve_term_id(pattern.predicate, prefixes)
            if term_id is None:
                return pl.DataFrame()
            df = df.filter(pl.col("p") == term_id)
        
        if not isinstance(pattern.object, (Variable, QuotedTriplePattern)):
            term_id = self._resolve_term_id(pattern.object, prefixes)
            if term_id is None:
                return pl.DataFrame()
            df = df.filter(pl.col("o") == term_id)
        
        # Build result with variable bindings
        renames = {}
        select_cols = []
        
        if isinstance(pattern.subject, Variable):
            renames["s"] = pattern.subject.name
            select_cols.append("s")
        
        if isinstance(pattern.predicate, Variable):
            renames["p"] = pattern.predicate.name
            select_cols.append("p")
        
        if isinstance(pattern.object, Variable):
            renames["o"] = pattern.object.name
            select_cols.append("o")
        
        # Include metadata columns for provenance filters
        for col in ["source", "confidence", "t_added", "process"]:
            if col in df.columns:
                renames[col] = f"_prov_{pattern_idx}_{col}"
                select_cols.append(col)
        
        if select_cols:
            result = df.select(select_cols).rename(renames)
        else:
            result = pl.DataFrame({"_match": [True] * len(df)})
        
        return result
    
    # =========================================================================
    # Property Path Execution
    # =========================================================================
    
    def _execute_property_path_pattern(
        self,
        pattern: TriplePattern,
        prefixes: dict[str, str],
        pattern_idx: int
    ) -> pl.DataFrame:
        """
        Execute a triple pattern with a property path predicate.
        
        Supports:
        - PathIRI: simple predicate (treated as normal pattern)
        - PathSequence: a/b/c navigation
        - PathAlternative: a|b|c any of these predicates
        - PathInverse: ^a reverse direction
        - PathMod: a*, a+, a? repetition
        - PathNegatedPropertySet: !(a|b) any predicate except these
        """
        path = pattern.predicate
        
        # Resolve subject/object
        subj_id = None
        if not isinstance(pattern.subject, Variable):
            subj_id = self._resolve_term_id(pattern.subject, prefixes)
            if subj_id is None:
                return pl.DataFrame()
        
        obj_id = None
        if not isinstance(pattern.object, Variable):
            obj_id = self._resolve_term_id(pattern.object, prefixes)
            if obj_id is None:
                return pl.DataFrame()
        
        # Execute path
        result = self._execute_path(path, subj_id, obj_id, prefixes)
        
        # Build output with variable bindings
        renames = {}
        if isinstance(pattern.subject, Variable):
            renames["start"] = pattern.subject.name
        if isinstance(pattern.object, Variable):
            renames["end"] = pattern.object.name
        
        if renames:
            result = result.rename(renames)
        
        # Select only needed columns
        select_cols = list(renames.values()) if renames else ["start", "end"]
        select_cols = [c for c in select_cols if c in result.columns]
        if select_cols:
            result = result.select(select_cols)
        
        return result.unique()
    
    def _execute_path(
        self,
        path: PropertyPath,
        start_id: Optional[int],
        end_id: Optional[int],
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """
        Execute a property path, returning (start, end) pairs.
        
        Args:
            path: The property path to execute
            start_id: Fixed start node (or None for variable)
            end_id: Fixed end node (or None for variable)
            prefixes: Namespace prefixes
            
        Returns:
            DataFrame with 'start' and 'end' columns
        """
        if isinstance(path, PathIRI):
            return self._execute_path_iri(path, start_id, end_id, prefixes)
        elif isinstance(path, PathSequence):
            return self._execute_path_sequence(path, start_id, end_id, prefixes)
        elif isinstance(path, PathAlternative):
            return self._execute_path_alternative(path, start_id, end_id, prefixes)
        elif isinstance(path, PathInverse):
            return self._execute_path_inverse(path, start_id, end_id, prefixes)
        elif isinstance(path, PathMod):
            return self._execute_path_mod(path, start_id, end_id, prefixes)
        elif isinstance(path, PathFixedLength):
            return self._execute_path_fixed_length(path, start_id, end_id, prefixes)
        elif isinstance(path, PathNegatedPropertySet):
            return self._execute_path_negated(path, start_id, end_id, prefixes)
        else:
            raise NotImplementedError(f"Path type {type(path)} not implemented")
    
    def _execute_path_iri(
        self,
        path: PathIRI,
        start_id: Optional[int],
        end_id: Optional[int],
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """Execute a simple IRI path (single predicate)."""
        pred_id = self._resolve_term_id(path.iri, prefixes)
        if pred_id is None:
            return pl.DataFrame({"start": [], "end": []})
        
        df = self.fact_store.scan_facts()
        df = df.filter(pl.col("p") == pred_id)
        
        if start_id is not None:
            df = df.filter(pl.col("s") == start_id)
        if end_id is not None:
            df = df.filter(pl.col("o") == end_id)
        
        return df.select([
            pl.col("s").alias("start"),
            pl.col("o").alias("end")
        ])
    
    def _execute_path_sequence(
        self,
        path: PathSequence,
        start_id: Optional[int],
        end_id: Optional[int],
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """Execute a path sequence (a/b/c)."""
        if not path.paths:
            return pl.DataFrame({"start": [], "end": []})
        
        # Execute first path
        result = self._execute_path(path.paths[0], start_id, None, prefixes)
        
        # Chain through remaining paths
        for i, subpath in enumerate(path.paths[1:], 1):
            is_last = i == len(path.paths) - 1
            
            # Execute next path segment
            next_end = end_id if is_last else None
            next_df = self._execute_path(subpath, None, next_end, prefixes)
            
            # Join: result.end = next_df.start
            result = result.join(
                next_df.rename({"start": "_join_start", "end": "_next_end"}),
                left_on="end",
                right_on="_join_start",
                how="inner"
            ).select([
                pl.col("start"),
                pl.col("_next_end").alias("end")
            ])
        
        return result
    
    def _execute_path_alternative(
        self,
        path: PathAlternative,
        start_id: Optional[int],
        end_id: Optional[int],
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """Execute a path alternative (a|b|c)."""
        results = []
        for subpath in path.paths:
            df = self._execute_path(subpath, start_id, end_id, prefixes)
            if len(df) > 0:
                results.append(df)
        
        if not results:
            return pl.DataFrame({"start": [], "end": []})
        
        return pl.concat(results).unique()
    
    def _execute_path_inverse(
        self,
        path: PathInverse,
        start_id: Optional[int],
        end_id: Optional[int],
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """Execute an inverse path (^a) - swap start and end."""
        # For inverse, we swap the direction
        inner = self._execute_path(path.path, end_id, start_id, prefixes)
        
        # Swap columns
        return inner.select([
            pl.col("end").alias("start"),
            pl.col("start").alias("end")
        ])
    
    def _execute_path_mod(
        self,
        path: PathMod,
        start_id: Optional[int],
        end_id: Optional[int],
        prefixes: dict[str, str],
        max_depth: int = 10
    ) -> pl.DataFrame:
        """Execute a modified path (a*, a+, a?)."""
        if path.modifier == PropertyPathModifier.ZERO_OR_ONE:
            # a? = identity OR one step
            one_step = self._execute_path(path.path, start_id, end_id, prefixes)
            
            # Add identity (start = end) for nodes
            if start_id is not None:
                identity = pl.DataFrame({"start": [start_id], "end": [start_id]})
            elif end_id is not None:
                identity = pl.DataFrame({"start": [end_id], "end": [end_id]})
            else:
                # Get all nodes
                all_nodes = self._get_all_nodes()
                identity = pl.DataFrame({"start": all_nodes, "end": all_nodes})
            
            return pl.concat([one_step, identity]).unique()
        
        elif path.modifier == PropertyPathModifier.ZERO_OR_MORE:
            # a* = identity + transitive closure
            return self._execute_transitive_closure(
                path.path, start_id, end_id, prefixes, 
                include_identity=True, max_depth=max_depth
            )
        
        elif path.modifier == PropertyPathModifier.ONE_OR_MORE:
            # a+ = at least one step, then transitive closure
            return self._execute_transitive_closure(
                path.path, start_id, end_id, prefixes,
                include_identity=False, max_depth=max_depth
            )
        
        else:
            raise NotImplementedError(f"Path modifier {path.modifier} not implemented")
    
    def _execute_transitive_closure(
        self,
        path: PropertyPath,
        start_id: Optional[int],
        end_id: Optional[int],
        prefixes: dict[str, str],
        include_identity: bool,
        max_depth: int = 10
    ) -> pl.DataFrame:
        """Compute transitive closure for path+ or path*."""
        # Get single-step edges
        edges = self._execute_path(path, None, None, prefixes)
        
        # Ensure edges have proper schema
        if len(edges) == 0:
            if include_identity:
                if start_id is not None:
                    return pl.DataFrame({
                        "start": pl.Series([start_id], dtype=pl.UInt64),
                        "end": pl.Series([start_id], dtype=pl.UInt64)
                    })
                elif end_id is not None:
                    return pl.DataFrame({
                        "start": pl.Series([end_id], dtype=pl.UInt64),
                        "end": pl.Series([end_id], dtype=pl.UInt64)
                    })
            return pl.DataFrame({
                "start": pl.Series([], dtype=pl.UInt64),
                "end": pl.Series([], dtype=pl.UInt64)
            })
        
        # Initialize reachable set with proper schema matching edges
        if include_identity:
            all_nodes = self._get_all_nodes()
            reachable = pl.DataFrame({
                "start": pl.Series(all_nodes, dtype=pl.UInt64),
                "end": pl.Series(all_nodes, dtype=pl.UInt64)
            })
            # Add single-step edges
            reachable = pl.concat([reachable, edges]).unique()
        else:
            # For ONE_OR_MORE, start with just the edges
            reachable = edges.clone()
        
        # Iteratively expand (fixed-point computation)
        for _ in range(max_depth):
            prev_len = len(reachable)
            
            # Join reachable with edges: (a, b) + (b, c) => (a, c)
            new_pairs = reachable.join(
                edges.rename({"start": "_mid", "end": "_new_end"}),
                left_on="end",
                right_on="_mid",
                how="inner"
            ).select([
                pl.col("start"),
                pl.col("_new_end").alias("end")
            ])
            
            reachable = pl.concat([reachable, new_pairs]).unique()
            
            if len(reachable) == prev_len:
                break  # Fixed point reached
        
        # Apply start/end filters
        if start_id is not None:
            reachable = reachable.filter(pl.col("start") == start_id)
        if end_id is not None:
            reachable = reachable.filter(pl.col("end") == end_id)
        
        return reachable
    
    def _execute_path_fixed_length(
        self,
        path: PathFixedLength,
        start_id: Optional[int],
        end_id: Optional[int],
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """
        Execute a fixed-length property path (path{n}, path{n,m}, path{n,}).
        
        Examples:
            foaf:knows{2}   - exactly 2 hops
            foaf:knows{2,4} - 2 to 4 hops
            foaf:knows{2,}  - 2 or more hops
        """
        min_len = path.min_length
        max_len = path.max_length
        
        # If max_len is None (unbounded), use a reasonable limit
        effective_max = max_len if max_len is not None else min_len + 10
        
        # Get single-step edges
        edges = self._execute_path(path.path, None, None, prefixes)
        
        if len(edges) == 0:
            return pl.DataFrame({
                "start": pl.Series([], dtype=pl.UInt64),
                "end": pl.Series([], dtype=pl.UInt64)
            })
        
        # Build paths of each length from min_len to effective_max
        result_dfs = []
        
        # Start with length 1
        if min_len <= 1 <= effective_max:
            length_1 = edges.clone()
            if start_id is not None:
                length_1 = length_1.filter(pl.col("start") == start_id)
            if end_id is not None:
                length_1 = length_1.filter(pl.col("end") == end_id)
            if min_len <= 1:
                result_dfs.append(length_1)
        
        # For paths of length > 1, we need to compose edges
        if effective_max >= 2:
            # current_paths holds paths of the current length
            current_paths = edges.clone()
            
            for length in range(2, effective_max + 1):
                # Extend paths by one step: (a, b) + (b, c) => (a, c)
                extended = current_paths.join(
                    edges.rename({"start": "_mid", "end": "_new_end"}),
                    left_on="end",
                    right_on="_mid",
                    how="inner"
                ).select([
                    pl.col("start"),
                    pl.col("_new_end").alias("end")
                ]).unique()
                
                current_paths = extended
                
                if len(current_paths) == 0:
                    break  # No more paths to extend
                
                # If this length is within our range, add to results
                if min_len <= length:
                    filtered = current_paths.clone()
                    if start_id is not None:
                        filtered = filtered.filter(pl.col("start") == start_id)
                    if end_id is not None:
                        filtered = filtered.filter(pl.col("end") == end_id)
                    result_dfs.append(filtered)
        
        # Handle unbounded case: continue until fixed point or max iterations
        if max_len is None and len(current_paths) > 0:
            for _ in range(10):  # Additional iterations for unbounded
                extended = current_paths.join(
                    edges.rename({"start": "_mid", "end": "_new_end"}),
                    left_on="end",
                    right_on="_mid",
                    how="inner"
                ).select([
                    pl.col("start"),
                    pl.col("_new_end").alias("end")
                ]).unique()
                
                if len(extended) == 0:
                    break
                
                current_paths = extended
                
                filtered = current_paths.clone()
                if start_id is not None:
                    filtered = filtered.filter(pl.col("start") == start_id)
                if end_id is not None:
                    filtered = filtered.filter(pl.col("end") == end_id)
                result_dfs.append(filtered)
        
        if not result_dfs:
            return pl.DataFrame({
                "start": pl.Series([], dtype=pl.UInt64),
                "end": pl.Series([], dtype=pl.UInt64)
            })
        
        return pl.concat(result_dfs).unique()
    
    def _execute_path_negated(
        self,
        path: PathNegatedPropertySet,
        start_id: Optional[int],
        end_id: Optional[int],
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """Execute a negated property set !(a|b) - any predicate except these."""
        # Get excluded predicate IDs
        excluded_ids = set()
        for iri in path.iris:
            pred_id = self._resolve_term_id(iri, prefixes)
            if pred_id is not None:
                excluded_ids.add(pred_id)
        
        df = self.fact_store.scan_facts()
        
        # Exclude the specified predicates
        if excluded_ids:
            df = df.filter(~pl.col("p").is_in(list(excluded_ids)))
        
        if start_id is not None:
            df = df.filter(pl.col("s") == start_id)
        if end_id is not None:
            df = df.filter(pl.col("o") == end_id)
        
        return df.select([
            pl.col("s").alias("start"),
            pl.col("o").alias("end")
        ])
    
    def _get_all_nodes(self) -> list[int]:
        """Get all unique node IDs (subjects and objects)."""
        df = self.fact_store.scan_facts()
        subjects = df.select("s").unique()["s"].to_list()
        objects = df.select("o").unique()["o"].to_list()
        return list(set(subjects) | set(objects))
    
    # =========================================================================
    # MINUS Pattern Execution
    # =========================================================================
    
    def _apply_minus(
        self,
        result_df: pl.DataFrame,
        minus: MinusPattern,
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """
        Apply a MINUS pattern to filter out matching solutions.
        
        MINUS implements set difference: returns rows from result_df
        that don't have compatible bindings in the minus pattern.
        
        SPARQL semantics: A solution µ1 is removed if there exists
        a solution µ2 in the MINUS clause such that:
        - µ1 and µ2 are compatible (agree on shared variables)
        - dom(µ1) ∩ dom(µ2) ≠ ∅ (they share at least one variable)
        """
        if len(result_df) == 0:
            return result_df
        
        # Execute the MINUS patterns to get solutions to exclude
        minus_df: Optional[pl.DataFrame] = None
        
        for i, pattern in enumerate(minus.patterns):
            if isinstance(pattern, QuotedTriplePattern):
                pattern_df = self._execute_quoted_pattern(pattern, prefixes, i)
            else:
                pattern_df = self._execute_pattern(pattern, prefixes, i)
            
            if minus_df is None:
                minus_df = pattern_df
            else:
                # Join on shared variables
                shared_cols = set(minus_df.columns) & set(pattern_df.columns)
                shared_cols -= {"_pattern_idx"}
                
                if shared_cols:
                    minus_df = minus_df.join(
                        pattern_df,
                        on=list(shared_cols),
                        how="inner"
                    )
                else:
                    minus_df = minus_df.join(pattern_df, how="cross")
        
        if minus_df is None or len(minus_df) == 0:
            return result_df
        
        # Apply filters from the MINUS clause
        for filter_clause in minus.filters:
            if isinstance(filter_clause, Filter):
                minus_df = self._apply_filter(minus_df, filter_clause, prefixes)
        
        # Find shared variables between result and minus
        shared_vars = set(result_df.columns) & set(minus_df.columns)
        shared_vars = {c for c in shared_vars if not c.startswith("_")}
        
        if not shared_vars:
            # No shared variables - MINUS has no effect (SPARQL semantics)
            return result_df
        
        # Perform anti-join: keep rows from result_df that don't match minus_df
        # We do this with a left join and then filter for nulls
        shared_list = list(shared_vars)
        
        # Add a marker column to minus_df to detect matches
        minus_df = minus_df.select(shared_list).unique()
        minus_df = minus_df.with_columns(pl.lit(True).alias("_minus_match"))
        
        # Left join
        result_df = result_df.join(
            minus_df,
            on=shared_list,
            how="left"
        )
        
        # Keep only rows where there was no match
        result_df = result_df.filter(pl.col("_minus_match").is_null())
        result_df = result_df.drop("_minus_match")
        
        return result_df
    
    def _apply_optional(
        self,
        result_df: pl.DataFrame,
        optional: OptionalPattern,
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """
        Apply an OPTIONAL pattern using left outer join.
        
        OPTIONAL { ... } patterns add bindings when matched but keep
        rows even when no match exists (with NULL for optional columns).
        """
        if len(result_df) == 0:
            return result_df
        
        # Execute the optional patterns
        optional_df: Optional[pl.DataFrame] = None
        
        for i, pattern in enumerate(optional.patterns):
            if isinstance(pattern, QuotedTriplePattern):
                pattern_df = self._execute_quoted_pattern(pattern, prefixes, 1000 + i)
            elif isinstance(pattern, TriplePattern):
                pattern_df = self._execute_pattern(pattern, prefixes, 1000 + i)
            else:
                continue  # Skip nested patterns for now
            
            if optional_df is None:
                optional_df = pattern_df
            else:
                shared_cols = set(optional_df.columns) & set(pattern_df.columns)
                shared_cols -= {"_pattern_idx"}
                shared_cols = {c for c in shared_cols if not c.startswith("_")}
                
                if shared_cols:
                    optional_df = optional_df.join(pattern_df, on=list(shared_cols), how="inner")
                else:
                    optional_df = optional_df.join(pattern_df, how="cross")
        
        if optional_df is None or len(optional_df) == 0:
            return result_df
        
        # Apply filters within the optional block
        for filter_clause in optional.filters:
            if isinstance(filter_clause, Filter):
                optional_df = self._apply_filter(optional_df, filter_clause, prefixes)
        
        # Remove internal columns from optional_df
        internal_cols = [c for c in optional_df.columns if c.startswith("_")]
        if internal_cols:
            optional_df = optional_df.drop(internal_cols)
        
        # Find shared columns for the join
        shared_cols = set(result_df.columns) & set(optional_df.columns)
        shared_cols = {c for c in shared_cols if not c.startswith("_")}
        
        if shared_cols:
            # Left outer join - keep all rows from result_df, add optional columns where matched
            return result_df.join(optional_df, on=list(shared_cols), how="left")
        else:
            # No shared columns - return original
            return result_df
    
    def _apply_union_standalone(
        self,
        union: UnionPattern,
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """
        Apply a UNION pattern as the primary query (no prior results).
        
        This is used when WHERE clause starts with a UNION.
        """
        union_results = []
        
        def build_where_clause(alternative):
            """Build a WhereClause from either list or dict alternative."""
            if isinstance(alternative, dict):
                return WhereClause(
                    patterns=alternative.get('patterns', []),
                    filters=alternative.get('filters', []),
                    binds=alternative.get('binds', [])
                )
            else:
                # Legacy format: just a list of patterns
                return WhereClause(patterns=alternative)
        
        for i, alternative in enumerate(union.alternatives):
            # Execute each alternative as a mini WHERE clause
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
            missing_cols = all_columns - set(r.columns)
            if missing_cols:
                # Add null columns for missing variables
                for col in missing_cols:
                    r = r.with_columns(pl.lit(None).alias(col))
            aligned_results.append(r.select(sorted(all_columns)))
        
        return pl.concat(aligned_results)
    
    def _apply_union(
        self,
        result_df: pl.DataFrame,
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
        
        def build_where_clause(alternative):
            """Build a WhereClause from either list or dict alternative."""
            if isinstance(alternative, dict):
                return WhereClause(
                    patterns=alternative.get('patterns', []),
                    filters=alternative.get('filters', []),
                    binds=alternative.get('binds', [])
                )
            else:
                # Legacy format: just a list of patterns
                return WhereClause(patterns=alternative)
        
        for i, alternative in enumerate(union.alternatives):
            # Execute each alternative as a mini WHERE clause
            alt_where = build_where_clause(alternative)
            alt_df = self._execute_where(alt_where, prefixes)
            
            if len(alt_df) > 0:
                union_results.append(alt_df)
        
        if not union_results:
            return result_df
        
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
                missing_cols = all_columns - set(r.columns)
                if missing_cols:
                    # Add null columns for missing variables
                    for col in missing_cols:
                        r = r.with_columns(pl.lit(None).alias(col))
                aligned_results.append(r.select(sorted(all_columns)))
            
            union_df = pl.concat(aligned_results)
        
        # If we have existing results, combine them with union
        if len(result_df) > 0 and len(result_df.columns) > 0:
            # Find shared columns
            shared_cols = set(result_df.columns) & set(union_df.columns)
            shared_cols = {c for c in shared_cols if not c.startswith("_")}
            
            if shared_cols:
                # Join union results with existing results
                return result_df.join(union_df, on=list(shared_cols), how="inner")
            else:
                # No shared columns - cross join
                return result_df.join(union_df, how="cross")
        else:
            return union_df

    def _execute_quoted_pattern(
        self,
        pattern: QuotedTriplePattern,
        prefixes: dict[str, str],
        pattern_idx: int
    ) -> pl.DataFrame:
        """
        Execute a quoted triple pattern (Q6: << s p o >> ?mp ?mo).
        
        This is the key RDF★ expansion pattern that finds metadata
        about quoted triples.
        """
        # Get the quoted triple components
        s_term = pattern.subject
        p_term = pattern.predicate  
        o_term = pattern.object
        
        # Check if the quoted triple itself is concrete or has variables
        qt_s_id = None if isinstance(s_term, Variable) else self._resolve_term_id(s_term, prefixes)
        qt_p_id = None if isinstance(p_term, Variable) else self._resolve_term_id(p_term, prefixes)
        qt_o_id = None if isinstance(o_term, Variable) else self._resolve_term_id(o_term, prefixes)
        
        # If all components are concrete, look up the qt_id
        if qt_s_id is not None and qt_p_id is not None and qt_o_id is not None:
            qt_id = self.qt_dict.lookup_id(qt_s_id, qt_p_id, qt_o_id)
            if qt_id is None:
                return pl.DataFrame()  # Quoted triple not found
            
            # Find facts where this qt_id appears as subject
            df = self.fact_store.scan_facts().filter(pl.col("s") == qt_id)
        else:
            # Need to join with qt_dict to expand
            df = self._expand_qt_metadata(qt_s_id, qt_p_id, qt_o_id)
        
        # Rename predicate/object vars if they exist in outer pattern
        # (for << s p o >> ?mp ?mo patterns)
        renames = {}
        select_cols = []
        
        # The quoted triple's metadata predicate/object
        if "p" in df.columns:
            renames["p"] = "mp" if pattern_idx == 0 else f"mp_{pattern_idx}"
            select_cols.append("p")
        if "o" in df.columns:
            renames["o"] = "mo" if pattern_idx == 0 else f"mo_{pattern_idx}"
            select_cols.append("o")
        
        # Include base triple components if variables
        if isinstance(s_term, Variable) and "base_s" in df.columns:
            renames["base_s"] = s_term.name
            select_cols.append("base_s")
        if isinstance(p_term, Variable) and "base_p" in df.columns:
            renames["base_p"] = p_term.name
            select_cols.append("base_p")
        if isinstance(o_term, Variable) and "base_o" in df.columns:
            renames["base_o"] = o_term.name
            select_cols.append("base_o")
        
        if select_cols:
            result = df.select(select_cols).rename(renames)
        else:
            result = df
        
        return result
    
    def _expand_qt_metadata(
        self,
        qt_s_id: Optional[TermId],
        qt_p_id: Optional[TermId],
        qt_o_id: Optional[TermId]
    ) -> pl.DataFrame:
        """
        Expand quoted triple metadata with optional filters on components.
        
        This implements the RDF★ expansion join: find metadata about
        quoted triples, optionally filtering by their s/p/o components.
        """
        # Get facts about quoted triples (metadata facts)
        df = self.fact_store.scan_metadata_facts()
        
        if len(df) == 0:
            return df
        
        # Expand qt_id to base (s, p, o)
        df = self.fact_store.expand_qt_metadata(df, self.qt_dict)
        
        # Apply filters on quoted triple components
        if qt_s_id is not None:
            df = df.filter(pl.col("base_s") == qt_s_id)
        if qt_p_id is not None:
            df = df.filter(pl.col("base_p") == qt_p_id)
        if qt_o_id is not None:
            df = df.filter(pl.col("base_o") == qt_o_id)
        
        return df
    
    def _resolve_term_id(
        self,
        term: Term,
        prefixes: dict[str, str]
    ) -> Optional[TermId]:
        """Resolve a term to its integer ID."""
        if isinstance(term, IRI):
            iri = self._expand_iri(term.value, prefixes)
            return self.term_dict.lookup_iri(iri)
        elif isinstance(term, Literal):
            return self.term_dict.lookup_literal(
                str(term.value),
                term.datatype,
                term.language
            )
        elif isinstance(term, BlankNode):
            return self.term_dict.lookup_bnode(term.label)
        return None
    
    def _expand_iri(self, iri: str, prefixes: dict[str, str]) -> str:
        """Expand prefixed IRI to full form."""
        if ":" in iri and not iri.startswith("http"):
            prefix, local = iri.split(":", 1)
            if prefix in prefixes:
                return prefixes[prefix] + local
        return iri
    
    def _decode_result(self, df: pl.DataFrame) -> pl.DataFrame:
        """Decode integer term IDs back to lexical forms for output.
        
        Uses vectorized Polars join instead of Python loops:
        1. Collect all unique term IDs across all UInt64 columns
        2. Build a single lookup DataFrame (term_id → lex)
        3. Left-join each column against it
        
        This is O(unique_ids) rather than O(total_rows × columns).
        """
        if len(df) == 0:
            return df
        
        # Find UInt64 columns that need decoding
        id_columns = [
            c for c in df.columns
            if not c.startswith("_prov") and df.schema[c] == pl.UInt64
        ]
        if not id_columns:
            return df
        
        # Collect all unique term IDs across all columns to decode
        all_ids_set: set[int] = set()
        for col in id_columns:
            all_ids_set.update(
                tid for tid in df[col].drop_nulls().unique().to_list()
            )
        
        if not all_ids_set:
            # All nulls — just cast columns to Utf8
            return df.with_columns([
                pl.col(c).cast(pl.Utf8) for c in id_columns
            ])
        
        # Build lookup dict: term_id → lex
        id_to_lex = self.term_dict.get_lex_batch(all_ids_set)
        
        # Build a Polars lookup DataFrame
        lookup_ids = list(id_to_lex.keys())
        lookup_lexes = list(id_to_lex.values())
        
        lookup_df = pl.DataFrame({
            "_tid": pl.Series(lookup_ids, dtype=pl.UInt64),
            "_lex": pl.Series(lookup_lexes, dtype=pl.Utf8),
        })
        
        # Join each column against the lookup
        for col in id_columns:
            df = (
                df
                .join(
                    lookup_df.rename({"_tid": col, "_lex": f"_dec_{col}"}),
                    on=col,
                    how="left",
                )
                .drop(col)
                .rename({f"_dec_{col}": col})
            )
        
        return df
    
    def _apply_filter(
        self,
        df: pl.DataFrame,
        filter_clause: Filter,
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """
        Apply a FILTER clause.
        
        For filters involving literal comparisons (especially numeric),
        we need to decode the term IDs to their actual values first,
        apply the filter, then re-encode if needed.
        """
        # Get variables used in the filter expression
        filter_vars = self._get_filter_variables(filter_clause.expression)
        
        # Check if all required variables exist in the dataframe
        missing_vars = [v for v in filter_vars if v.name not in df.columns]
        if missing_vars:
            # If any filter variable is missing, no rows can match
            return df.head(0)
        
        # Decode the filter-relevant columns to actual values
        decoded_df = df.clone()
        for var in filter_vars:
            if var.name in decoded_df.columns:
                col = decoded_df[var.name]
                if col.dtype == pl.UInt64:
                    # Decode this column to its lexical values
                    decoded_values = []
                    for term_id in col.to_list():
                        if term_id is None:
                            decoded_values.append(None)
                        else:
                            lex = self.term_dict.get_lex(term_id)
                            # Try to convert to numeric if possible
                            if lex is not None:
                                try:
                                    # Try int first, then float
                                    decoded_values.append(int(lex))
                                except ValueError:
                                    try:
                                        decoded_values.append(float(lex))
                                    except ValueError:
                                        decoded_values.append(lex)
                            else:
                                decoded_values.append(None)
                    decoded_df = decoded_df.with_columns(
                        pl.Series(f"_decoded_{var.name}", decoded_values)
                    )
                else:
                    # Column already decoded (e.g., string type) - just alias it
                    decoded_df = decoded_df.with_columns(
                        pl.col(var.name).alias(f"_decoded_{var.name}")
                    )
        
        # Build filter expression using decoded columns
        expr = self._build_filter_expression_decoded(filter_clause.expression, prefixes)
        if expr is not None:
            # Filter using decoded values, keep original columns
            filtered = decoded_df.filter(expr)
            # Drop the decoded columns
            decoded_cols = [c for c in filtered.columns if c.startswith("_decoded_")]
            if decoded_cols:
                filtered = filtered.drop(decoded_cols)
            return filtered
        
        return df
    
    def _get_filter_variables(self, expr) -> set:
        """Get all variables referenced in a filter expression."""
        from rdf_starbase.sparql.ast import Variable, Comparison, LogicalExpression
        
        variables = set()
        if isinstance(expr, Variable):
            variables.add(expr)
        elif isinstance(expr, Comparison):
            if isinstance(expr.left, Variable):
                variables.add(expr.left)
            if isinstance(expr.right, Variable):
                variables.add(expr.right)
        elif isinstance(expr, LogicalExpression):
            for operand in expr.operands:
                variables.update(self._get_filter_variables(operand))
        return variables
    
    def _build_filter_expression_decoded(
        self,
        expr: Union[Comparison, LogicalExpression, FunctionCall],
        prefixes: dict[str, str]
    ) -> Optional[pl.Expr]:
        """Build Polars filter expression using decoded column names."""
        if isinstance(expr, Comparison):
            left = self._term_to_expr_decoded(expr.left, prefixes)
            right = self._term_to_expr_decoded(expr.right, prefixes)
            
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
                self._build_filter_expression_decoded(op, prefixes) 
                for op in expr.operands
            ]
            valid_exprs = [e for e in operand_exprs if e is not None]
            
            if not valid_exprs:
                return None
            
            if expr.operator == LogicalOp.AND:
                result = valid_exprs[0]
                for e in valid_exprs[1:]:
                    result = result & e
                return result
            elif expr.operator == LogicalOp.OR:
                result = valid_exprs[0]
                for e in valid_exprs[1:]:
                    result = result | e
                return result
            elif expr.operator == LogicalOp.NOT:
                return ~valid_exprs[0]
        
        return None
    
    def _term_to_expr_decoded(
        self,
        term: Term,
        prefixes: dict[str, str]
    ) -> Optional[pl.Expr]:
        """Convert a term to a Polars expression using decoded column names."""
        if isinstance(term, Variable):
            # Use the decoded column if it exists
            return pl.col(f"_decoded_{term.name}")
        elif isinstance(term, Literal):
            # Convert literal to appropriate type
            try:
                return pl.lit(int(term.value))
            except ValueError:
                try:
                    return pl.lit(float(term.value))
                except ValueError:
                    return pl.lit(term.value)
        elif isinstance(term, IRI):
            # For IRI comparisons, use the full IRI string
            iri_str = self._expand_iri(term, prefixes)
            return pl.lit(iri_str)
        return None
    
    def _apply_provenance_filter(
        self,
        df: pl.DataFrame,
        filter_clause: ProvenanceFilter
    ) -> pl.DataFrame:
        """Apply provenance FILTER (FILTER_CONFIDENCE, FILTER_SOURCE, etc.)."""
        field = filter_clause.provenance_field
        prov_cols = [c for c in df.columns if c.endswith(f"_{field}")]
        
        if not prov_cols:
            return df
        
        expr = filter_clause.expression
        if isinstance(expr, Comparison):
            combined_expr = None
            for col in prov_cols:
                col_expr = self._build_provenance_comparison(expr, col)
                if col_expr is not None:
                    if combined_expr is None:
                        combined_expr = col_expr
                    else:
                        combined_expr = combined_expr | col_expr
            
            if combined_expr is not None:
                return df.filter(combined_expr)
        
        return df
    
    def _build_provenance_comparison(
        self,
        expr: Comparison,
        prov_col: str
    ) -> Optional[pl.Expr]:
        """Build comparison expression for provenance filtering."""
        if isinstance(expr.left, Variable):
            left = pl.col(prov_col)
            right = self._literal_to_polars(expr.right)
        elif isinstance(expr.right, Variable):
            left = self._literal_to_polars(expr.left)
            right = pl.col(prov_col)
        else:
            left = self._literal_to_polars(expr.left)
            right = self._literal_to_polars(expr.right)
        
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
    
    def _build_filter_expression(
        self,
        expr: Union[Comparison, LogicalExpression, FunctionCall],
        prefixes: dict[str, str]
    ) -> Optional[pl.Expr]:
        """Build Polars filter expression from SPARQL filter AST."""
        if isinstance(expr, Comparison):
            left = self._term_to_expr(expr.left, prefixes)
            right = self._term_to_expr(expr.right, prefixes)
            
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
                self._build_filter_expression(op, prefixes) 
                for op in expr.operands
            ]
            valid_exprs = [e for e in operand_exprs if e is not None]
            
            if not valid_exprs:
                return None
            
            if expr.operator == LogicalOp.AND:
                result = valid_exprs[0]
                for e in valid_exprs[1:]:
                    result = result & e
                return result
            elif expr.operator == LogicalOp.OR:
                result = valid_exprs[0]
                for e in valid_exprs[1:]:
                    result = result | e
                return result
            elif expr.operator == LogicalOp.NOT:
                return ~valid_exprs[0]
        
        return None
    
    def _term_to_expr(
        self,
        term: Term,
        prefixes: dict[str, str]
    ) -> Optional[pl.Expr]:
        """Convert a term to a Polars expression."""
        if isinstance(term, Variable):
            return pl.col(term.name)
        elif isinstance(term, Literal):
            return pl.lit(term.value)
        elif isinstance(term, IRI):
            # For IRI comparisons, use the term ID
            term_id = self._resolve_term_id(term, prefixes)
            return pl.lit(term_id) if term_id else None
        return None
    
    def _literal_to_polars(self, term: Term) -> Optional[Any]:
        """Convert a literal term to a Polars literal."""
        if isinstance(term, Literal):
            return pl.lit(term.value)
        elif isinstance(term, Variable):
            return pl.col(term.name)
        return None


# === RDF★ Expansion Query Patterns ===
# These implement Q6-Q12 from the SPARQL-Star test suite

class ExpansionPatterns:
    """
    Factory for common RDF★ expansion query patterns.
    
    These patterns efficiently query metadata about quoted triples
    and expand them back to (s, p, o) components.
    """
    
    def __init__(
        self,
        term_dict: TermDict,
        qt_dict: QtDict,
        fact_store: FactStore
    ):
        self.term_dict = term_dict
        self.qt_dict = qt_dict
        self.fact_store = fact_store
    
    def q6_metadata_for_triple(
        self,
        subject: str,
        predicate: str,
        obj: str
    ) -> pl.DataFrame:
        """
        Q6: Fetch all metadata about a specific quoted triple.
        
        SELECT ?mp ?mo WHERE {
            << subject predicate object >> ?mp ?mo .
        }
        """
        # Look up term IDs
        s_id = self.term_dict.lookup_iri(subject)
        p_id = self.term_dict.lookup_iri(predicate)
        o_id = self.term_dict.lookup_iri(obj)
        
        if s_id is None or p_id is None or o_id is None:
            return pl.DataFrame({"mp": [], "mo": []})
        
        # Look up the quoted triple
        qt_id = self.qt_dict.get_id(s_id, p_id, o_id)
        if qt_id is None:
            return pl.DataFrame({"mp": [], "mo": []})
        
        # Find facts where qt_id is the subject
        df = self.fact_store.scan_facts().filter(pl.col("s") == qt_id)
        
        # Decode predicates and objects
        result = []
        for row in df.iter_rows(named=True):
            mp = self.term_dict.get_lex(row["p"])
            mo = self.term_dict.get_lex(row["o"])
            result.append({"mp": mp, "mo": mo})
        
        return pl.DataFrame(result) if result else pl.DataFrame({"mp": [], "mo": []})
    
    def q7_expand_by_source(self, source_uri: str) -> pl.DataFrame:
        """
        Q7: Given a source, find all quoted triples derived from it
        and expand them to base (s, p, o).
        
        SELECT ?s ?p ?o WHERE {
            ?qt prov:wasDerivedFrom source_uri .
            # Expand the quoted triple
        }
        """
        # Look up the source term
        source_id = self.term_dict.lookup_iri(source_uri)
        if source_id is None:
            return pl.DataFrame({"s": [], "p": [], "o": []})
        
        # Look up prov:wasDerivedFrom predicate
        prov_pred = self.term_dict.lookup_iri(
            "http://www.w3.org/ns/prov#wasDerivedFrom"
        )
        if prov_pred is None:
            return pl.DataFrame({"s": [], "p": [], "o": []})
        
        # Find metadata facts with this predicate and source
        df = self.fact_store.scan_metadata_facts()
        df = df.filter(
            (pl.col("p") == prov_pred) & 
            (pl.col("o") == source_id)
        )
        
        # Expand qt_ids to (s, p, o)
        df = self.fact_store.expand_metadata_df(df)
        
        # Decode to lexical forms
        result = []
        for row in df.iter_rows(named=True):
            s = self.term_dict.get_lex(row["base_s"])
            p = self.term_dict.get_lex(row["base_p"])
            o = self.term_dict.get_lex(row["base_o"])
            result.append({"s": s, "p": p, "o": o})
        
        return pl.DataFrame(result) if result else pl.DataFrame({"s": [], "p": [], "o": []})
    
    def q8_expand_by_activity(self, activity_uri: str) -> pl.DataFrame:
        """
        Q8: List all statements generated by a given run/activity and expand.
        
        SELECT ?s ?p ?o WHERE {
            ?qt prov:wasGeneratedBy activity_uri .
        }
        """
        activity_id = self.term_dict.lookup_iri(activity_uri)
        if activity_id is None:
            return pl.DataFrame({"s": [], "p": [], "o": []})
        
        gen_pred = self.term_dict.lookup_iri(
            "http://www.w3.org/ns/prov#wasGeneratedBy"
        )
        if gen_pred is None:
            return pl.DataFrame({"s": [], "p": [], "o": []})
        
        df = self.fact_store.scan_metadata_facts()
        df = df.filter(
            (pl.col("p") == gen_pred) & 
            (pl.col("o") == activity_id)
        )
        
        df = self.fact_store.expand_metadata_df(df)
        
        result = []
        for row in df.iter_rows(named=True):
            s = self.term_dict.get_lex(row["base_s"])
            p = self.term_dict.get_lex(row["base_p"])
            o = self.term_dict.get_lex(row["base_o"])
            result.append({"s": s, "p": p, "o": o})
        
        return pl.DataFrame(result) if result else pl.DataFrame({"s": [], "p": [], "o": []})
    
    def q9_filter_by_confidence(
        self,
        min_confidence: float,
        max_confidence: Optional[float] = None,
        expand_lex: bool = True,
    ) -> pl.DataFrame:
        """
        Q9: Filter statements by confidence and expand.
        
        SELECT ?s ?p ?o ?c WHERE {
            ?qt ex:confidence ?c .
            FILTER(?c > min_confidence)
        }
        
        Uses pure Polars join for vectorized performance.
        
        Args:
            min_confidence: Minimum confidence threshold (exclusive)
            max_confidence: Maximum confidence threshold (inclusive, optional)
            expand_lex: If True, return lexical forms. If False, return term IDs (faster).
        """
        conf_pred = self.term_dict.lookup_iri("http://example.org/confidence")
        if conf_pred is None:
            cols = {"s": [], "p": [], "o": [], "c": []}
            return pl.DataFrame(cols)
        
        # Get confidence facts
        df = self.fact_store.scan_metadata_facts()
        df = df.filter(pl.col("p") == conf_pred)
        
        if df.is_empty():
            cols = {"s": [], "p": [], "o": [], "c": []}
            return pl.DataFrame(cols)
        
        # Build float map as a Polars DataFrame for vectorized join
        float_map = self.term_dict.build_literal_to_float_map()
        if not float_map:
            cols = {"s": [], "p": [], "o": [], "c": []}
            return pl.DataFrame(cols)
        
        # Create lookup DataFrame: term_id -> float value
        map_df = pl.DataFrame({
            "term_id": list(float_map.keys()),
            "conf_value": list(float_map.values()),
        }).cast({"term_id": pl.UInt64, "conf_value": pl.Float64})
        
        # Join to get confidence values (pure Polars, no Python iteration!)
        df = df.join(map_df, left_on="o", right_on="term_id", how="inner")
        
        # Filter by confidence threshold (vectorized!)
        df = df.filter(pl.col("conf_value") > min_confidence)
        if max_confidence is not None:
            df = df.filter(pl.col("conf_value") <= max_confidence)
        
        if df.is_empty():
            cols = {"s": [], "p": [], "o": [], "c": []}
            return pl.DataFrame(cols)
        
        # Expand to get base triple components
        df = self.fact_store.expand_metadata_df(df)
        
        if not expand_lex:
            # Return term IDs directly (much faster for large results)
            return df.select([
                pl.col("base_s").alias("s"),
                pl.col("base_p").alias("p"),
                pl.col("base_o").alias("o"),
                pl.col("conf_value").alias("c"),
            ])
        
        # Map term IDs to lexical forms using vectorized lookup
        s_lex = self.term_dict.get_lex_series(df["base_s"])
        p_lex = self.term_dict.get_lex_series(df["base_p"])
        o_lex = self.term_dict.get_lex_series(df["base_o"])
        
        return pl.DataFrame({
            "s": s_lex,
            "p": p_lex,
            "o": o_lex,
            "c": df["conf_value"],
        })
    
    def q9_count_by_confidence(
        self,
        min_confidence: float,
        max_confidence: Optional[float] = None,
    ) -> int:
        """
        Count statements above confidence threshold (fast).
        
        SELECT (COUNT(*) as ?count) WHERE {
            ?qt ex:confidence ?c .
            FILTER(?c > min_confidence)
        }
        """
        df = self.q9_filter_by_confidence(min_confidence, max_confidence, expand_lex=False)
        return len(df)
    
    def q9_native_filter_by_confidence(
        self,
        min_confidence: float,
        max_confidence: Optional[float] = None,
        expand_lex: bool = True,
    ) -> pl.DataFrame:
        """
        Q9 (Native): Filter facts by confidence using native column.
        
        This is the FAST version that uses the native `confidence` column
        in the FactStore schema, avoiding any string parsing or joins.
        
        Use this when facts were ingested with `add_facts_with_provenance()`
        which stores confidence directly in the native column.
        
        SELECT ?s ?p ?o ?c WHERE {
            FILTER(confidence > min_confidence)
        }
        
        Args:
            min_confidence: Minimum confidence threshold (exclusive)
            max_confidence: Maximum confidence threshold (inclusive, optional)
            expand_lex: If True, return lexical forms. If False, return term IDs.
        """
        # Pure vectorized scan on native column - no joins!
        df = self.fact_store.scan_by_confidence(
            min_confidence, 
            max_confidence,
            include_metadata=False,  # Only base facts
        )
        
        if df.is_empty():
            return pl.DataFrame({"s": [], "p": [], "o": [], "c": []})
        
        if not expand_lex:
            return df.select([
                pl.col("s"),
                pl.col("p"),
                pl.col("o"),
                pl.col("confidence").alias("c"),
            ])
        
        # Map term IDs to lexical forms
        s_lex = self.term_dict.get_lex_series(df["s"])
        p_lex = self.term_dict.get_lex_series(df["p"])
        o_lex = self.term_dict.get_lex_series(df["o"])
        
        return pl.DataFrame({
            "s": s_lex,
            "p": p_lex,
            "o": o_lex,
            "c": df["confidence"],
        })
    
    def q9_native_count(
        self,
        min_confidence: float,
        max_confidence: Optional[float] = None,
    ) -> int:
        """
        Count facts by confidence using native column (fastest).
        """
        df = self.fact_store.scan_by_confidence(
            min_confidence,
            max_confidence,
            include_metadata=False,
        )
        return len(df)
    
    def q10_filter_by_time_range(
        self,
        start: datetime,
        end: datetime
    ) -> pl.DataFrame:
        """
        Q10: Filter by time range on metadata.
        
        SELECT ?qt ?t WHERE {
            ?qt prov:generatedAtTime ?t .
            FILTER(?t >= start && ?t < end)
        }
        """
        time_pred = self.term_dict.lookup_iri(
            "http://www.w3.org/ns/prov#generatedAtTime"
        )
        if time_pred is None:
            return pl.DataFrame({"qt": [], "t": []})
        
        df = self.fact_store.scan_metadata_facts()
        df = df.filter(pl.col("p") == time_pred)
        
        result = []
        for row in df.iter_rows(named=True):
            time_lex = self.term_dict.get_lex(row["o"])
            if time_lex is None:
                continue
            try:
                # Parse ISO datetime
                t = datetime.fromisoformat(time_lex.replace("Z", "+00:00"))
                if start <= t < end:
                    qt_lex = self._qt_to_string(row["s"])
                    result.append({"qt": qt_lex, "t": time_lex})
            except ValueError:
                continue
        
        return pl.DataFrame(result) if result else pl.DataFrame({"qt": [], "t": []})
    
    def q11_count_by_source(self) -> pl.DataFrame:
        """
        Q11: Count statements per source.
        
        SELECT ?src (COUNT(?qt) AS ?n) WHERE {
            ?qt prov:wasDerivedFrom ?src .
        } GROUP BY ?src ORDER BY DESC(?n)
        """
        prov_pred = self.term_dict.lookup_iri(
            "http://www.w3.org/ns/prov#wasDerivedFrom"
        )
        if prov_pred is None:
            return pl.DataFrame({"src": [], "n": []})
        
        df = self.fact_store.scan_metadata_facts()
        df = df.filter(pl.col("p") == prov_pred)
        
        # Group by source (object) and count
        grouped = df.group_by("o").agg(pl.len().alias("n"))
        grouped = grouped.sort("n", descending=True)
        
        # Decode source URIs
        result = []
        for row in grouped.iter_rows(named=True):
            src = self.term_dict.get_lex(row["o"])
            result.append({"src": src, "n": row["n"]})
        
        return pl.DataFrame(result) if result else pl.DataFrame({"src": [], "n": []})
    
    def q12_count_by_run(self) -> pl.DataFrame:
        """
        Q12: Count statements per run.
        
        SELECT ?run (COUNT(?qt) AS ?n) WHERE {
            ?qt prov:wasGeneratedBy ?run .
        } GROUP BY ?run ORDER BY DESC(?n)
        """
        gen_pred = self.term_dict.lookup_iri(
            "http://www.w3.org/ns/prov#wasGeneratedBy"
        )
        if gen_pred is None:
            return pl.DataFrame({"run": [], "n": []})
        
        df = self.fact_store.scan_metadata_facts()
        df = df.filter(pl.col("p") == gen_pred)
        
        grouped = df.group_by("o").agg(pl.len().alias("n"))
        grouped = grouped.sort("n", descending=True)
        
        result = []
        for row in grouped.iter_rows(named=True):
            run = self.term_dict.get_lex(row["o"])
            result.append({"run": run, "n": row["n"]})
        
        return pl.DataFrame(result) if result else pl.DataFrame({"run": [], "n": []})
    
    def _qt_to_string(self, qt_id: TermId) -> str:
        """Convert a quoted triple ID to << s p o >> string form."""
        qt = self.qt_dict.lookup(qt_id)
        if qt is None:
            return f"<unknown qt:{qt_id}>"
        
        s = self.term_dict.get_lex(qt.s) or f"<{qt.s}>"
        p = self.term_dict.get_lex(qt.p) or f"<{qt.p}>"
        o = self.term_dict.get_lex(qt.o) or f"<{qt.o}>"
        
        return f"<< {s} {p} {o} >>"
