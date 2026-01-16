"""
SPARQL-Star Query Executor using Polars.

Translates SPARQL-Star AST to Polars operations for blazingly fast execution.
"""

from typing import Any, Optional, Union
from datetime import datetime

import polars as pl

from rdf_starbase.sparql.ast import (
    Query, SelectQuery, AskQuery,
    TriplePattern, QuotedTriplePattern,
    Variable, IRI, Literal, BlankNode,
    Filter, Comparison, LogicalExpression, FunctionCall,
    ComparisonOp, LogicalOp,
    WhereClause, ProvenanceFilter,
    Term,
)
from rdf_starbase.store import TripleStore


class SPARQLExecutor:
    """
    Executes SPARQL-Star queries against a TripleStore.
    
    Translation strategy:
    - Each TriplePattern becomes a filtered view of the DataFrame
    - Variables become column selections
    - Joins are performed for patterns sharing variables
    - Filters become Polars filter expressions
    - Uses lazy evaluation for query optimization
    """
    
    def __init__(self, store: TripleStore):
        """
        Initialize executor with a triple store.
        
        Args:
            store: The TripleStore to query
        """
        self.store = store
        self._var_counter = 0
    
    def execute(self, query: Query) -> Union[pl.DataFrame, bool]:
        """
        Execute a SPARQL-Star query.
        
        Args:
            query: Parsed Query AST
            
        Returns:
            DataFrame for SELECT queries, bool for ASK queries
        """
        if isinstance(query, SelectQuery):
            return self._execute_select(query)
        elif isinstance(query, AskQuery):
            return self._execute_ask(query)
        else:
            raise NotImplementedError(f"Query type {type(query)} not yet supported")
    
    def _execute_select(self, query: SelectQuery) -> pl.DataFrame:
        """Execute a SELECT query."""
        # Start with lazy frame for optimization
        df = self._execute_where(query.where, query.prefixes)
        
        # Apply DISTINCT if requested
        if query.distinct:
            df = df.unique()
        
        # Apply ORDER BY
        if query.order_by:
            order_cols = []
            descending = []
            for var, asc in query.order_by:
                order_cols.append(var.name)
                descending.append(not asc)
            df = df.sort(order_cols, descending=descending)
        
        # Apply LIMIT and OFFSET
        if query.offset:
            df = df.slice(query.offset, query.limit or len(df))
        elif query.limit:
            df = df.head(query.limit)
        
        # Select only requested variables (or all if SELECT *)
        if not query.is_select_all():
            select_cols = [v.name for v in query.variables if v.name in df.columns]
            df = df.select(select_cols)
        
        return df
    
    def _execute_ask(self, query: AskQuery) -> bool:
        """Execute an ASK query."""
        df = self._execute_where(query.where, query.prefixes)
        return len(df) > 0
    
    def _execute_where(
        self,
        where: WhereClause,
        prefixes: dict[str, str]
    ) -> pl.DataFrame:
        """
        Execute a WHERE clause and return matching bindings.
        """
        if not where.patterns:
            return pl.DataFrame()
        
        # Execute each triple pattern and join results
        result_df: Optional[pl.DataFrame] = None
        
        for i, pattern in enumerate(where.patterns):
            pattern_df = self._execute_pattern(pattern, prefixes, i)
            
            if result_df is None:
                result_df = pattern_df
            else:
                # Find shared variables to join on
                shared_cols = set(result_df.columns) & set(pattern_df.columns)
                shared_cols -= {"_pattern_idx"}  # Don't join on internal columns
                
                if shared_cols:
                    result_df = result_df.join(
                        pattern_df,
                        on=list(shared_cols),
                        how="inner"
                    )
                else:
                    # Cross join if no shared variables
                    result_df = result_df.join(pattern_df, how="cross")
        
        if result_df is None:
            return pl.DataFrame()
        
        # Apply standard FILTER clauses
        for filter_clause in where.filters:
            if isinstance(filter_clause, Filter):
                result_df = self._apply_filter(result_df, filter_clause)
            elif isinstance(filter_clause, ProvenanceFilter):
                result_df = self._apply_provenance_filter(result_df, filter_clause)
        
        # Check if we have matches before removing internal columns
        has_matches = len(result_df) > 0
        
        # Remove internal columns
        internal_cols = [c for c in result_df.columns if c.startswith("_")]
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
        pattern_idx: int
    ) -> pl.DataFrame:
        """
        Execute a single triple pattern against the store.
        
        Returns a DataFrame with columns for each variable in the pattern.
        """
        # Start with all assertions
        df = self.store._df.lazy()
        
        # Apply filters for concrete terms
        if not isinstance(pattern.subject, Variable):
            value = self._resolve_term(pattern.subject, prefixes)
            df = df.filter(pl.col("subject") == value)
        
        if not isinstance(pattern.predicate, Variable):
            value = self._resolve_term(pattern.predicate, prefixes)
            df = df.filter(pl.col("predicate") == value)
        
        if not isinstance(pattern.object, (Variable, QuotedTriplePattern)):
            value = self._resolve_term(pattern.object, prefixes)
            df = df.filter(pl.col("object") == str(value))
        
        # Exclude deprecated by default
        df = df.filter(~pl.col("deprecated"))
        
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
        
        # Always include provenance columns for provenance filters
        provenance_cols = ["source", "confidence", "timestamp", "process"]
        for col in provenance_cols:
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
    
    def _resolve_term(self, term: Term, prefixes: dict[str, str]) -> str:
        """Resolve a term to its string value."""
        if isinstance(term, IRI):
            value = term.value
            # Expand prefixed names
            if ":" in value and not value.startswith("http"):
                prefix, local = value.split(":", 1)
                if prefix in prefixes:
                    return prefixes[prefix] + local
            return value
        elif isinstance(term, Literal):
            return str(term.value)
        elif isinstance(term, BlankNode):
            return f"_:{term.label}"
        else:
            return str(term)
    
    def _apply_filter(self, df: pl.DataFrame, filter_clause: Filter) -> pl.DataFrame:
        """Apply a standard FILTER to the DataFrame."""
        expr = self._build_filter_expression(filter_clause.expression)
        if expr is not None:
            return df.filter(expr)
        return df
    
    def _apply_provenance_filter(
        self,
        df: pl.DataFrame,
        filter_clause: ProvenanceFilter
    ) -> pl.DataFrame:
        """
        Apply a provenance FILTER to the DataFrame.
        
        RDF-StarBase extension for filtering by provenance metadata.
        """
        field = filter_clause.provenance_field
        
        # Find the provenance column (may be from any pattern)
        prov_cols = [c for c in df.columns if c.endswith(f"_{field}")]
        
        if not prov_cols:
            # No provenance column found - may need to add from store
            return df
        
        # Build filter expression with provenance column substitution
        expr = filter_clause.expression
        
        if isinstance(expr, Comparison):
            # Apply to all matching provenance columns (OR logic for multiple patterns)
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
        """Build a Polars comparison for provenance filtering.
        
        Substitutes the variable in the comparison with the actual provenance column.
        """
        # Determine which side has the variable (to be replaced with prov column)
        # and which side has the literal value
        if isinstance(expr.left, Variable):
            left = pl.col(prov_col)
            right = self._term_to_expr(expr.right)
        elif isinstance(expr.right, Variable):
            left = self._term_to_expr(expr.left)
            right = pl.col(prov_col)
        else:
            # Both sides are literals - unusual but handle it
            left = self._term_to_expr(expr.left)
            right = self._term_to_expr(expr.right)
        
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
        expr: Union[Comparison, LogicalExpression, FunctionCall]
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
                self._build_filter_expression(op) for op in expr.operands
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
        
        return None
    
    def _build_comparison_operands(
        self,
        left_term: Union[Variable, Literal, IRI, FunctionCall],
        right_term: Union[Variable, Literal, IRI, FunctionCall]
    ) -> tuple[Optional[pl.Expr], Optional[pl.Expr]]:
        """
        Build comparison operands with proper type coercion.
        
        When comparing a variable (column) with a typed literal, casts the 
        column to match the literal's type for proper numeric/string comparison.
        """
        left = self._term_to_expr(left_term)
        right = self._term_to_expr(right_term)
        
        if left is None or right is None:
            return left, right
        
        # Check if we need to cast a variable column to match a literal type
        if isinstance(left_term, Variable) and isinstance(right_term, Literal):
            if right_term.datatype:
                left = self._cast_column_for_comparison(left, right_term.datatype)
        elif isinstance(right_term, Variable) and isinstance(left_term, Literal):
            if left_term.datatype:
                right = self._cast_column_for_comparison(right, left_term.datatype)
        
        return left, right
    
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
        
        # Add more functions as needed
        
        return None


def execute_sparql(store: TripleStore, query_string: str) -> Union[pl.DataFrame, bool]:
    """
    Convenience function to parse and execute a SPARQL-Star query.
    
    Args:
        store: The TripleStore to query
        query_string: SPARQL-Star query string
        
    Returns:
        Query results (DataFrame for SELECT, bool for ASK)
    """
    from rdf_starbase.sparql.parser import parse_query
    
    query = parse_query(query_string)
    executor = SPARQLExecutor(store)
    return executor.execute(query)
