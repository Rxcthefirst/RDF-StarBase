"""
Columnar ETL Engine - High-performance tabular → RDF transformation

This engine uses Polars vectorized operations for maximum throughput,
processing entire columns at once rather than row-by-row iteration.
Aligns with RDF-StarBase's columnar storage architecture.

Supports:
- YARRRML mapping format (native)
- YARRRML-STAR extension for RDF-Star quoted triples
- CSV, Excel, JSON data sources
- Turtle, N-Triples, JSON-LD output formats
"""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Optional, Union

import polars as pl
import yaml


# =============================================================================
# Configuration Types
# =============================================================================

class OutputFormat(str, Enum):
    """Supported RDF serialization formats."""
    TURTLE = "ttl"
    NTRIPLES = "nt"
    JSONLD = "jsonld"
    TRIG = "trig"


@dataclass
class ColumnMapping:
    """Single column-to-predicate mapping."""
    source_column: str
    predicate: str
    datatype: Optional[str] = None
    language: Optional[str] = None
    is_iri: bool = False
    template: Optional[str] = None  # IRI template for object


@dataclass
class SubjectTemplate:
    """Template for generating subject IRIs."""
    template: str  # e.g., "http://example.org/person/$(id)"
    columns: list[str] = field(default_factory=list)  # extracted column refs


@dataclass 
class TriplesMap:
    """A complete triples map (one logical source → one subject pattern)."""
    name: str
    source: str  # filename or reference
    subject: SubjectTemplate
    predicate_objects: list[ColumnMapping]
    graph: Optional[str] = None
    # YARRRML-STAR: quoted triple annotations
    annotations: list[dict] = field(default_factory=list)


@dataclass
class YARRRMLMapping:
    """Parsed YARRRML document."""
    prefixes: dict[str, str]
    mappings: list[TriplesMap]
    base: Optional[str] = None


# =============================================================================
# YARRRML Parser
# =============================================================================

class YARRRMLParser:
    """
    Parse YARRRML documents into structured mapping configurations.
    
    Supports:
    - Standard YARRRML syntax
    - YARRRML-STAR extensions for quoted triple annotations
    - Shorthand notations (array predicateObjectMaps, etc.)
    """
    
    # Pattern to extract column references: $(column_name)
    COLUMN_REF_PATTERN = re.compile(r'\$\(([^)]+)\)')
    
    def parse(self, yaml_content: str) -> YARRRMLMapping:
        """Parse YARRRML YAML string into mapping structure."""
        doc = yaml.safe_load(yaml_content)
        
        prefixes = doc.get('prefixes', {})
        base = doc.get('base', None)
        mappings = []
        
        for map_name, map_def in doc.get('mappings', {}).items():
            triples_map = self._parse_triples_map(map_name, map_def, prefixes)
            mappings.append(triples_map)
        
        return YARRRMLMapping(prefixes=prefixes, mappings=mappings, base=base)
    
    def parse_file(self, path: Path) -> YARRRMLMapping:
        """Parse YARRRML file."""
        return self.parse(path.read_text(encoding='utf-8'))
    
    def _parse_triples_map(
        self, 
        name: str, 
        map_def: dict, 
        prefixes: dict[str, str]
    ) -> TriplesMap:
        """Parse a single triples map definition."""
        # Source(s)
        sources = map_def.get('sources', map_def.get('source', []))
        if isinstance(sources, str):
            sources = [sources]
        source = sources[0] if sources else ''
        # Handle source format: [file.csv~csv] or just file.csv
        if isinstance(source, list):
            source = source[0]
        source = source.split('~')[0] if '~' in str(source) else str(source)
        
        # Subject
        subject_def = map_def.get('s', map_def.get('subject', ''))
        subject = self._parse_subject(subject_def, prefixes)
        
        # Predicate-object maps
        po_maps = map_def.get('po', map_def.get('predicateObjectMaps', []))
        predicate_objects = self._parse_predicate_objects(po_maps, prefixes)
        
        # Graph (optional)
        graph = map_def.get('g', map_def.get('graph', None))
        
        # YARRRML-STAR annotations
        annotations = map_def.get('annotations', [])
        
        return TriplesMap(
            name=name,
            source=source,
            subject=subject,
            predicate_objects=predicate_objects,
            graph=graph,
            annotations=annotations,
        )
    
    def _parse_subject(
        self, 
        subject_def: Union[str, dict], 
        prefixes: dict[str, str]
    ) -> SubjectTemplate:
        """Parse subject template."""
        if isinstance(subject_def, str):
            template = self._expand_prefix(subject_def, prefixes)
        elif isinstance(subject_def, dict):
            template = subject_def.get('value', subject_def.get('template', ''))
            template = self._expand_prefix(template, prefixes)
        else:
            template = str(subject_def)
        
        columns = self.COLUMN_REF_PATTERN.findall(template)
        return SubjectTemplate(template=template, columns=columns)
    
    def _parse_predicate_objects(
        self, 
        po_maps: list, 
        prefixes: dict[str, str]
    ) -> list[ColumnMapping]:
        """Parse predicate-object map list."""
        mappings = []
        
        for po in po_maps:
            if isinstance(po, list) and len(po) >= 2:
                # Shorthand: [predicate, object]
                predicate = self._expand_prefix(str(po[0]), prefixes)
                obj = po[1]
                
                mapping = self._create_column_mapping(predicate, obj, prefixes)
                if mapping:
                    mappings.append(mapping)
            elif isinstance(po, dict):
                # Full form with p/o keys
                predicate = po.get('p', po.get('predicate', ''))
                predicate = self._expand_prefix(str(predicate), prefixes)
                obj = po.get('o', po.get('object', ''))
                
                mapping = self._create_column_mapping(predicate, obj, prefixes)
                if mapping:
                    mappings.append(mapping)
        
        return mappings
    
    def _create_column_mapping(
        self, 
        predicate: str, 
        obj: Any, 
        prefixes: dict[str, str]
    ) -> Optional[ColumnMapping]:
        """Create a ColumnMapping from predicate and object definition."""
        if isinstance(obj, str):
            # Simple column reference: $(column)
            cols = self.COLUMN_REF_PATTERN.findall(obj)
            if cols:
                return ColumnMapping(
                    source_column=cols[0],
                    predicate=predicate,
                    is_iri='://' in obj or obj.startswith('ex:'),
                    template=obj if '://' in obj else None,
                )
            else:
                # Literal value
                return None  # Skip static values for now
        elif isinstance(obj, dict):
            value = obj.get('value', obj.get('template', ''))
            cols = self.COLUMN_REF_PATTERN.findall(str(value))
            if cols:
                return ColumnMapping(
                    source_column=cols[0],
                    predicate=predicate,
                    datatype=obj.get('datatype'),
                    language=obj.get('language'),
                    is_iri=obj.get('type') == 'iri',
                    template=str(value) if '://' in str(value) else None,
                )
        return None
    
    def _expand_prefix(self, value: str, prefixes: dict[str, str]) -> str:
        """Expand prefixed URIs to full URIs."""
        if ':' in value and not value.startswith('http'):
            prefix, local = value.split(':', 1)
            if prefix in prefixes:
                return prefixes[prefix] + local
        return value


# =============================================================================
# Columnar RDF Generator
# =============================================================================

class ColumnarRDFGenerator:
    """
    High-performance RDF generator using Polars vectorized operations.
    
    Key optimizations:
    - Process entire columns at once, not row-by-row
    - Use Polars expressions for string templating
    - Batch output writing
    - Lazy evaluation where possible
    """
    
    XSD_MAPPINGS = {
        pl.Int8: 'http://www.w3.org/2001/XMLSchema#integer',
        pl.Int16: 'http://www.w3.org/2001/XMLSchema#integer',
        pl.Int32: 'http://www.w3.org/2001/XMLSchema#integer',
        pl.Int64: 'http://www.w3.org/2001/XMLSchema#integer',
        pl.UInt8: 'http://www.w3.org/2001/XMLSchema#nonNegativeInteger',
        pl.UInt16: 'http://www.w3.org/2001/XMLSchema#nonNegativeInteger',
        pl.UInt32: 'http://www.w3.org/2001/XMLSchema#nonNegativeInteger',
        pl.UInt64: 'http://www.w3.org/2001/XMLSchema#nonNegativeInteger',
        pl.Float32: 'http://www.w3.org/2001/XMLSchema#double',
        pl.Float64: 'http://www.w3.org/2001/XMLSchema#double',
        pl.Boolean: 'http://www.w3.org/2001/XMLSchema#boolean',
        pl.Date: 'http://www.w3.org/2001/XMLSchema#date',
        pl.Datetime: 'http://www.w3.org/2001/XMLSchema#dateTime',
        pl.Time: 'http://www.w3.org/2001/XMLSchema#time',
    }
    
    def __init__(self):
        self.triple_count = 0
        self.warnings: list[str] = []
    
    def generate(
        self,
        df: pl.DataFrame,
        mapping: YARRRMLMapping,
        output_format: OutputFormat = OutputFormat.TURTLE,
    ) -> str:
        """
        Generate RDF from DataFrame using YARRRML mapping.
        
        Returns serialized RDF string.
        """
        self.triple_count = 0
        self.warnings = []
        
        if output_format == OutputFormat.NTRIPLES:
            return self._generate_ntriples(df, mapping)
        elif output_format == OutputFormat.TURTLE:
            return self._generate_turtle(df, mapping)
        elif output_format == OutputFormat.JSONLD:
            return self._generate_jsonld(df, mapping)
        else:
            # Default to N-Triples (simplest)
            return self._generate_ntriples(df, mapping)
    
    def _generate_ntriples(
        self, 
        df: pl.DataFrame, 
        mapping: YARRRMLMapping
    ) -> str:
        """Generate N-Triples format using columnar operations."""
        lines: list[str] = []
        
        for triples_map in mapping.mappings:
            # Generate subject column
            subject_col = self._build_subject_column(df, triples_map.subject)
            
            # For each predicate-object mapping, generate triples
            for po in triples_map.predicate_objects:
                if po.source_column not in df.columns:
                    self.warnings.append(
                        f"Column '{po.source_column}' not found in data"
                    )
                    continue
                
                # Build object column with proper RDF formatting
                object_col = self._build_object_column(df, po)
                
                # Vectorized triple generation
                predicate = f"<{po.predicate}>"
                
                # Create triples using Polars expressions
                triples_df = df.select([
                    subject_col.alias('s'),
                    pl.lit(predicate).alias('p'),
                    object_col.alias('o'),
                ]).filter(
                    pl.col('o').is_not_null() & (pl.col('o') != '')
                )
                
                # Format as N-Triples lines
                triple_lines = triples_df.select(
                    pl.concat_str([
                        pl.col('s'),
                        pl.lit(' '),
                        pl.col('p'),
                        pl.lit(' '),
                        pl.col('o'),
                        pl.lit(' .'),
                    ])
                ).to_series().to_list()
                
                lines.extend(triple_lines)
                self.triple_count += len(triple_lines)
        
        return '\n'.join(lines)
    
    def _generate_turtle(
        self, 
        df: pl.DataFrame, 
        mapping: YARRRMLMapping
    ) -> str:
        """Generate Turtle format with prefix declarations."""
        output = StringIO()
        
        # Write prefixes
        for prefix, uri in mapping.prefixes.items():
            output.write(f"@prefix {prefix}: <{uri}> .\n")
        output.write('\n')
        
        # Group triples by subject for prettier output
        for triples_map in mapping.mappings:
            subject_col = self._build_subject_column(df, triples_map.subject)
            
            # Build all predicate-objects for this map
            po_data = []
            for po in triples_map.predicate_objects:
                if po.source_column not in df.columns:
                    continue
                object_col = self._build_object_column(df, po)
                po_data.append((po.predicate, object_col))
            
            if not po_data:
                continue
            
            # Process each row (convert to dict for grouping)
            result_df = df.with_columns([
                subject_col.alias('__subject__'),
                *[col.alias(f'__obj_{i}__') for i, (_, col) in enumerate(po_data)]
            ])
            
            # Group by subject
            for row in result_df.iter_rows(named=True):
                subject = row['__subject__']
                if not subject:
                    continue
                
                output.write(f"{subject}\n")
                
                po_lines = []
                for i, (predicate, _) in enumerate(po_data):
                    obj = row[f'__obj_{i}__']
                    if obj:
                        # Compact predicate using prefixes
                        compact_pred = self._compact_uri(predicate, mapping.prefixes)
                        po_lines.append(f"    {compact_pred} {obj}")
                        self.triple_count += 1
                
                if po_lines:
                    output.write(' ;\n'.join(po_lines))
                    output.write(' .\n\n')
        
        return output.getvalue()
    
    def _generate_jsonld(
        self, 
        df: pl.DataFrame, 
        mapping: YARRRMLMapping
    ) -> str:
        """Generate JSON-LD format."""
        import json
        
        context = {"@vocab": mapping.base or "http://example.org/"}
        context.update(mapping.prefixes)
        
        graph = []
        
        for triples_map in mapping.mappings:
            subject_col = self._build_subject_column(
                df, triples_map.subject, for_jsonld=True
            )
            
            po_data = []
            for po in triples_map.predicate_objects:
                if po.source_column not in df.columns:
                    continue
                po_data.append(po)
            
            if not po_data:
                continue
            
            result_df = df.with_columns([subject_col.alias('__subject__')])
            
            for row in result_df.iter_rows(named=True):
                subject = row['__subject__']
                if not subject:
                    continue
                
                node = {"@id": subject}
                
                for po in po_data:
                    value = row.get(po.source_column)
                    if value is not None and str(value).strip():
                        pred_key = self._compact_uri(po.predicate, mapping.prefixes)
                        if po.is_iri:
                            node[pred_key] = {"@id": str(value)}
                        elif po.datatype:
                            node[pred_key] = {
                                "@value": str(value),
                                "@type": po.datatype
                            }
                        elif po.language:
                            node[pred_key] = {
                                "@value": str(value),
                                "@language": po.language
                            }
                        else:
                            node[pred_key] = value
                        self.triple_count += 1
                
                if len(node) > 1:  # Has more than just @id
                    graph.append(node)
        
        return json.dumps({
            "@context": context,
            "@graph": graph
        }, indent=2)
    
    def _build_subject_column(
        self, 
        df: pl.DataFrame, 
        subject: SubjectTemplate,
        for_jsonld: bool = False,
    ) -> pl.Expr:
        """
        Build subject IRI column using vectorized string operations.
        
        Handles templates like: http://example.org/person/$(id)
        """
        template = subject.template
        
        if not subject.columns:
            # Static subject (rare)
            if for_jsonld:
                return pl.lit(template)
            return pl.lit(f"<{template}>")
        
        # Parse template into parts: literals and column references
        # e.g., "http://example.org/person/$(id)" -> ["http://example.org/person/", $(id)]
        parts = self._parse_template_parts(template, df.columns)
        
        # Build concatenation expression
        if len(parts) == 1 and isinstance(parts[0], pl.Expr):
            expr = parts[0]
        else:
            expr = pl.concat_str(parts)
        
        if for_jsonld:
            return expr
        
        # Wrap in angle brackets for N-Triples/Turtle
        return pl.concat_str([pl.lit('<'), expr, pl.lit('>')])
    
    def _parse_template_parts(
        self, 
        template: str, 
        available_columns: list[str]
    ) -> list[Union[pl.Expr, str]]:
        """
        Parse a template string into a list of literal strings and column expressions.
        
        Example: "http://example.org/$(id)/$(name)" 
        Returns: [lit("http://example.org/"), col("id"), lit("/"), col("name")]
        """
        import re
        parts = []
        last_end = 0
        
        for match in re.finditer(r'\$\(([^)]+)\)', template):
            # Add literal before this match
            if match.start() > last_end:
                parts.append(pl.lit(template[last_end:match.start()]))
            
            # Add column reference
            col_name = match.group(1)
            if col_name in available_columns:
                parts.append(pl.col(col_name).cast(pl.Utf8).fill_null(''))
            else:
                # Column not found, keep as literal placeholder
                parts.append(pl.lit(match.group(0)))
            
            last_end = match.end()
        
        # Add remaining literal
        if last_end < len(template):
            parts.append(pl.lit(template[last_end:]))
        
        return parts if parts else [pl.lit(template)]
    def _build_object_column(
        self, 
        df: pl.DataFrame, 
        mapping: ColumnMapping
    ) -> pl.Expr:
        """
        Build object column with proper RDF literal/IRI formatting.
        
        Uses vectorized operations for datatype detection and formatting.
        """
        col = pl.col(mapping.source_column)
        dtype = df.schema.get(mapping.source_column)
        
        # Handle IRI objects
        if mapping.is_iri:
            if mapping.template:
                # Template-based IRI - use parse_template_parts
                parts = self._parse_template_parts(mapping.template, df.columns)
                if len(parts) == 1:
                    expr = parts[0]
                else:
                    expr = pl.concat_str(parts)
                return pl.concat_str([pl.lit('<'), expr, pl.lit('>')])
            else:
                return pl.concat_str([pl.lit('<'), col.cast(pl.Utf8), pl.lit('>')])
        
        # Handle literals
        if mapping.language:
            # Language-tagged string
            return pl.concat_str([
                pl.lit('"'),
                self._escape_string_expr(col),
                pl.lit('"@'),
                pl.lit(mapping.language),
            ])
        
        if mapping.datatype:
            # Explicit datatype
            return pl.concat_str([
                pl.lit('"'),
                self._escape_string_expr(col),
                pl.lit('"^^<'),
                pl.lit(mapping.datatype),
                pl.lit('>'),
            ])
        
        # Auto-detect datatype from Polars type
        xsd_type = self.XSD_MAPPINGS.get(dtype)
        
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            # Integer - can be unquoted in Turtle, but use typed literal for safety
            return pl.concat_str([
                pl.lit('"'),
                col.cast(pl.Utf8),
                pl.lit('"^^<http://www.w3.org/2001/XMLSchema#integer>'),
            ])
        
        if dtype in (pl.Float32, pl.Float64):
            return pl.concat_str([
                pl.lit('"'),
                col.cast(pl.Utf8),
                pl.lit('"^^<http://www.w3.org/2001/XMLSchema#double>'),
            ])
        
        if dtype == pl.Boolean:
            return pl.concat_str([
                pl.lit('"'),
                col.cast(pl.Utf8).str.to_lowercase(),
                pl.lit('"^^<http://www.w3.org/2001/XMLSchema#boolean>'),
            ])
        
        if dtype == pl.Date:
            return pl.concat_str([
                pl.lit('"'),
                col.cast(pl.Utf8),
                pl.lit('"^^<http://www.w3.org/2001/XMLSchema#date>'),
            ])
        
        if dtype == pl.Datetime:
            return pl.concat_str([
                pl.lit('"'),
                col.cast(pl.Utf8),
                pl.lit('"^^<http://www.w3.org/2001/XMLSchema#dateTime>'),
            ])
        
        # Default: plain string literal
        return pl.concat_str([
            pl.lit('"'),
            self._escape_string_expr(col),
            pl.lit('"'),
        ])
    
    def _escape_string_expr(self, col: pl.Expr) -> pl.Expr:
        """Escape special characters in string literals (vectorized)."""
        return (
            col.cast(pl.Utf8)
            .fill_null('')
            .str.replace_all('\\', '\\\\', literal=True)
            .str.replace_all('"', '\\"', literal=True)
            .str.replace_all('\n', '\\n', literal=True)
            .str.replace_all('\r', '\\r', literal=True)
            .str.replace_all('\t', '\\t', literal=True)
        )
    
    def _compact_uri(self, uri: str, prefixes: dict[str, str]) -> str:
        """Compact a full URI using prefixes."""
        for prefix, namespace in prefixes.items():
            if uri.startswith(namespace):
                return f"{prefix}:{uri[len(namespace):]}"
        return f"<{uri}>"


# =============================================================================
# Main ETL Engine
# =============================================================================

class ETLEngine:
    """
    High-performance columnar ETL engine for tabular → RDF transformation.
    
    Usage:
        engine = ETLEngine()
        
        # From YARRRML file
        rdf = engine.transform_file(
            data_file='data.csv',
            mapping_file='mapping.yarrrml.yml',
            output_format='ttl'
        )
        
        # From mapping dict (e.g., from Starchart UI)
        rdf = engine.transform(
            data_file='data.csv',
            mapping_config={
                'prefixes': {...},
                'mappings': {...}
            }
        )
    """
    
    def __init__(self):
        self.parser = YARRRMLParser()
        self.generator = ColumnarRDFGenerator()
    
    def transform(
        self,
        data_file: Union[str, Path],
        mapping_config: Union[dict, str, Path],
        output_format: OutputFormat = OutputFormat.TURTLE,
        limit: Optional[int] = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Transform tabular data to RDF.
        
        Args:
            data_file: Path to CSV, Excel, or JSON file
            mapping_config: YARRRML dict, YAML string, or path to file
            output_format: Output serialization format
            limit: Optional row limit for testing
            
        Returns:
            Tuple of (rdf_content, report_dict)
        """
        # Load data
        df = self._load_data(Path(data_file), limit)
        
        # Parse mapping
        mapping = self._parse_mapping(mapping_config)
        
        # Generate RDF
        rdf_content = self.generator.generate(df, mapping, output_format)
        
        report = {
            'triple_count': self.generator.triple_count,
            'row_count': len(df),
            'warnings': self.generator.warnings,
        }
        
        return rdf_content, report
    
    def transform_file(
        self,
        data_file: Union[str, Path],
        mapping_file: Union[str, Path],
        output_format: OutputFormat = OutputFormat.TURTLE,
        limit: Optional[int] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Transform using a mapping file."""
        return self.transform(
            data_file=data_file,
            mapping_config=Path(mapping_file),
            output_format=output_format,
            limit=limit,
        )
    
    def _load_data(self, path: Path, limit: Optional[int] = None) -> pl.DataFrame:
        """Load data file into Polars DataFrame."""
        suffix = path.suffix.lower()
        
        if suffix == '.csv':
            df = pl.read_csv(path, infer_schema_length=10000)
        elif suffix in ('.xlsx', '.xls'):
            df = pl.read_excel(path)
        elif suffix == '.json':
            df = pl.read_json(path)
        elif suffix == '.parquet':
            df = pl.read_parquet(path)
        else:
            # Try CSV as fallback
            df = pl.read_csv(path, infer_schema_length=10000)
        
        if limit:
            df = df.head(limit)
        
        return df
    
    def _parse_mapping(
        self, 
        config: Union[dict, str, Path]
    ) -> YARRRMLMapping:
        """Parse mapping from various input formats."""
        if isinstance(config, Path):
            return self.parser.parse_file(config)
        elif isinstance(config, str):
            # Could be YAML string or file path
            if config.strip().startswith(('prefixes:', 'mappings:')):
                return self.parser.parse(config)
            else:
                return self.parser.parse_file(Path(config))
        elif isinstance(config, dict):
            # Convert dict to YARRRML format
            return self._dict_to_mapping(config)
        else:
            raise ValueError(f"Unsupported mapping config type: {type(config)}")
    
    def _dict_to_mapping(self, config: dict) -> YARRRMLMapping:
        """Convert a simple dict mapping to YARRRMLMapping."""
        prefixes = config.get('prefixes', {
            'ex': 'http://example.org/',
            'schema': 'http://schema.org/',
            'foaf': 'http://xmlns.com/foaf/0.1/',
            'xsd': 'http://www.w3.org/2001/XMLSchema#',
        })
        
        # Handle Starchart-style mappings
        if 'mappings' in config and isinstance(config['mappings'], list):
            # List of {source_column, predicate} dicts
            po_maps = []
            for m in config['mappings']:
                po_maps.append(ColumnMapping(
                    source_column=m['source_column'],
                    predicate=m['predicate'],
                    datatype=m.get('datatype'),
                    language=m.get('language'),
                    is_iri=m.get('is_iri', False),
                ))
            
            # Build subject template
            subject_template = config.get('subject_template', 'http://example.org/resource/$(id)')
            subject = SubjectTemplate(
                template=subject_template,
                columns=YARRRMLParser.COLUMN_REF_PATTERN.findall(subject_template)
            )
            
            source = config.get('sources', [{}])[0].get('file', 'data.csv')
            
            triples_map = TriplesMap(
                name='starchart_mapping',
                source=source,
                subject=subject,
                predicate_objects=po_maps,
            )
            
            return YARRRMLMapping(
                prefixes=prefixes,
                mappings=[triples_map],
            )
        
        # Assume it's already YARRRML-style dict
        yaml_str = yaml.dump(config)
        return self.parser.parse(yaml_str)


# =============================================================================
# Factory function
# =============================================================================

def create_etl_engine() -> ETLEngine:
    """Create and return an ETL engine instance."""
    return ETLEngine()
