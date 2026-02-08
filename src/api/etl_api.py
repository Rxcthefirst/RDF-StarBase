"""
ETL API Endpoints - REST API for columnar RDF transformation

Provides endpoints for:
- Converting tabular data to RDF using YARRRML mappings
- Importing/exporting YARRRML mappings
- Direct loading into repositories
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rdf_starbase.etl_engine import (
    ETLEngine,
    OutputFormat,
    YARRRMLParser,
    create_etl_engine,
)

router = APIRouter(prefix="/etl", tags=["ETL"])

# Always available - no external dependencies
ETL_AVAILABLE = True


# ============================================================================
# Request/Response Models
# ============================================================================

class ConvertRequest(BaseModel):
    """Request to convert data to RDF."""
    mapping: dict[str, Any] = Field(..., description="Mapping configuration")
    output_format: str = Field("ttl", description="Output format: ttl, nt, jsonld")
    limit: Optional[int] = Field(None, description="Limit rows to process (for testing)")
    load_to_repository: Optional[str] = Field(None, description="Repository ID to load results into")


class ConvertResponse(BaseModel):
    """Response from conversion."""
    rdf_content: str = Field(..., description="Generated RDF content")
    triple_count: int = Field(..., description="Number of triples generated")
    row_count: int = Field(0, description="Number of data rows processed")
    format: str = Field(..., description="Output format used")
    warnings: list[str] = Field(default_factory=list, description="Warnings during conversion")
    loaded: bool = Field(False, description="Whether data was loaded to repository")


class YARRRMLExportRequest(BaseModel):
    """Request to export mapping as YARRRML."""
    mapping: dict[str, Any] = Field(..., description="Mapping configuration to export")


class YARRRMLExportResponse(BaseModel):
    """Response with YARRRML content."""
    yarrrml: str = Field(..., description="YARRRML content")
    format: str = Field("yarrrml", description="Format identifier")


class YARRRMLImportResponse(BaseModel):
    """Response from YARRRML import."""
    mapping: dict[str, Any] = Field(..., description="Imported mapping configuration")
    source_format: str = Field("yarrrml", description="Original format")


class ETLStatusResponse(BaseModel):
    """ETL service status."""
    available: bool = Field(..., description="Whether ETL service is available")
    engine: str = Field("columnar", description="Engine type")
    features: list[str] = Field(default_factory=list, description="Available features")


class StarchartMapping(BaseModel):
    """Mapping from Starchart UI."""
    columns: dict[str, Optional[str]] = Field(..., description="Column to property mapping")
    subject_template: str = Field("http://example.org/resource/$(id)", description="Subject IRI template")
    prefixes: dict[str, str] = Field(default_factory=dict, description="Namespace prefixes")
    source_file: str = Field("data.csv", description="Source filename")


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/status", response_model=ETLStatusResponse)
async def get_etl_status():
    """Check if ETL service is available and get capabilities."""
    return ETLStatusResponse(
        available=True,
        engine="columnar",
        features=[
            "csv_transform",
            "excel_transform", 
            "json_transform",
            "yarrrml_import",
            "yarrrml_export",
            "turtle_output",
            "ntriples_output",
            "jsonld_output",
            "columnar_processing",
        ]
    )


@router.post("/convert", response_model=ConvertResponse)
async def convert_data(
    data_file: UploadFile = File(..., description="Data file to convert"),
    mapping: str = Form(..., description="Mapping configuration as JSON string"),
    output_format: str = Form("ttl"),
    limit: Optional[int] = Form(None),
    load_to_repository: Optional[str] = Form(None),
):
    """
    Convert tabular data to RDF using columnar processing.
    
    Supports CSV, Excel, JSON input formats.
    Output can be Turtle, N-Triples, or JSON-LD.
    """
    try:
        mapping_config = json.loads(mapping)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid mapping JSON: {str(e)}")
    
    try:
        engine = create_etl_engine()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save data file
            data_path = Path(tmpdir) / data_file.filename
            with open(data_path, "wb") as f:
                f.write(await data_file.read())
            
            # Map output format string to enum
            fmt_map = {
                'ttl': OutputFormat.TURTLE,
                'turtle': OutputFormat.TURTLE,
                'nt': OutputFormat.NTRIPLES,
                'ntriples': OutputFormat.NTRIPLES,
                'jsonld': OutputFormat.JSONLD,
                'json-ld': OutputFormat.JSONLD,
            }
            out_fmt = fmt_map.get(output_format.lower(), OutputFormat.TURTLE)
            
            # Run transformation
            rdf_content, report = engine.transform(
                data_file=data_path,
                mapping_config=mapping_config,
                output_format=out_fmt,
                limit=limit,
            )
            
            # Optionally load to repository
            loaded = False
            warnings = report.get('warnings', [])
            
            if load_to_repository:
                try:
                    from rdf_starbase.repositories import get_repository
                    repo = get_repository(load_to_repository)
                    if repo:
                        repo.load_rdf(rdf_content, format=output_format)
                        loaded = True
                except Exception as e:
                    warnings.append(f"Failed to load to repository: {str(e)}")
            
            return ConvertResponse(
                rdf_content=rdf_content,
                triple_count=report.get('triple_count', 0),
                row_count=report.get('row_count', 0),
                format=output_format,
                warnings=warnings,
                loaded=loaded,
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")


@router.post("/yarrrml/export", response_model=YARRRMLExportResponse)
async def export_yarrrml(request: YARRRMLExportRequest):
    """
    Export a mapping configuration as YARRRML.
    
    Converts internal mapping format to standard YARRRML syntax.
    """
    try:
        mapping = request.mapping
        
        # Build YARRRML structure
        yarrrml = {
            'prefixes': mapping.get('prefixes', {
                'ex': 'http://example.org/',
                'schema': 'http://schema.org/',
                'foaf': 'http://xmlns.com/foaf/0.1/',
                'xsd': 'http://www.w3.org/2001/XMLSchema#',
            }),
            'mappings': {}
        }
        
        # Handle Starchart-style mappings
        if 'mappings' in mapping and isinstance(mapping['mappings'], list):
            source_file = mapping.get('sources', [{}])[0].get('file', 'data.csv')
            subject_template = mapping.get('subject_template', 'ex:resource/$(id)')
            
            po_list = []
            for m in mapping['mappings']:
                col = m.get('source_column')
                pred = m.get('predicate')
                if col and pred:
                    po_list.append([pred, f"$({col})"])
            
            yarrrml['mappings']['main'] = {
                'sources': [[f"{source_file}~csv"]],
                's': subject_template,
                'po': po_list,
            }
        else:
            # Pass through as-is if already YARRRML-like
            yarrrml['mappings'] = mapping.get('mappings', {})
        
        yarrrml_str = yaml.dump(yarrrml, default_flow_style=False, sort_keys=False)
        
        return YARRRMLExportResponse(
            yarrrml=yarrrml_str,
            format="yarrrml"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YARRRML export failed: {str(e)}")


@router.post("/yarrrml/import", response_model=YARRRMLImportResponse)
async def import_yarrrml(
    yarrrml_file: UploadFile = File(..., description="YARRRML file to import")
):
    """
    Import a YARRRML file and convert to internal mapping format.
    """
    try:
        content = await yarrrml_file.read()
        yarrrml_str = content.decode('utf-8')
        
        # Parse YARRRML
        parser = YARRRMLParser()
        parsed = parser.parse(yarrrml_str)
        
        # Convert to internal format
        mapping = {
            'prefixes': parsed.prefixes,
            'mappings': [],
            'sources': [],
        }
        
        for triples_map in parsed.mappings:
            if triples_map.source:
                mapping['sources'].append({'file': triples_map.source, 'type': 'csv'})
            
            mapping['subject_template'] = triples_map.subject.template
            
            for po in triples_map.predicate_objects:
                mapping['mappings'].append({
                    'source_column': po.source_column,
                    'predicate': po.predicate,
                    'datatype': po.datatype,
                    'language': po.language,
                    'is_iri': po.is_iri,
                })
        
        return YARRRMLImportResponse(
            mapping=mapping,
            source_format="yarrrml"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YARRRML import failed: {str(e)}")


@router.post("/mapping/from-starchart")
async def convert_starchart_mapping(request: StarchartMapping):
    """
    Convert Starchart UI mapping format to internal mapping configuration.
    
    This takes the column→property mappings from the visual editor
    and produces a configuration ready for the ETL engine.
    """
    try:
        # Filter out unmapped columns
        mapped = {col: prop for col, prop in request.columns.items() if prop}
        
        # Build internal mapping format
        mapping_config = {
            'prefixes': request.prefixes or {
                'ex': 'http://example.org/',
                'schema': 'http://schema.org/',
                'foaf': 'http://xmlns.com/foaf/0.1/',
                'xsd': 'http://www.w3.org/2001/XMLSchema#',
            },
            'sources': [{'file': request.source_file, 'type': 'csv'}],
            'subject_template': request.subject_template,
            'mappings': [
                {'source_column': col, 'predicate': prop}
                for col, prop in mapped.items()
            ],
        }
        
        return {
            'mapping': mapping_config,
            'column_count': len(mapped),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mapping conversion failed: {str(e)}")
