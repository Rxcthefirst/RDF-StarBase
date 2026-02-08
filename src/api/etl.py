"""
ETL API Endpoints - REST API for RDF transformation pipeline

Provides endpoints for:
- Generating mappings from data + ontology
- Converting data to RDF using mappings
- Importing/exporting YARRRML mappings
- SHACL validation
- Direct loading into repositories
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

try:
    from rdf_starbase.etl import ETLService, MappingFormat, OutputFormat, create_etl_service
    ETL_AVAILABLE = True
except ImportError:
    ETL_AVAILABLE = False


router = APIRouter(prefix="/etl", tags=["ETL"])


# ============================================================================
# Request/Response Models
# ============================================================================

class GenerateMappingRequest(BaseModel):
    """Request to auto-generate a mapping configuration."""
    target_class: Optional[str] = Field(None, description="Target ontology class (auto-detected if omitted)")
    base_iri: str = Field("http://example.org/", description="Base IRI for generated resources")
    use_ai: bool = Field(True, description="Use AI-powered semantic matching")


class GenerateMappingResponse(BaseModel):
    """Response containing generated mapping."""
    mapping: dict[str, Any] = Field(..., description="Generated mapping configuration")
    analysis: dict[str, Any] = Field(default_factory=dict, description="Column analysis results")
    confidence_scores: dict[str, float] = Field(default_factory=dict, description="Confidence scores for mappings")


class ConvertRequest(BaseModel):
    """Request to convert data to RDF."""
    mapping: dict[str, Any] = Field(..., description="Mapping configuration")
    output_format: str = Field("ttl", description="Output format: ttl, nt, jsonld, xml, trig")
    run_validation: bool = Field(False, description="Run SHACL validation after conversion")
    limit: Optional[int] = Field(None, description="Limit rows to process (for testing)")
    load_to_repository: Optional[str] = Field(None, description="Repository ID to load results into")


class ConvertResponse(BaseModel):
    """Response from conversion."""
    rdf_content: str = Field(..., description="Generated RDF content")
    triple_count: int = Field(..., description="Number of triples generated")
    format: str = Field(..., description="Output format used")
    warnings: list[str] = Field(default_factory=list, description="Warnings during conversion")
    validation: Optional[dict] = Field(None, description="SHACL validation results if requested")
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
    version: Optional[str] = Field(None, description="RDFMap version")
    features: list[str] = Field(default_factory=list, description="Available features")


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/status", response_model=ETLStatusResponse)
async def get_etl_status():
    """Check if ETL service is available and get capabilities."""
    if not ETL_AVAILABLE:
        return ETLStatusResponse(
            available=False,
            version=None,
            features=[]
        )
    
    try:
        import semantic_rdf_mapper
        version = getattr(semantic_rdf_mapper, "__version__", "0.3.0")
    except ImportError:
        version = "unknown"
    
    return ETLStatusResponse(
        available=True,
        version=version,
        features=[
            "generate_mapping",
            "convert",
            "yarrrml_import",
            "yarrrml_export",
            "shacl_validation",
            "ai_matching",
        ]
    )


@router.post("/generate", response_model=GenerateMappingResponse)
async def generate_mapping(
    data_file: UploadFile = File(..., description="Data file (CSV, Excel, JSON, XML)"),
    ontology_file: UploadFile = File(..., description="Ontology file (TTL, RDF/XML, etc.)"),
    target_class: Optional[str] = Form(None),
    base_iri: str = Form("http://example.org/"),
    use_ai: bool = Form(True),
):
    """
    Auto-generate a mapping configuration from data and ontology.
    
    Uses AI-powered semantic matching to intelligently align data columns
    with ontology properties based on names, types, and sample values.
    """
    if not ETL_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ETL service not available. Install with: pip install semantic-rdf-mapper"
        )
    
    try:
        etl = create_etl_service()
        
        # Save uploaded files to temp location
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / data_file.filename
            ont_path = Path(tmpdir) / ontology_file.filename
            
            with open(data_path, "wb") as f:
                f.write(await data_file.read())
            with open(ont_path, "wb") as f:
                f.write(await ontology_file.read())
            
            # Generate mapping
            mapping = etl.generate_mapping(
                data_file=data_path,
                ontology_file=ont_path,
                target_class=target_class,
                base_iri=base_iri,
                use_ai=use_ai,
            )
            
            return GenerateMappingResponse(
                mapping=mapping,
                analysis={},  # Could extract from generator
                confidence_scores={},  # Could extract from generator
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mapping generation failed: {str(e)}")


@router.post("/convert", response_model=ConvertResponse)
async def convert_data(
    data_file: UploadFile = File(..., description="Data file to convert"),
    mapping: str = Form(..., description="Mapping configuration as JSON string"),
    output_format: str = Form("ttl"),
    run_validation: bool = Form(False),
    limit: Optional[int] = Form(None),
    ontology_file: Optional[UploadFile] = File(None, description="Optional ontology for validation"),
    shacl_file: Optional[UploadFile] = File(None, description="Optional SHACL shapes for validation"),
    load_to_repository: Optional[str] = Form(None),
):
    """
    Convert tabular data to RDF using a mapping configuration.
    
    Supports CSV, Excel, JSON, and XML input formats.
    Output can be Turtle, N-Triples, JSON-LD, or RDF/XML.
    """
    if not ETL_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ETL service not available. Install with: pip install semantic-rdf-mapper"
        )
    
    import json
    
    try:
        mapping_config = json.loads(mapping)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid mapping JSON: {str(e)}")
    
    try:
        etl = create_etl_service()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save data file
            data_path = Path(tmpdir) / data_file.filename
            with open(data_path, "wb") as f:
                f.write(await data_file.read())
            
            # Save optional files
            ont_path = None
            if ontology_file:
                ont_path = Path(tmpdir) / ontology_file.filename
                with open(ont_path, "wb") as f:
                    f.write(await ontology_file.read())
            
            shacl_path = None
            if shacl_file:
                shacl_path = Path(tmpdir) / shacl_file.filename
                with open(shacl_path, "wb") as f:
                    f.write(await shacl_file.read())
            
            # Run conversion
            rdf_content, report = etl.convert(
                data_file=data_path,
                mapping_config=mapping_config,
                output_format=OutputFormat(output_format),
                ontology_file=ont_path,
                validate=run_validation,
                shacl_shapes=shacl_path,
                limit=limit,
            )
            
            # Optionally load to repository
            loaded = False
            if load_to_repository:
                # Import here to avoid circular imports
                from rdf_starbase.repositories import get_repository
                try:
                    repo = get_repository(load_to_repository)
                    if repo:
                        repo.load_rdf(rdf_content, format=output_format)
                        loaded = True
                except Exception as e:
                    report["warnings"] = report.get("warnings", []) + [
                        f"Failed to load to repository: {str(e)}"
                    ]
            
            return ConvertResponse(
                rdf_content=rdf_content,
                triple_count=report.get("triple_count", 0),
                format=output_format,
                warnings=report.get("warnings", []),
                validation=report.get("validation"),
                loaded=loaded,
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")


@router.post("/yarrrml/export", response_model=YARRRMLExportResponse)
async def export_yarrrml(request: YARRRMLExportRequest):
    """
    Export a mapping configuration to YARRRML format.
    
    YARRRML is a human-friendly YAML syntax for RML mappings,
    compatible with major RML processors (RMLMapper, RocketRML, Morph-KGC).
    """
    if not ETL_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ETL service not available. Install with: pip install semantic-rdf-mapper"
        )
    
    try:
        etl = create_etl_service()
        yarrrml_content = etl.mapping_to_yarrrml(request.mapping)
        
        return YARRRMLExportResponse(
            yarrrml=yarrrml_content,
            format="yarrrml"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YARRRML export failed: {str(e)}")


@router.post("/yarrrml/import", response_model=YARRRMLImportResponse)
async def import_yarrrml(
    yarrrml_file: UploadFile = File(..., description="YARRRML file to import")
):
    """
    Import a YARRRML mapping file and convert to internal format.
    
    Supports standard YARRRML syntax as well as extended YARRRML-STAR
    features for RDF-Star quoted triples.
    """
    if not ETL_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ETL service not available. Install with: pip install semantic-rdf-mapper"
        )
    
    try:
        etl = create_etl_service()
        
        content = await yarrrml_file.read()
        yarrrml_content = content.decode("utf-8")
        
        mapping = etl.yarrrml_to_mapping(yarrrml_content)
        
        return YARRRMLImportResponse(
            mapping=mapping,
            source_format="yarrrml"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YARRRML import failed: {str(e)}")


@router.post("/mapping/from-starchart")
async def create_mapping_from_starchart(
    mappings: dict[str, Any] = Form(..., description="Starchart column mappings"),
    source_filename: str = Form(..., description="Original data filename"),
    base_iri: str = Form("http://example.org/"),
    target_class: Optional[str] = Form(None),
):
    """
    Convert Starchart UI mappings to rdfmap configuration format.
    
    This endpoint bridges the Starchart visual mapper with the rdfmap
    conversion engine, allowing direct ETL from the UI.
    """
    import json
    
    try:
        if isinstance(mappings, str):
            mappings = json.loads(mappings)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid mappings JSON: {str(e)}")
    
    # Convert Starchart mappings to rdfmap format
    mapping_config = {
        "namespaces": {
            "ex": base_iri,
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "foaf": "http://xmlns.com/foaf/0.1/",
            "schema": "http://schema.org/",
        },
        "defaults": {
            "base_iri": base_iri,
        },
        "sheets": [
            {
                "name": "main",
                "source": source_filename,
                "row_resource": {
                    "class": target_class or "ex:Record",
                    "iri_template": f"{base_iri}record/" + "{_row_id}",
                },
                "columns": {},
            }
        ]
    }
    
    # Convert each column mapping
    for column_name, property_uri in mappings.items():
        if property_uri:
            # Determine datatype from property
            datatype = "xsd:string"  # Default
            if "date" in property_uri.lower():
                datatype = "xsd:date"
            elif "price" in property_uri.lower() or "amount" in property_uri.lower():
                datatype = "xsd:decimal"
            elif "age" in property_uri.lower() or "count" in property_uri.lower():
                datatype = "xsd:integer"
            
            mapping_config["sheets"][0]["columns"][column_name] = {
                "as": property_uri,
                "datatype": datatype,
            }
    
    return {"mapping": mapping_config}
