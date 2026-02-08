"""
ETL Service - Transform tabular data to RDF using semantic-rdf-mapper (rdfmap)

Supports:
- CSV, Excel, JSON, XML data sources
- RML, YARRRML, and YARRRML-STAR mapping formats
- AI-powered automatic mapping generation
- SHACL validation of output
- Direct loading into RDF-StarBase repositories
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    from rdfmap.cli import app as rdfmap_cli
    from rdfmap.converter import RDFConverter
    from rdfmap.generator import MappingGenerator
    from rdfmap.models.mapping import MappingConfig
    from rdfmap.validators.shacl_validator import SHACLValidator
    RDFMAP_AVAILABLE = True
except ImportError:
    RDFMAP_AVAILABLE = False


class MappingFormat(str, Enum):
    """Supported mapping configuration formats."""
    YARRRML = "yarrrml"
    RML = "rml"
    INTERNAL = "internal"  # rdfmap's native YAML format


class OutputFormat(str, Enum):
    """Supported RDF output formats."""
    TURTLE = "ttl"
    NTRIPLES = "nt"
    JSONLD = "jsonld"
    RDFXML = "xml"
    TRIG = "trig"  # For named graphs


@dataclass
class ETLJob:
    """Represents an ETL transformation job."""
    
    id: str
    name: str
    source_file: Path
    mapping_config: dict[str, Any]
    output_format: OutputFormat = OutputFormat.TURTLE
    ontology_file: Optional[Path] = None
    shacl_shapes_file: Optional[Path] = None
    base_iri: str = "http://example.org/"
    
    # Results
    status: str = "pending"  # pending, running, completed, failed
    output_file: Optional[Path] = None
    triple_count: int = 0
    error_message: Optional[str] = None
    validation_report: Optional[dict] = None
    warnings: list[str] = field(default_factory=list)


class ETLService:
    """
    ETL Service for transforming tabular data to RDF.
    
    Wraps the semantic-rdf-mapper (rdfmap) library to provide:
    - Programmatic mapping generation
    - Data conversion to RDF
    - SHACL validation
    - YARRRML import/export
    """
    
    def __init__(self):
        if not RDFMAP_AVAILABLE:
            raise ImportError(
                "semantic-rdf-mapper is not installed. "
                "Install with: pip install semantic-rdf-mapper"
            )
    
    def generate_mapping(
        self,
        data_file: Path,
        ontology_file: Path,
        target_class: Optional[str] = None,
        base_iri: str = "http://example.org/",
        use_ai: bool = True,
    ) -> dict[str, Any]:
        """
        Auto-generate a mapping configuration from data and ontology.
        
        Uses AI-powered semantic matching to align columns with ontology properties.
        
        Args:
            data_file: Path to CSV/Excel/JSON/XML data file
            ontology_file: Path to OWL/RDFS ontology file
            target_class: Target ontology class (auto-detected if None)
            base_iri: Base IRI for generated resources
            use_ai: Use BERT embeddings for semantic matching
            
        Returns:
            Mapping configuration as dictionary
        """
        generator = MappingGenerator(
            ontology_path=str(ontology_file),
            base_iri=base_iri,
            use_embeddings=use_ai,
        )
        
        mapping = generator.generate(
            data_path=str(data_file),
            target_class=target_class,
        )
        
        return mapping.model_dump()
    
    def convert(
        self,
        data_file: Path,
        mapping_config: dict[str, Any],
        output_format: OutputFormat = OutputFormat.TURTLE,
        ontology_file: Optional[Path] = None,
        validate: bool = False,
        shacl_shapes: Optional[Path] = None,
        limit: Optional[int] = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Convert tabular data to RDF using a mapping configuration.
        
        Args:
            data_file: Path to source data file
            mapping_config: Mapping configuration dictionary
            output_format: Desired RDF output format
            ontology_file: Optional ontology for validation
            validate: Run SHACL validation after conversion
            shacl_shapes: SHACL shapes file for validation
            limit: Limit number of rows to process (for testing)
            
        Returns:
            Tuple of (rdf_content, report_dict)
        """
        # Create mapping config object
        config = MappingConfig(**mapping_config)
        
        # Initialize converter
        converter = RDFConverter(
            mapping_config=config,
            ontology_path=str(ontology_file) if ontology_file else None,
        )
        
        # Run conversion
        graph = converter.convert(
            data_path=str(data_file),
            limit=limit,
        )
        
        # Serialize to requested format
        rdf_content = graph.serialize(format=output_format.value)
        
        # Build report
        report = {
            "triple_count": len(graph),
            "format": output_format.value,
            "warnings": converter.warnings if hasattr(converter, 'warnings') else [],
        }
        
        # Optional SHACL validation
        if validate and shacl_shapes:
            validator = SHACLValidator(shapes_path=str(shacl_shapes))
            validation_result = validator.validate(graph)
            report["validation"] = {
                "conforms": validation_result.conforms,
                "violations": [
                    {
                        "focus_node": str(v.focus_node),
                        "path": str(v.result_path) if v.result_path else None,
                        "message": str(v.message),
                        "severity": str(v.severity),
                    }
                    for v in validation_result.violations
                ] if not validation_result.conforms else []
            }
        
        return rdf_content, report
    
    def mapping_to_yarrrml(self, mapping_config: dict[str, Any]) -> str:
        """
        Export mapping configuration to YARRRML format.
        
        YARRRML is a human-friendly YAML-based syntax for RML mappings,
        compatible with RMLMapper, RocketRML, Morph-KGC, and SDM-RDFizer.
        
        Args:
            mapping_config: Internal mapping configuration
            
        Returns:
            YARRRML content as string
        """
        yarrrml = self._convert_to_yarrrml(mapping_config)
        return yaml.dump(yarrrml, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    def yarrrml_to_mapping(self, yarrrml_content: str) -> dict[str, Any]:
        """
        Import YARRRML mapping and convert to internal format.
        
        Args:
            yarrrml_content: YARRRML YAML content
            
        Returns:
            Internal mapping configuration
        """
        yarrrml = yaml.safe_load(yarrrml_content)
        return self._convert_from_yarrrml(yarrrml)
    
    def _convert_to_yarrrml(self, mapping: dict[str, Any]) -> dict[str, Any]:
        """Convert internal mapping format to YARRRML."""
        yarrrml = {
            "prefixes": mapping.get("namespaces", {}),
            "mappings": {}
        }
        
        # Convert each sheet to a YARRRML mapping
        for sheet in mapping.get("sheets", []):
            mapping_name = sheet.get("name", "default")
            source_file = sheet.get("source", "data.csv")
            row_resource = sheet.get("row_resource", {})
            
            yarrrml_mapping = {
                "sources": [[source_file + "~csv"]],
                "s": row_resource.get("iri_template", "http://example.org/{id}"),
                "po": []
            }
            
            # Add rdf:type
            if "class" in row_resource:
                yarrrml_mapping["po"].append(["a", row_resource["class"]])
            
            # Convert column mappings
            for col_name, col_config in sheet.get("columns", {}).items():
                if isinstance(col_config, dict):
                    predicate = col_config.get("as", f"ex:{col_name}")
                    datatype = col_config.get("datatype")
                    
                    po_entry = [predicate, f"$({col_name})"]
                    if datatype:
                        po_entry = [predicate, f"$({col_name})~{datatype}"]
                    
                    yarrrml_mapping["po"].append(po_entry)
            
            # Convert object mappings (relationships)
            for obj_name, obj_config in sheet.get("objects", {}).items():
                predicate = obj_config.get("predicate", f"ex:has{obj_name.title()}")
                iri_template = obj_config.get("iri_template", f"http://example.org/{obj_name}/" + "{" + obj_name + "_id}")
                
                yarrrml_mapping["po"].append([
                    predicate,
                    {"mapping": f"{obj_name}_mapping"}
                ])
                
                # Create linked mapping if it has properties
                if obj_config.get("properties"):
                    linked_mapping = {
                        "sources": [[source_file + "~csv"]],
                        "s": iri_template,
                        "po": []
                    }
                    
                    if "class" in obj_config:
                        linked_mapping["po"].append(["a", obj_config["class"]])
                    
                    for prop in obj_config.get("properties", []):
                        col = prop.get("column")
                        pred = prop.get("as")
                        if col and pred:
                            linked_mapping["po"].append([pred, f"$({col})"])
                    
                    yarrrml["mappings"][f"{obj_name}_mapping"] = linked_mapping
            
            yarrrml["mappings"][mapping_name] = yarrrml_mapping
        
        return yarrrml
    
    def _convert_from_yarrrml(self, yarrrml: dict[str, Any]) -> dict[str, Any]:
        """Convert YARRRML to internal mapping format."""
        mapping = {
            "namespaces": yarrrml.get("prefixes", {}),
            "defaults": {
                "base_iri": "http://example.org/"
            },
            "sheets": []
        }
        
        for mapping_name, yarrrml_mapping in yarrrml.get("mappings", {}).items():
            # Parse source
            sources = yarrrml_mapping.get("sources", [[]])
            source_spec = sources[0][0] if sources and sources[0] else "data.csv"
            source_file = source_spec.replace("~csv", "").replace("~json", "")
            
            sheet = {
                "name": mapping_name,
                "source": source_file,
                "row_resource": {
                    "iri_template": yarrrml_mapping.get("s", "http://example.org/{id}")
                },
                "columns": {},
                "objects": {}
            }
            
            # Parse predicate-object pairs
            for po in yarrrml_mapping.get("po", []):
                if len(po) < 2:
                    continue
                    
                predicate, obj = po[0], po[1]
                
                # Handle rdf:type
                if predicate == "a":
                    sheet["row_resource"]["class"] = obj
                    continue
                
                # Handle object references
                if isinstance(obj, dict) and "mapping" in obj:
                    # This is a relationship to another mapping
                    continue
                
                # Handle literal values
                if isinstance(obj, str):
                    # Parse column reference: $(column_name) or $(column_name)~datatype
                    if obj.startswith("$(") and ")" in obj:
                        col_part = obj[2:]
                        if "~" in col_part:
                            col_name, datatype = col_part.split("~", 1)
                            col_name = col_name.rstrip(")")
                            datatype = datatype.rstrip(")")
                        else:
                            col_name = col_part.rstrip(")")
                            datatype = None
                        
                        sheet["columns"][col_name] = {
                            "as": predicate,
                        }
                        if datatype:
                            sheet["columns"][col_name]["datatype"] = datatype
            
            mapping["sheets"].append(sheet)
        
        return mapping


def create_etl_service() -> ETLService:
    """Factory function to create ETL service instance."""
    return ETLService()
