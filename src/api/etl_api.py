"""
ETL API Endpoints - REST API for columnar RDF transformation

Provides endpoints for:
- Converting tabular data to RDF using YARRRML mappings
- Importing/exporting YARRRML mappings
- Direct loading into repositories
- RDF-Star metadata generation
- Async job processing for large datasets with progress polling
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
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
# Job Management for Large Dataset Processing
# ============================================================================

class JobStatus(str, Enum):
    """Status of an ETL job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ETLJob:
    """Represents an async ETL transformation job."""
    
    def __init__(self, job_id: str, config: dict, total_rows: int = 0):
        self.job_id = job_id
        self.config = config
        self.status = JobStatus.PENDING
        self.created_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.total_rows = total_rows
        self.processed_rows = 0
        self.triple_count = 0
        self.annotation_count = 0
        self.progress_percent = 0.0
        self.current_phase = "initializing"
        self.error: Optional[str] = None
        self.warnings: list[str] = []
        self.result: Optional[dict] = None
        self._rdf_content: Optional[str] = None
        self._lock = threading.Lock()
    
    def update_progress(self, rows: int, phase: str = None):
        """Thread-safe progress update."""
        with self._lock:
            self.processed_rows = rows
            if self.total_rows > 0:
                self.progress_percent = min(100.0, (rows / self.total_rows) * 100)
            if phase:
                self.current_phase = phase
    
    def to_dict(self, include_result: bool = False) -> dict:
        """Convert job to dictionary for API response."""
        data = {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_rows": self.total_rows,
            "processed_rows": self.processed_rows,
            "triple_count": self.triple_count,
            "annotation_count": self.annotation_count,
            "progress_percent": round(self.progress_percent, 1),
            "current_phase": self.current_phase,
            "error": self.error,
            "warnings": self.warnings,
        }
        if include_result and self.result:
            data["result"] = self.result
        return data


# In-memory job store (in production, use Redis or database)
_job_store: dict[str, ETLJob] = {}
_job_store_lock = threading.Lock()

def get_job(job_id: str) -> Optional[ETLJob]:
    """Get a job by ID."""
    with _job_store_lock:
        return _job_store.get(job_id)

def create_job(config: dict, total_rows: int = 0) -> ETLJob:
    """Create a new job."""
    job_id = str(uuid.uuid4())[:8]  # Short ID for easier use
    job = ETLJob(job_id, config, total_rows)
    with _job_store_lock:
        _job_store[job_id] = job
        # Clean up old completed jobs (keep last 100)
        if len(_job_store) > 100:
            completed = [
                (jid, j) for jid, j in _job_store.items() 
                if j.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
            ]
            completed.sort(key=lambda x: x[1].created_at)
            for jid, _ in completed[:-50]:  # Keep most recent 50 completed
                del _job_store[jid]
    return job


# ============================================================================
# RDF-Star Annotation Generator
# ============================================================================

def add_rdfstar_annotations(
    rdf_content: str,
    source: str,
    confidence: float = 1.0,
    generate_timestamp: bool = True,
    output_format: str = "ttl",
) -> tuple[str, int]:
    """
    Add RDF-Star provenance annotations to existing RDF triples.
    
    For each triple in the input, generates annotations using:
    - prov:wasDerivedFrom (source)
    - prov:value (confidence)  
    - prov:generatedAtTime (timestamp)
    
    Handles both single-line triples and grouped Turtle syntax.
    
    Returns:
        Tuple of (annotated RDF content, annotation count)
    """
    if output_format not in ('ttl', 'turtle'):
        # For now, only support Turtle format for RDF-Star
        return rdf_content, 0
    
    lines = rdf_content.split('\n')
    output_lines = []
    annotations = []
    annotation_count = 0
    
    # Add PROV prefix if not present
    has_prov_prefix = False
    has_xsd_prefix = False
    for line in lines:
        if '@prefix prov:' in line.lower():
            has_prov_prefix = True
        if '@prefix xsd:' in line.lower():
            has_xsd_prefix = True
    
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Parse Turtle to extract triples
    # Handle both single-line and grouped predicate-object syntax
    current_subject = None
    
    for line in lines:
        output_lines.append(line)
        stripped = line.strip()
        
        # Skip prefixes, comments, blank lines
        if not stripped or stripped.startswith('@') or stripped.startswith('#'):
            continue
        
        # Check if this starts a new subject (non-whitespace at start of line)
        if not line[0].isspace() if line else False:
            # This might be a new subject
            # Pattern: <subject> or prefix:local 
            subject_match = re.match(r'^(<[^>]+>|[a-zA-Z_][a-zA-Z0-9_]*:[a-zA-Z0-9_-]+)', stripped)
            if subject_match:
                current_subject = subject_match.group(1)
        
        # Skip lines that look like structure/blank nodes
        if '[' in stripped or ']' in stripped or stripped.startswith('a '):
            continue
        
        # Match predicate-object pairs (both single-line and grouped)
        # Single line: <s> <p> <o> .
        # Grouped: <p> <o> ; or <p> <o> .
        
        # Try to match a complete triple on one line
        full_triple_match = re.match(
            r'^(<[^>]+>|[a-zA-Z_][a-zA-Z0-9_]*:[a-zA-Z0-9_-]+)\s+'
            r'(<[^>]+>|[a-zA-Z_][a-zA-Z0-9_]*:[a-zA-Z0-9_-]+|a)\s+'
            r'(.+?)\s*[\.;]\s*$',
            stripped
        )
        
        if full_triple_match:
            subject = full_triple_match.group(1)
            predicate = full_triple_match.group(2)
            obj = full_triple_match.group(3).rstrip(' .;')
            annotations.append((subject, predicate, obj))
            annotation_count += 1
            continue
        
        # Try to match predicate-object (grouped syntax continuation)
        # e.g., "    schema:name "Product" ;"
        if current_subject and stripped:
            po_match = re.match(
                r'^(<[^>]+>|[a-zA-Z_][a-zA-Z0-9_]*:[a-zA-Z0-9_-]+|a)\s+'
                r'(.+?)\s*[\.;,]\s*$',
                stripped
            )
            if po_match:
                predicate = po_match.group(1)
                obj = po_match.group(2).rstrip(' .;,')
                # Don't add annotations for type declarations
                if predicate != 'a':
                    annotations.append((current_subject, predicate, obj))
                    annotation_count += 1
    
    # Generate RDF-Star annotations at the end
    if annotations:
        output_lines.append('\n# RDF-Star provenance annotations')
        
        for subject, predicate, obj in annotations:
            annotation_parts = []
            
            # prov:wasDerivedFrom
            if source:
                if source.startswith('http://') or source.startswith('https://') or source.startswith('urn:'):
                    annotation_parts.append(f'prov:wasDerivedFrom <{source}>')
                else:
                    annotation_parts.append(f'prov:wasDerivedFrom "{source}"')
            
            # prov:value (confidence)
            if confidence is not None:
                annotation_parts.append(f'prov:value {confidence}')
            
            # prov:generatedAtTime
            if generate_timestamp:
                annotation_parts.append(f'prov:generatedAtTime "{timestamp}"^^xsd:dateTime')
            
            if annotation_parts:
                annotation = f'<< {subject} {predicate} {obj} >> {" ; ".join(annotation_parts)} .'
                output_lines.append(annotation)
    
    # Add PROV and XSD prefixes at the start if needed
    prefixes_to_add = []
    if not has_prov_prefix and annotation_count > 0:
        prefixes_to_add.append('@prefix prov: <http://www.w3.org/ns/prov#> .')
    if not has_xsd_prefix and annotation_count > 0:
        prefixes_to_add.append('@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .')
    
    if prefixes_to_add:
        # Find where to insert (after other prefixes)
        insert_idx = 0
        for i, line in enumerate(output_lines):
            if line.strip().startswith('@prefix'):
                insert_idx = i + 1
            elif line.strip() and not line.strip().startswith('#'):
                break
        for prefix in reversed(prefixes_to_add):
            output_lines.insert(insert_idx, prefix)
    
    return '\n'.join(output_lines), annotation_count


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
    annotation_count: int = Field(0, description="Number of RDF-Star annotations generated")
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


class JobCreateResponse(BaseModel):
    """Response when creating a new job."""
    job_id: str = Field(..., description="Unique job identifier for polling")
    status: str = Field(..., description="Initial job status")
    message: str = Field(..., description="Status message")


class JobStatusResponse(BaseModel):
    """Response with job status and progress."""
    job_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_rows: int = 0
    processed_rows: int = 0
    triple_count: int = 0
    annotation_count: int = 0
    progress_percent: float = 0.0
    current_phase: str = "pending"
    error: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


class JobResultResponse(BaseModel):
    """Response with completed job result."""
    job_id: str
    status: str
    rdf_content: str = ""
    triple_count: int = 0
    annotation_count: int = 0
    row_count: int = 0
    format: str = "ttl"
    warnings: list[str] = Field(default_factory=list)
    loaded: bool = False


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
            "async_jobs",
        ]
    )


# ============================================================================
# Async Job Processing for Large Datasets
# ============================================================================

def run_etl_job(
    job: ETLJob,
    data_path: Path,
    repo_manager: Any,
) -> None:
    """
    Background task to run ETL transformation.
    Updates job progress as it processes.
    """
    try:
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        job.current_phase = "parsing"
        
        config = job.config
        mapping_config = config['mapping_config']
        output_format = config['output_format']
        export_mode = config.get('export_mode', 'plain')
        metadata_options = config.get('metadata_options')
        target_repo = config.get('target_repo')
        
        # Create engine
        engine = create_etl_engine()
        
        # Map output format
        fmt_map = {
            'ttl': OutputFormat.TURTLE,
            'turtle': OutputFormat.TURTLE,
            'nt': OutputFormat.NTRIPLES,
            'ntriples': OutputFormat.NTRIPLES,
            'jsonld': OutputFormat.JSONLD,
            'json-ld': OutputFormat.JSONLD,
        }
        out_fmt = fmt_map.get(output_format.lower(), OutputFormat.TURTLE)
        
        job.current_phase = "transforming"
        
        # Run transformation with progress callback
        def progress_callback(rows_processed: int, total_rows: int):
            # Get phase from engine if available
            phase = getattr(engine, '_current_phase', 'transforming')
            job.update_progress(rows_processed, phase)
            job.total_rows = total_rows
        
        rdf_content, report = engine.transform(
            data_file=data_path,
            mapping_config=mapping_config,
            output_format=out_fmt,
            progress_callback=progress_callback,
        )
        
        job.triple_count = report.get('triple_count', 0)
        job.processed_rows = report.get('row_count', 0)
        job.warnings = report.get('warnings', [])
        
        # Generate RDF-Star annotations if enabled
        annotation_count = 0
        if export_mode in ('rdfstar', 'load') and metadata_options and metadata_options.get('enabled'):
            job.current_phase = "annotating"
            rdf_content, annotation_count = add_rdfstar_annotations(
                rdf_content=rdf_content,
                source=metadata_options.get('source', f'starchart:etl_job'),
                confidence=metadata_options.get('confidence', 1.0),
                generate_timestamp=metadata_options.get('generate_timestamp', True),
                output_format=output_format,
            )
        
        job.annotation_count = annotation_count
        
        # Load to repository if requested
        loaded = False
        if export_mode == 'load' and target_repo and repo_manager:
            job.current_phase = "loading"
            try:
                store = repo_manager.get_store(target_repo)
                
                # Write content to temp file and load
                ext_map = {'ttl': '.ttl', 'nt': '.nt', 'jsonld': '.jsonld', 'xml': '.rdf'}
                ext = ext_map.get(output_format, '.ttl')
                with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False, encoding='utf-8') as f:
                    f.write(rdf_content)
                    temp_path = f.name
                
                try:
                    temp_path_obj = Path(temp_path)
                    file_uri = temp_path_obj.as_uri()
                    count = store.load_graph(file_uri)
                    repo_manager.save(target_repo)
                    loaded = True
                finally:
                    import os
                    os.unlink(temp_path)
            except Exception as e:
                job.warnings.append(f"Failed to load to repository: {str(e)}")
        
        # Store result
        job._rdf_content = rdf_content
        job.result = {
            'rdf_content': rdf_content,
            'triple_count': job.triple_count,
            'annotation_count': job.annotation_count,
            'row_count': job.processed_rows,
            'format': output_format,
            'warnings': job.warnings,
            'loaded': loaded,
        }
        
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        job.current_phase = "completed"
        job.progress_percent = 100.0
        
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now(timezone.utc)
        job.current_phase = "failed"
    finally:
        # Clean up data file
        try:
            if data_path.exists():
                data_path.unlink()
        except:
            pass


@router.post("/jobs", response_model=JobCreateResponse)
async def create_etl_job(
    request: Request,
    background_tasks: BackgroundTasks,
    data_file: UploadFile = File(..., description="Data file to convert"),
    mapping: str = Form(..., description="Mapping configuration as JSON string"),
    output_format: str = Form("ttl"),
    metadata: Optional[str] = Form(None, description="RDF-Star metadata options as JSON"),
    export_mode: str = Form("plain", description="Export mode: plain, rdfstar, load"),
    repository: Optional[str] = Form(None, description="Target repository for load mode"),
):
    """
    Create an async ETL job for large dataset processing.
    
    Returns a job_id immediately. Poll /etl/jobs/{job_id} for progress.
    When complete, fetch result from /etl/jobs/{job_id}/result.
    """
    try:
        mapping_config = json.loads(mapping)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid mapping JSON: {str(e)}")
    
    # Parse metadata options
    metadata_options = None
    if metadata:
        try:
            metadata_options = json.loads(metadata)
        except json.JSONDecodeError:
            metadata_options = None
    
    # Save uploaded file to temp location
    tmp_dir = Path(tempfile.gettempdir()) / "etl_jobs"
    tmp_dir.mkdir(exist_ok=True)
    
    job_file_id = str(uuid.uuid4())[:8]
    data_path = tmp_dir / f"{job_file_id}_{data_file.filename}"
    
    content = await data_file.read()
    data_path.write_bytes(content)
    
    # Count rows for progress tracking (for CSV)
    total_rows = 0
    if data_file.filename.lower().endswith('.csv'):
        try:
            total_rows = content.decode('utf-8').count('\n') - 1  # Minus header
        except:
            pass
    
    # Create job config
    config = {
        'mapping_config': mapping_config,
        'output_format': output_format,
        'export_mode': export_mode,
        'metadata_options': metadata_options,
        'target_repo': repository,
        'filename': data_file.filename,
    }
    
    # Create job
    job = create_job(config, total_rows)
    
    # Get repo manager for loading
    repo_manager = getattr(request.app.state, 'repo_manager', None)
    
    # Run in background thread (not asyncio) for CPU-bound work
    def run_job():
        run_etl_job(job, data_path, repo_manager)
    
    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()
    
    return JobCreateResponse(
        job_id=job.job_id,
        status=job.status.value,
        message=f"Job created. Poll /etl/jobs/{job.job_id} for progress."
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status and progress of an ETL job.
    
    Poll this endpoint to track progress of long-running jobs.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JobStatusResponse(**job.to_dict())


@router.get("/jobs/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(job_id: str):
    """
    Get the result of a completed ETL job.
    
    Only available after job status is 'completed'.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status == JobStatus.RUNNING or job.status == JobStatus.PENDING:
        raise HTTPException(
            status_code=202, 
            detail=f"Job {job_id} is still {job.status.value}. Poll /etl/jobs/{job_id} for progress."
        )
    
    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=f"Job failed: {job.error}")
    
    if not job.result:
        raise HTTPException(status_code=500, detail="Job completed but no result available")
    
    return JobResultResponse(
        job_id=job.job_id,
        status=job.status.value,
        rdf_content=job.result.get('rdf_content', ''),
        triple_count=job.result.get('triple_count', 0),
        annotation_count=job.result.get('annotation_count', 0),
        row_count=job.result.get('row_count', 0),
        format=job.result.get('format', 'ttl'),
        warnings=job.result.get('warnings', []),
        loaded=job.result.get('loaded', False),
    )


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job (best effort)."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
        return {"message": f"Job {job_id} already {job.status.value}"}
    
    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.now(timezone.utc)
    return {"message": f"Job {job_id} cancelled"}


@router.get("/jobs")
async def list_jobs(limit: int = 20):
    """List recent ETL jobs."""
    with _job_store_lock:
        jobs = sorted(
            _job_store.values(),
            key=lambda j: j.created_at,
            reverse=True
        )[:limit]
        return {
            "count": len(jobs),
            "jobs": [j.to_dict() for j in jobs]
        }


@router.post("/convert", response_model=ConvertResponse)
async def convert_data(
    request: Request,
    data_file: UploadFile = File(..., description="Data file to convert"),
    mapping: str = Form(..., description="Mapping configuration as JSON string"),
    output_format: str = Form("ttl"),
    limit: Optional[int] = Form(None),
    load_to_repository: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None, description="RDF-Star metadata options as JSON"),
    export_mode: str = Form("plain", description="Export mode: plain, rdfstar, load"),
    repository: Optional[str] = Form(None, description="Target repository for load mode"),
):
    """
    Convert tabular data to RDF using columnar processing.
    
    Supports CSV, Excel, JSON input formats.
    Output can be Turtle, N-Triples, or JSON-LD.
    
    Export modes:
    - plain: Standard RDF triples only
    - rdfstar: RDF-Star with provenance annotations
    - load: Transform and load directly to repository
    """
    try:
        mapping_config = json.loads(mapping)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid mapping JSON: {str(e)}")
    
    # Parse metadata options
    metadata_options = None
    if metadata:
        try:
            metadata_options = json.loads(metadata)
        except json.JSONDecodeError:
            metadata_options = None
    
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
            
            # Generate RDF-Star annotations if enabled
            annotation_count = 0
            if export_mode in ('rdfstar', 'load') and metadata_options and metadata_options.get('enabled'):
                rdf_content, annotation_count = add_rdfstar_annotations(
                    rdf_content=rdf_content,
                    source=metadata_options.get('source', f'starchart:{data_file.filename}'),
                    confidence=metadata_options.get('confidence', 1.0),
                    generate_timestamp=metadata_options.get('generate_timestamp', True),
                    output_format=output_format,
                )
            
            # Load to repository if requested
            loaded = False
            warnings = report.get('warnings', [])
            target_repo = repository or load_to_repository
            
            if export_mode == 'load' and target_repo:
                try:
                    # Use the shared repo_manager from app.state
                    repo_manager = getattr(request.app.state, 'repo_manager', None)
                    if not repo_manager:
                        raise ValueError("Repository manager not configured")
                    
                    store = repo_manager.get_store(target_repo)
                    
                    # Write content to temp file and load
                    ext_map = {'ttl': '.ttl', 'nt': '.nt', 'jsonld': '.jsonld', 'xml': '.rdf'}
                    ext = ext_map.get(output_format, '.ttl')
                    with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False, encoding='utf-8') as f:
                        f.write(rdf_content)
                        temp_path = f.name
                    
                    try:
                        # Convert Windows path to proper file URI format
                        # Windows: C:\path\file.ttl -> file:///C:/path/file.ttl
                        temp_path_obj = Path(temp_path)
                        file_uri = temp_path_obj.as_uri()  # Handles Windows/Unix correctly
                        
                        count = store.load_graph(file_uri)
                        repo_manager.save(target_repo)
                        loaded = True
                        
                        # Update triple count in response to reflect actual loaded count
                        if count > 0:
                            report['triple_count'] = report.get('triple_count', 0) + count
                    finally:
                        import os
                        os.unlink(temp_path)
                except Exception as e:
                    import traceback
                    warnings.append(f"Failed to load to repository: {str(e)}")
            
            return ConvertResponse(
                rdf_content=rdf_content,
                triple_count=report.get('triple_count', 0),
                annotation_count=annotation_count,
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


# ============================================================================
# Embedding-based Property Recommendations
# ============================================================================

class PropertyRecommendRequest(BaseModel):
    """Request for property recommendations."""
    column_header: str = Field(..., description="CSV column header to match")
    properties: list[dict] = Field(..., description="List of ontology properties")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of recommendations")


class BatchRecommendRequest(BaseModel):
    """Request for batch property recommendations."""
    column_headers: list[str] = Field(..., description="List of column headers")
    properties: list[dict] = Field(..., description="List of ontology properties")
    top_k: int = Field(default=5, ge=1, le=20, description="Results per column")


class RecommendationResult(BaseModel):
    """Single property recommendation."""
    uri: str
    label: str
    score: float
    confidence: str  # high, medium, low
    match_type: str  # label, alias, description


class PropertyRecommendResponse(BaseModel):
    """Response for property recommendations."""
    column: str
    recommendations: list[RecommendationResult]
    using_embeddings: bool


class BatchRecommendResponse(BaseModel):
    """Response for batch recommendations."""
    results: dict[str, list[RecommendationResult]]
    using_embeddings: bool


@router.post("/recommend", response_model=PropertyRecommendResponse)
async def recommend_properties(request: PropertyRecommendRequest):
    """
    Get property recommendations for a column header using semantic embeddings.
    
    Uses sentence-transformers for high-quality semantic similarity matching.
    Falls back to string-based matching if embeddings not available.
    
    The request should include:
    - column_header: The CSV column name to match
    - properties: List of ontology properties with uri, label, aliases, description
    - top_k: Number of recommendations to return (1-20)
    
    Each property in the list should have:
    - uri (required): The property URI
    - label (optional): Human-readable label
    - aliases (optional): List of alternative names
    - description (optional): Property description
    """
    try:
        from rdf_starbase.embeddings import rank_column_to_properties, get_embedder
        
        embedder = get_embedder()
        using_embeddings = embedder.embeddings_available
        
        results = rank_column_to_properties(
            request.column_header,
            request.properties,
            request.top_k,
        )
        
        return PropertyRecommendResponse(
            column=request.column_header,
            recommendations=[RecommendationResult(**r) for r in results],
            using_embeddings=using_embeddings,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}"
        )


@router.post("/recommend/batch", response_model=BatchRecommendResponse)
async def recommend_properties_batch(request: BatchRecommendRequest):
    """
    Get property recommendations for multiple columns efficiently.
    
    This batches all columns together for efficient embedding computation.
    Much faster than calling /recommend multiple times for many columns.
    """
    try:
        from rdf_starbase.embeddings import batch_rank_columns_to_properties, get_embedder
        
        embedder = get_embedder()
        using_embeddings = embedder.embeddings_available
        
        results = batch_rank_columns_to_properties(
            request.column_headers,
            request.properties,
            request.top_k,
        )
        
        return BatchRecommendResponse(
            results={
                col: [RecommendationResult(**r) for r in recs]
                for col, recs in results.items()
            },
            using_embeddings=using_embeddings,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch recommendation failed: {str(e)}"
        )


@router.get("/recommend/status")
async def recommend_status():
    """
    Check if embedding-based recommendations are available.
    
    Returns whether sentence-transformers is installed and working.
    """
    try:
        from rdf_starbase.embeddings import get_embedder
        
        embedder = get_embedder()
        return {
            "embeddings_available": embedder.embeddings_available,
            "model_name": embedder.model_name,
            "fallback_mode": not embedder.embeddings_available,
        }
    
    except Exception as e:
        return {
            "embeddings_available": False,
            "model_name": None,
            "fallback_mode": True,
            "error": str(e),
        }


# ============================================================================
# Ontology Loading for Starchart
# ============================================================================

class OntologyInfo(BaseModel):
    """Information about a loaded ontology."""
    name: str
    property_count: int
    class_count: int
    prefixes: dict[str, str]


class OntologyProperty(BaseModel):
    """A property from the ontology."""
    uri: str
    label: str
    aliases: list[str] = []
    range: Optional[str] = None
    domain: Optional[str] = None
    description: Optional[str] = None


class OntologyClass(BaseModel):
    """A class from the ontology."""
    uri: str
    label: str
    description: Optional[str] = None
    properties: list[str] = []


class LoadOntologyResponse(BaseModel):
    """Response from loading an ontology."""
    success: bool
    info: OntologyInfo
    classes: list[OntologyClass]
    properties: list[OntologyProperty]
    prefixes: dict[str, str]


def _parse_ontology_ttl(content: str, base_namespace: str = None) -> dict:
    """
    Parse a TTL ontology file and extract classes, properties, and metadata.
    
    Extracts:
    - owl:Class / rdfs:Class definitions
    - owl:DatatypeProperty / owl:ObjectProperty definitions
    - rdfs:label, rdfs:comment, skos:altLabel for labels/aliases
    - rdfs:domain / rdfs:range
    - Prefixes
    """
    from rdf_starbase.formats.turtle import parse_turtle
    
    parsed = parse_turtle(content)
    
    # Build prefix lookup (reverse map: namespace -> prefix)
    prefix_map = parsed.prefixes.copy()
    reverse_prefixes = {v: k for k, v in prefix_map.items()}
    
    def compact_uri(uri: str) -> str:
        """Convert full URI to prefixed form."""
        for ns, prefix in reverse_prefixes.items():
            if uri.startswith(ns):
                return f"{prefix}:{uri[len(ns):]}"
        return uri
    
    def extract_label(uri: str) -> str:
        """Extract a human-readable label from URI."""
        if '#' in uri:
            return uri.split('#')[-1]
        elif '/' in uri:
            return uri.split('/')[-1]
        return uri
    
    # Group triples by subject
    by_subject: dict[str, dict] = {}
    for triple in parsed.triples:
        subj = triple.subject
        if subj not in by_subject:
            by_subject[subj] = {'predicates': {}}
        
        pred = triple.predicate
        obj = triple.object
        
        if pred not in by_subject[subj]['predicates']:
            by_subject[subj]['predicates'][pred] = []
        by_subject[subj]['predicates'][pred].append(obj)
    
    # Detect classes and properties
    rdf_type = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
    rdfs_label = 'http://www.w3.org/2000/01/rdf-schema#label'
    rdfs_comment = 'http://www.w3.org/2000/01/rdf-schema#comment'
    rdfs_domain = 'http://www.w3.org/2000/01/rdf-schema#domain'
    rdfs_range = 'http://www.w3.org/2000/01/rdf-schema#range'
    skos_altLabel = 'http://www.w3.org/2004/02/skos/core#altLabel'
    
    owl_class = 'http://www.w3.org/2002/07/owl#Class'
    rdfs_class = 'http://www.w3.org/2000/01/rdf-schema#Class'
    owl_datatype_prop = 'http://www.w3.org/2002/07/owl#DatatypeProperty'
    owl_object_prop = 'http://www.w3.org/2002/07/owl#ObjectProperty'
    rdf_property = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#Property'
    
    classes = []
    properties = []
    
    for subj, data in by_subject.items():
        preds = data['predicates']
        types = preds.get(rdf_type, [])
        
        # Get labels
        labels = preds.get(rdfs_label, [])
        label = labels[0].strip('"').split('@')[0].strip('"') if labels else extract_label(subj)
        
        # Get description
        comments = preds.get(rdfs_comment, [])
        description = comments[0].strip('"').split('@')[0].strip('"') if comments else None
        
        # Get aliases from skos:altLabel
        raw_aliases = preds.get(skos_altLabel, [])
        aliases = [a.strip('"').split('@')[0].strip('"') for a in raw_aliases]
        
        # Check if it's a class
        if owl_class in types or rdfs_class in types:
            classes.append({
                'uri': compact_uri(subj),
                'label': label,
                'description': description,
                'properties': [],  # Will be filled based on domain
            })
        
        # Check if it's a property
        elif owl_datatype_prop in types or owl_object_prop in types or rdf_property in types:
            domains = preds.get(rdfs_domain, [])
            ranges = preds.get(rdfs_range, [])
            
            properties.append({
                'uri': compact_uri(subj),
                'label': label,
                'aliases': aliases,
                'description': description,
                'domain': compact_uri(domains[0]) if domains else None,
                'range': compact_uri(ranges[0]) if ranges else 'xsd:string',
            })
    
    # Associate properties with classes by domain
    for prop in properties:
        domain = prop.get('domain')
        if domain:
            for cls in classes:
                if cls['uri'] == domain:
                    cls['properties'].append(prop['uri'])
    
    return {
        'classes': classes,
        'properties': properties,
        'prefixes': prefix_map,
    }


@router.post("/ontology/load", response_model=LoadOntologyResponse)
async def load_ontology(
    file: UploadFile = File(..., description="Ontology file (TTL/Turtle format)"),
):
    """
    Load an ontology file and extract classes/properties for mapping.
    
    Parses a Turtle/TTL file and extracts:
    - Classes (owl:Class, rdfs:Class)
    - Properties (owl:DatatypeProperty, owl:ObjectProperty)
    - Labels (rdfs:label)
    - Aliases (skos:altLabel) - important for column matching
    - Descriptions (rdfs:comment)
    - Domain/Range information
    
    Returns structured data ready for the Starchart mapping UI.
    """
    try:
        # Read the file
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Parse the ontology
        result = _parse_ontology_ttl(content_str)
        
        return LoadOntologyResponse(
            success=True,
            info=OntologyInfo(
                name=file.filename or 'ontology',
                property_count=len(result['properties']),
                class_count=len(result['classes']),
                prefixes=result['prefixes'],
            ),
            classes=[OntologyClass(**c) for c in result['classes']],
            properties=[OntologyProperty(**p) for p in result['properties']],
            prefixes=result['prefixes'],
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse ontology: {str(e)}"
        )


@router.get("/ontology/sample")
async def get_sample_ontologies():
    """
    List available sample ontologies that can be loaded.
    """
    import os
    # Path: src/api/etl_api.py -> src/api -> src -> project_root
    sample_dir = Path(__file__).parent.parent.parent / "data" / "sample"
    
    ontologies = []
    if sample_dir.exists():
        for f in sample_dir.glob("*.ttl"):
            ontologies.append({
                "name": f.stem,
                "filename": f.name,
                "path": str(f),
            })
    
    return {
        "ontologies": ontologies,
        "sample_dir": str(sample_dir),
    }


@router.get("/ontology/sample/{name}")
async def load_sample_ontology(name: str):
    """
    Load a sample ontology by name.
    
    Use /ontology/sample to list available sample ontologies.
    """
    # Path: src/api/etl_api.py -> src/api -> src -> project_root
    sample_dir = Path(__file__).parent.parent.parent / "data" / "sample"
    
    # Try exact name or with .ttl extension
    file_path = sample_dir / name
    if not file_path.exists():
        file_path = sample_dir / f"{name}.ttl"
    if not file_path.exists():
        file_path = sample_dir / f"{name}_ontology.ttl"
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Sample ontology '{name}' not found. Use /ontology/sample to list available ontologies."
        )
    
    try:
        content = file_path.read_text(encoding='utf-8')
        result = _parse_ontology_ttl(content)
        
        return {
            "success": True,
            "info": {
                "name": file_path.stem,
                "property_count": len(result['properties']),
                "class_count": len(result['classes']),
                "prefixes": result['prefixes'],
            },
            "classes": result['classes'],
            "properties": result['properties'],
            "prefixes": result['prefixes'],
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse ontology: {str(e)}"
        )
