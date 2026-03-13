"""
Bulk Load Service for RDF-StarBase.

GraphDB-style bulk loading with:
- Async job queue with progress tracking
- Streaming ingestion for multi-GB files
- Compression support (.gz, .bz2, .zip)
- Batch provenance metadata
- Transaction support with rollback
- Smart deduplication (same triple+metadata = skip, different metadata = append)
"""

import asyncio
import gzip
import bz2
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
from uuid import uuid4
import io

from rdf_starbase import TripleStore, ProvenanceContext
from rdf_starbase.storage.transactions import Transaction


class BulkLoadStatus(str, Enum):
    """Status of a bulk load job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BulkLoadProgress:
    """Real-time progress metrics for a bulk load job."""
    total_bytes: int = 0
    bytes_processed: int = 0
    triples_parsed: int = 0
    triples_loaded: int = 0
    triples_skipped: int = 0  # Duplicates
    triples_failed: int = 0
    files_processed: int = 0
    files_total: int = 0
    current_file: str = ""
    percent_complete: float = 0.0
    elapsed_seconds: float = 0.0
    triples_per_second: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "total_bytes": self.total_bytes,
            "bytes_processed": self.bytes_processed,
            "triples_parsed": self.triples_parsed,
            "triples_loaded": self.triples_loaded,
            "triples_skipped": self.triples_skipped,
            "triples_failed": self.triples_failed,
            "files_processed": self.files_processed,
            "files_total": self.files_total,
            "current_file": self.current_file,
            "percent_complete": round(self.percent_complete, 2),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "triples_per_second": round(self.triples_per_second, 2),
        }


@dataclass
class BulkLoadJob:
    """Metadata for a bulk load job."""
    job_id: str
    repository_name: str
    files: List[str]
    status: BulkLoadStatus
    progress: BulkLoadProgress = field(default_factory=BulkLoadProgress)
    
    # Provenance metadata (batch defaults)
    batch_provenance: Optional[Dict[str, Any]] = None
    
    # Results
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Transaction ID for rollback
    transaction_id: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "repository_name": self.repository_name,
            "files": self.files,
            "status": self.status.value,
            "progress": self.progress.to_dict(),
            "batch_provenance": self.batch_provenance,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
        }


class BulkLoadService:
    """
    Service for managing bulk load jobs.
    
    Features:
    - GraphDB-style file-based bulk loading
    - Async job queue with progress tracking
    - Streaming for multi-GB files (GLEIF L1Data.ttl = 8.7GB)
    - Compression support
    - Batch provenance (file-level metadata)
    - Transaction support
    - Smart deduplication
    
    Usage:
        service = BulkLoadService(repository_manager, import_base_path)
        
        # Submit bulk load job
        job = service.submit_job(
            repository_name="gleif",
            file_patterns=["data/import/gleif/*.ttl"],
            batch_provenance={
                "source": "gleif:L1Data",
                "confidence": 1.0,
                "process": "bulk_import"
            }
        )
        
        # Check status
        status = service.get_job_status(job.job_id)
    """
    
    def __init__(
        self,
        repository_manager,
        import_base_path: Path,
        batch_size: int = 100_000,
        max_workers: int = 4,
    ):
        """
        Initialize the bulk load service.
        
        Args:
            repository_manager: RepositoryManager instance
            import_base_path: Base path for import files
            batch_size: Triples per batch (tune for memory)
            max_workers: Concurrent processing threads
        """
        self.repository_manager = repository_manager
        self.import_base_path = Path(import_base_path)
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Job tracking
        self._jobs: Dict[str, BulkLoadJob] = {}
        self._lock = asyncio.Lock()
        
        # Background task queue
        self._running_tasks: Dict[str, asyncio.Task] = {}
    
    def submit_job(
        self,
        repository_name: str,
        files: List[str],
        batch_provenance: Optional[Dict[str, Any]] = None,
    ) -> BulkLoadJob:
        """
        Submit a bulk load job for async processing.
        
        Args:
            repository_name: Target repository
            files: List of file paths relative to import_base_path
            batch_provenance: Default provenance for triples without metadata
                              (source, confidence, process, etc.)
        
        Returns:
            BulkLoadJob with job_id for tracking
        """
        # Resolve file paths
        resolved_files = []
        total_bytes = 0
        
        for file_path in files:
            abs_path = self.import_base_path / file_path
            if not abs_path.exists():
                raise FileNotFoundError(f"File not found: {abs_path}")
            resolved_files.append(str(abs_path))
            total_bytes += abs_path.stat().st_size
        
        # Create job
        job = BulkLoadJob(
            job_id=str(uuid4()),
            repository_name=repository_name,
            files=resolved_files,
            status=BulkLoadStatus.PENDING,
            batch_provenance=batch_provenance,
        )
        job.progress.total_bytes = total_bytes
        job.progress.files_total = len(resolved_files)
        
        self._jobs[job.job_id] = job
        
        # Start async processing
        task = asyncio.create_task(self._process_job(job))
        self._running_tasks[job.job_id] = task
        
        return job
    
    async def _process_job(self, job: BulkLoadJob) -> None:
        """Process a bulk load job asynchronously.
        
        Uses a lightweight bulk-load store that loads only the term
        dictionaries (needed for interning) but NOT the existing facts.
        New facts are appended to the on-disk Parquet via streaming
        concat, keeping memory proportional to *new* data only.
        """
        job.status = BulkLoadStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        start_time = asyncio.get_event_loop().time()

        try:
            # Lightweight store: TermDict + QtDict loaded, FactStore empty
            if hasattr(self.repository_manager, 'get_store_for_bulk_load'):
                store = self.repository_manager.get_store_for_bulk_load(job.repository_name)
                use_bulk_save = True
            else:
                store = self.repository_manager.get_store(job.repository_name)
                use_bulk_save = False

            # Begin transaction
            txn = store.begin_transaction()
            job.transaction_id = txn.txn_id if hasattr(txn, 'txn_id') else None

            try:
                # Process each file, skipping individual failures
                failed_files: list[str] = []
                for file_path in job.files:
                    try:
                        await self._process_file(job, store, file_path)
                    except Exception as file_err:
                        import logging
                        fname = Path(file_path).name
                        logging.getLogger(__name__).warning(
                            f"Skipping {fname}: {file_err}"
                        )
                        failed_files.append(f"{fname}: {file_err}")
                    job.progress.files_processed += 1

                # Commit transaction
                if hasattr(txn, 'commit'):
                    txn.commit()

                # Persist: append new facts to existing Parquet (constant memory)
                if use_bulk_save:
                    self.repository_manager.save_bulk(job.repository_name, store)
                else:
                    self.repository_manager.save(job.repository_name)

                # Success (with possible warnings)
                job.status = BulkLoadStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                if failed_files:
                    job.error_message = (
                        f"{len(failed_files)} file(s) skipped due to parse errors: "
                        + "; ".join(failed_files[:10])
                    )

            except Exception as e:
                # Rollback transaction
                if hasattr(txn, 'rollback'):
                    txn.rollback()
                raise
                
        except Exception as e:
            job.status = BulkLoadStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)
        
        finally:
            # Update timing stats
            elapsed = asyncio.get_event_loop().time() - start_time
            job.progress.elapsed_seconds = elapsed
            if elapsed > 0:
                job.progress.triples_per_second = job.progress.triples_loaded / elapsed
            job.progress.percent_complete = 100.0
            
            # Cleanup
            if job.job_id in self._running_tasks:
                del self._running_tasks[job.job_id]
    
    async def _process_file(
        self,
        job: BulkLoadJob,
        store: TripleStore,
        file_path: str,
    ) -> None:
        """
        Process a single file with streaming and compression support.
        
        Uses store.bulk_load_file() as the primary path — it streams directly
        from the file handle through Oxigraph with no string buffering, handling
        multi-GB files in constant memory.
        
        Falls back to string-chunk path only for formats bulk_load_file doesn't
        support (e.g. JSON-LD, TriG).
        """
        file_path_obj = Path(file_path)
        job.progress.current_file = file_path_obj.name
        file_size = file_path_obj.stat().st_size
        
        # Prepare provenance source
        batch_prov = job.batch_provenance or {}
        source = batch_prov.get('source', f'file:{file_path_obj.name}')
        
        # --- Primary path: store.bulk_load_file (streaming, memory-safe) ---
        if hasattr(store, 'bulk_load_file'):
            loop = asyncio.get_event_loop()
            
            def _progress(loaded: int):
                job.progress.triples_loaded = loaded
                if file_size > 0:
                    # Estimate percent from triples loaded vs expected
                    job.progress.percent_complete = min(
                        (job.progress.files_processed / max(job.progress.files_total, 1)) * 100,
                        99.0,
                    )
            
            try:
                count = await loop.run_in_executor(
                    None,
                    lambda: store.bulk_load_file(
                        file_path,
                        source=source,
                        on_progress=_progress,
                    ),
                )
                job.progress.triples_loaded += count
                job.progress.triples_parsed += count
                job.progress.bytes_processed += file_size
                return
            except Exception as e:
                # bulk_load_file doesn't support this format — fall through
                import logging
                logging.getLogger(__name__).info(
                    f"bulk_load_file not applicable for {file_path_obj.name}, "
                    f"falling back to chunk loader: {e}"
                )
        
        # --- Fallback: string-chunk path for other formats ---
        format_type = self._detect_format(file_path_obj)
        
        # Process file in chunks to avoid memory exhaustion
        file_size = file_path_obj.stat().st_size
        
        # For small files (<50MB), read all at once (faster)
        if file_size < 50 * 1024 * 1024:
            content = ""
            async for chunk in self._read_file_chunked(file_path_obj):
                content += chunk
            
            await self._parse_and_load_chunks(
                job=job,
                store=store,
                content=content,
                format_type=format_type,
                file_name=file_path_obj.name,
            )
        else:
            # Large files: process in chunks
            accumulated = ""
            chunk_num = 0
            
            async for chunk in self._read_file_chunked(file_path_obj, chunk_size=50 * 1024 * 1024):
                accumulated += chunk
                chunk_num += 1
                
                # Process complete statements (wait for blank line or reasonable buffer)
                if len(accumulated) > 50 * 1024 * 1024 or chunk_num > 1:
                    await self._parse_and_load_chunks(
                        job=job,
                        store=store,
                        content=accumulated,
                        format_type=format_type,
                        file_name=f"{file_path_obj.name}_chunk{chunk_num}",
                    )
                    accumulated = ""  # Clear for next chunk
    
    def _detect_format(self, file_path: Path) -> str:
        """Detect RDF format from file extension."""
        # Handle compression
        if file_path.suffix == '.gz':
            file_path = file_path.with_suffix('')
        elif file_path.suffix == '.bz2':
            file_path = file_path.with_suffix('')
        elif file_path.suffix == '.zip':
            file_path = file_path.with_suffix('')
        
        # Detect format
        ext = file_path.suffix.lower()
        format_map = {
            '.ttl': 'turtle',
            '.nt': 'ntriples',
            '.nq': 'nquads',
            '.trig': 'trig',
            '.n3': 'n3',
            '.jsonld': 'jsonld',
            '.rdf': 'rdfxml',
            '.xml': 'rdfxml',
        }
        return format_map.get(ext, 'turtle')
    
    async def _read_file_chunked(self, file_path: Path, chunk_size: int = 100 * 1024 * 1024):
        """Read file in chunks with compression support (streaming for large files).
        
        Args:
            file_path: Path to file
            chunk_size: Size of each chunk in bytes (default 100MB)
            
        Yields:
            Chunks of file content as strings
        """
        # Detect compression
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        elif file_path.suffix == '.bz2':
            with bz2.open(file_path, 'rt', encoding='utf-8') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        elif file_path.suffix == '.zip':
            with zipfile.ZipFile(file_path, 'r') as z:
                names = z.namelist()
                if not names:
                    raise ValueError(f"Empty zip file: {file_path}")
                with z.open(names[0]) as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk.decode('utf-8')
        else:
            # Plain text - chunk reading for large files
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
    
    async def _parse_and_load_chunks(
        self,
        job: BulkLoadJob,
        store: TripleStore,
        content: str,
        format_type: str,
        file_name: str,
    ) -> None:
        """
        Parse and load in chunks with batch provenance.
        
        Uses fast bulk_loader.py with Oxigraph acceleration.
        """
        from rdf_starbase.storage.bulk_loader import stream_load_string_with_dedup
        
        # Prepare batch provenance
        batch_prov = job.batch_provenance or {}
        source = batch_prov.get('source', f'file:{file_name}')
        confidence = batch_prov.get('confidence', 1.0)
        process = batch_prov.get('process', 'bulk_import')
        
        provenance = ProvenanceContext(
            source=source,
            confidence=confidence,
            process=process,
            metadata={
                "file_name": file_name,
                "import_time": datetime.now(timezone.utc).isoformat(),
                **batch_prov.get('metadata', {}),
            }
        )
        
        # Stream load with progress tracking
        async def progress_callback(loaded: int, skipped: int, failed: int):
            job.progress.triples_loaded += loaded
            job.progress.triples_skipped += skipped
            job.progress.triples_failed += failed
            
            # Update percent
            if job.progress.total_bytes > 0:
                job.progress.percent_complete = (
                    job.progress.bytes_processed / job.progress.total_bytes * 100
                )
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._load_sync,
            store,
            content,
            format_type,
            provenance,
            job,
        )
    
    def _load_sync(
        self,
        store: TripleStore,
        content: str,
        format_type: str,
        provenance: ProvenanceContext,
        job: BulkLoadJob,
    ) -> None:
        """Synchronous load (called from executor)."""
        # Use bulk_loader streaming with deduplication
        from rdf_starbase.storage.bulk_loader import stream_load_string_with_dedup
        
        result = stream_load_string_with_dedup(
            store=store,
            content=content,
            format_type=format_type,
            provenance=provenance,
            batch_size=self.batch_size,
        )
        
        # Update progress
        job.progress.triples_parsed += result.get('triples_parsed', 0)
        job.progress.triples_loaded += result.get('triples_loaded', 0)
        job.progress.triples_skipped += result.get('triples_skipped', 0)
        job.progress.bytes_processed += len(content.encode('utf-8'))
    
    def get_job_status(self, job_id: str) -> Optional[BulkLoadJob]:
        """Get status of a job."""
        return self._jobs.get(job_id)
    
    def list_jobs(self, repository_name: Optional[str] = None) -> List[BulkLoadJob]:
        """List all jobs, optionally filtered by repository."""
        jobs = list(self._jobs.values())
        if repository_name:
            jobs = [j for j in jobs if j.repository_name == repository_name]
        return jobs
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        if job.status not in (BulkLoadStatus.PENDING, BulkLoadStatus.RUNNING):
            return False
        
        # Cancel task
        task = self._running_tasks.get(job_id)
        if task:
            task.cancel()
        
        job.status = BulkLoadStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc)
        return True
