#!/usr/bin/env python3
"""
Bulk Load Benchmark for GLEIF and FIBO datasets.

Tests the performance of RDF-StarBase bulk loading with:
- GLEIF L1Data.ttl (8.7GB) - Legal Entity Identifiers
- GLEIF L2Data.ttl (781MB) - Relationships
- FIBO prod.ttl.zip (Financial Industry Business Ontology)

Usage:
    python benchmarks/bulk_load_benchmark.py --dataset gleif
    python benchmarks/bulk_load_benchmark.py --dataset fibo
    python benchmarks/bulk_load_benchmark.py --dataset all
"""

import argparse
import asyncio
import time
import json
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rdf_starbase import TripleStore, ProvenanceContext
from rdf_starbase.repositories import RepositoryManager
from api.bulk_load_service import BulkLoadService


def format_bytes(bytes: int) -> str:
    """Format bytes as human-readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def format_duration(seconds: float) -> str:
    """Format duration as human-readable."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes}m {secs:.0f}s"


class BenchmarkRunner:
    """Benchmark runner for bulk load operations."""
    
    def __init__(self, workspace_path: str = "./data/repositories"):
        self.workspace_path = Path(workspace_path)
        self.import_path = Path("./data/import")
        self.manager = RepositoryManager(workspace_path)
        self.service = BulkLoadService(
            repository_manager=self.manager,
            import_base_path=self.import_path,
            batch_size=100_000,
            max_workers=4,
        )
        self.results = []
    
    async def benchmark_dataset(
        self,
        dataset_name: str,
        files: list[str],
        provenance: dict,
    ) -> dict:
        """
        Benchmark loading a dataset.
        
        Args:
            dataset_name: Name for the repository
            files: List of files to load
            provenance: Provenance metadata
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*80}")
        print(f"BENCHMARK: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Create repository if it doesn't exist
        try:
            self.manager.create(
                name=dataset_name,
                description=f"Bulk load benchmark - {dataset_name}",
                tags=["benchmark"],
            )
            print(f"✓ Created repository: {dataset_name}")
        except ValueError:
            # Repository already exists
            print(f"✓ Using existing repository: {dataset_name}")
        
        # Calculate total file size
        total_size = 0
        for file in files:
            file_path = self.import_path / file
            if file_path.exists():
                total_size += file_path.stat().st_size
            else:
                print(f"⚠ Warning: File not found: {file_path}")
        
        print(f"Files: {len(files)}")
        print(f"Total size: {format_bytes(total_size)}")
        print(f"Provenance: {provenance}")
        
        # Submit bulk load job
        print(f"\n⏳ Submitting bulk load job...")
        start_time = time.time()
        
        job = self.service.submit_job(
            repository_name=dataset_name,
            files=files,
            batch_provenance=provenance,
        )
        
        print(f"✓ Job submitted: {job.job_id}")
        print(f"Status: {job.status.value}")
        
        # Monitor progress
        last_progress = 0
        last_update = time.time()
        
        while True:
            job = self.service.get_job_status(job.job_id)
            
            if job.status.value in ("completed", "failed", "cancelled"):
                break
            
            # Print progress updates
            current_time = time.time()
            if current_time - last_update >= 5.0:  # Update every 5 seconds
                progress = job.progress.to_dict()
                percent = progress['percent_complete']
                rate = progress['triples_per_second']
                loaded = progress['triples_loaded']
                skipped = progress['triples_skipped']
                
                print(f"  {percent:.1f}% | {loaded:,} triples | {rate:.0f} t/s | {skipped:,} skipped")
                
                last_update = current_time
                last_progress = percent
            
            await asyncio.sleep(1)
        
        # Final results
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"\n{'='*80}")
        print(f"RESULTS")
        print(f"{'='*80}")
        
        result = {
            "dataset": dataset_name,
            "status": job.status.value,
            "files": len(files),
            "total_size_bytes": total_size,
            "total_size": format_bytes(total_size),
            "elapsed_seconds": round(elapsed, 2),
            "elapsed": format_duration(elapsed),
            "triples_parsed": job.progress.triples_parsed,
            "triples_loaded": job.progress.triples_loaded,
            "triples_skipped": job.progress.triples_skipped,
            "triples_failed": job.progress.triples_failed,
            "triples_per_second": round(job.progress.triples_per_second, 2),
            "mb_per_second": round(total_size / (1024 * 1024) / elapsed, 2) if elapsed > 0 else 0,
            "error": job.error_message,
        }
        
        print(f"Status: {result['status']}")
        print(f"Duration: {result['elapsed']}")
        print(f"Triples parsed: {result['triples_parsed']:,}")
        print(f"Triples loaded: {result['triples_loaded']:,}")
        print(f"Triples skipped: {result['triples_skipped']:,} (deduplication)")
        print(f"Triples failed: {result['triples_failed']:,}")
        print(f"Throughput: {result['triples_per_second']:,.0f} triples/second")
        print(f"I/O speed: {result['mb_per_second']:.2f} MB/second")
        
        if result['status'] == 'failed':
            print(f"Error: {result['error']}")
        
        self.results.append(result)
        return result
    
    async def benchmark_gleif(self):
        """Benchmark GLEIF dataset."""
        files = [
            "gleif/L1Data.ttl",      # 8.7GB - Level 1 (legal entities)
            "gleif/L2Data.ttl",      # 781MB - Level 2 (relationships)
            "gleif/RepExData.ttl",   # 1.7GB - Reporting exceptions
        ]
        
        provenance = {
            "source": "gleif:2026-02",
            "confidence": 1.0,
            "process": "bulk_import",
            "metadata": {
                "release_date": "2026-02-12",
                "dataset": "GLEIF Level 1 & 2",
                "url": "https://www.gleif.org/"
            }
        }
        
        return await self.benchmark_dataset("gleif", files, provenance)
    
    async def benchmark_fibo(self):
        """Benchmark FIBO dataset."""
        files = [
            "fibo/prod.ttl.zip",  # Financial Industry Business Ontology
        ]
        
        provenance = {
            "source": "fibo:prod",
            "confidence": 1.0,
            "process": "bulk_import",
            "metadata": {
                "dataset": "FIBO Production",
                "url": "https://spec.edmcouncil.org/fibo/"
            }
        }
        
        return await self.benchmark_dataset("fibo", files, provenance)
    
    def save_results(self, output_file: str = "benchmarks/bulk_load_results.json"):
        """Save results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                "benchmark_date": datetime.now().isoformat(),
                "results": self.results,
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="Bulk load benchmark for GLEIF and FIBO")
    parser.add_argument(
        "--dataset",
        choices=["gleif", "fibo", "all"],
        required=True,
        help="Dataset to benchmark"
    )
    parser.add_argument(
        "--output",
        default="benchmarks/bulk_load_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner()
    
    print(f"\n{'='*80}")
    print("RDF-STARBASE BULK LOAD BENCHMARK")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    if args.dataset == "gleif" or args.dataset == "all":
        await runner.benchmark_gleif()
    
    if args.dataset == "fibo" or args.dataset == "all":
        await runner.benchmark_fibo()
    
    # Save results
    runner.save_results(args.output)
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
