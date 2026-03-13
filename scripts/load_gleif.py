#!/usr/bin/env python3
"""
Load GLEIF data using the bulk loader with streaming support.
"""
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rdf_starbase import RepositoryManager, ProvenanceContext
from rdf_starbase.storage.bulk_loader import stream_load_string_with_dedup

def load_file_chunked(repo_manager, repo_name, file_path, chunk_size_mb=50):
    """Load a large RDF file in chunks to avoid memory exhaustion."""
    print(f"Loading {file_path.name} ({file_path.stat().st_size / 1048576:.1f} MB)...")
    
    store = repo_manager.get_store(repo_name)
    
    # Start transaction
    if hasattr(store, '_transaction_manager'):
        tx = store._transaction_manager.begin()
        try:
            # Read and process in chunks
            total_loaded = 0
            chunk_num = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(chunk_size_mb * 1024 * 1024)
                    if not chunk:
                        break
                    
                    chunk_num += 1
                    print(f"  Processing chunk {chunk_num}...")
                    
                    # Load chunk
                    provenance = ProvenanceContext(
                        source=f"file:{file_path.name}",
                        confidence=1.0
                    )
                    
                    result = stream_load_string_with_dedup(
                        store=store,
                        content=chunk,
                        format_type="turtle",
                        provenance=provenance,
                        batch_size=100000,
                    )
                    
                    loaded = result['triples_loaded']
                    skipped = result['triples_skipped']
                    total_loaded += loaded
                    
                    print(f"    Loaded: {loaded:,} | Skipped: {skipped:,} | Total: {total_loaded:,}")
            
            # Commit transaction
            tx.commit()
            print(f"  Transaction committed. Saving repository...")
            
            # Save to disk
            repo_manager.save(repo_name)
            print(f"  ✓ {file_path.name}: {total_loaded:,} triples loaded")
            
            return total_loaded
            
        except Exception as e:
            tx.rollback()
            print(f"  ✗ Error loading {file_path.name}: {e}")
            print(f"  Transaction rolled back")
            return 0
    else:
        # No transaction support - direct load
        total_loaded = 0
        chunk_num = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size_mb * 1024 * 1024)
                if not chunk:
                    break
                
                chunk_num += 1
                print(f"  Processing chunk {chunk_num}...")
                
                provenance = ProvenanceContext(
                    source=f"file:{file_path.name}",
                    confidence=1.0
                )
                
                result = stream_load_string_with_dedup(
                    store=store,
                    content=chunk,
                    format_type="turtle",
                    provenance=provenance,
                    batch_size=100000,
                )
                
                loaded = result['triples_loaded']
                skipped = result['triples_skipped']
                total_loaded += loaded
                
                print(f"    Loaded: {loaded:,} | Skipped: {skipped:,} | Total: {total_loaded:,}")
        
        # Save to disk
        repo_manager.save(repo_name)
        print(f"  ✓ {file_path.name}: {total_loaded:,} triples loaded")
        
        return total_loaded


if __name__ == "__main__":
    # Initialize repository manager
    data_dir = Path("/data/repositories")  # Docker path
    if not data_dir.exists():
        data_dir = Path(__file__).parent / "data" / "repositories"  # Local path
    
    print(f"Using data directory: {data_dir}")
    manager = RepositoryManager(str(data_dir))
    
    # GLEIF files to load
    gleif_dir = Path("/data/import/gleif")  # Docker path
    if not gleif_dir.exists():
        gleif_dir = Path(__file__).parent / "data" / "gleif-lei-data"  # Local path
    
    print(f"GLEIF data directory: {gleif_dir}")
    
    # List of files to load (in order: small to large)
    files_to_load = [
        "RegistrationAuthorityData.ttl",  # 672KB
        "EntityLegalFormData.ttl",  # 1.3MB
        "BICData.ttl",  # 4.7MB
        "L2Data.ttl",  # 781MB
        "L1Data.ttl",  # 8.7GB
    ]
    
    start_time = time.time()
    total_triples = 0
    
    for filename in files_to_load:
        file_path = gleif_dir / filename
        if not file_path.exists():
            print(f"⚠ Skipping {filename} (not found)")
            continue
        
        loaded = load_file_chunked(manager, "gleif", file_path, chunk_size_mb=50)
        total_triples += loaded
        
        # Show stats
        stats = manager.get_store("gleif").stats()
        print(f"  Repository stats: {stats.get('active_assertions', 0):,} triples total\n")
    
    elapsed = time.time() - start_time
    print(f"\n✓ Complete!")
    print(f"  Total triples loaded: {total_triples:,}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Average throughput: {total_triples/elapsed:,.0f} triples/sec")
