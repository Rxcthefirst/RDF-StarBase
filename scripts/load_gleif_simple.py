#!/usr/bin/env python3
"""
Load GLEIF data using the repository's built-in bulk loader.
"""
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rdf_starbase import RepositoryManager

def load_file(repo_manager, repo_name, file_path):
    """Load an RDF file using the built-in bulk loader."""
    print(f"\nLoading {file_path.name} ({file_path.stat().st_size / 1048576:.1f} MB)...")
    
    store = repo_manager.get_store(repo_name)
    
    start = time.time()
    try:
        # Use built-in bulk loader which handles large files properly
        count = store.bulk_load_file(
            str(file_path),
            source=f"file:{file_path.name}",
        )
        
        elapsed = time.time() - start
        print(f"  ✓ Loaded {count:,} triples in {elapsed:.1f}s ({count/elapsed:,.0f} triples/sec)")
        
        # Save to disk
        print(f"  Saving repository...")
        repo_manager.save(repo_name)
        
        return count
        
    except Exception as e:
        print(f"  ✗ Error loading {file_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return 0


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
        # Already loaded:
        # "RegistrationAuthorityData.ttl",  # 672KB - loaded
        # "EntityLegalFormData.ttl",  # 1.3MB - loaded
        # "BICData.ttl",  # 4.7MB - loaded
        
        # Load now:
        "L2Data.ttl",  # 781MB
        "L1Data.ttl",  # 8.7GB
    ]
    
    # Show current stats
    stats = manager.get_store("gleif").stats()
    print(f"\nCurrent repository stats:")
    print(f"  Triples: {stats.get('active_assertions', 0):,}")
    print(f"  Subjects: {stats.get('unique_subjects', 0):,}")
    print(f"  Predicates: {stats.get('unique_predicates', 0):,}")
    
    overall_start = time.time()
    total_triples = 0
    
    for filename in files_to_load:
        file_path = gleif_dir / filename
        if not file_path.exists():
            print(f"\n⚠ Skipping {filename} (not found)")
            continue
        
        loaded = load_file(manager, "gleif", file_path)
        total_triples += loaded
        
        # Show updated stats
        stats = manager.get_store("gleif").stats()
        print(f"  Repository now has {stats.get('active_assertions', 0):,} triples total")
    
    overall_elapsed = time.time() - overall_start
    print(f"\n✓ Complete!")
    print(f"  New triples loaded: {total_triples:,}")
    print(f"  Total time: {overall_elapsed/60:.1f} minutes")
    if total_triples > 0:
        print(f"  Average throughput: {total_triples/overall_elapsed:,.0f} triples/sec")
