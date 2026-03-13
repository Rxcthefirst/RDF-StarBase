#!/usr/bin/env python3
"""Minimal test - load L2Data directly with bulk_load_file."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rdf_starbase import RepositoryManager

# Paths
data_dir = Path("/data/repositories")
if not data_dir.exists():
    data_dir = Path(__file__).parent / "data" / "repositories"

gleif_dir = Path("/data/import/gleif")
if not gleif_dir.exists():
    gleif_dir = Path(__file__).parent / "data" / "gleif-lei-data"

l2_file = gleif_dir / "L2Data.ttl"

print(f"Loading: {l2_file}")
print(f"Size: {l2_file.stat().st_size / 1048576:.1f} MB")

manager = RepositoryManager(str(data_dir))
store = manager.get_store("gleif")

print(f"\nBefore: {store.stats().get('active_assertions', 0):,} triples")

# Direct call to bulk_load_file - this already streams properly
count = store.bulk_load_file(
    str(l2_file),
    source=f"file:{l2_file.name}",
)

print(f"\nLoaded: {count:,} triples")
print(f"Saving repository...")

manager.save("gleif")

stats = store.stats()
print(f"After: {stats.get('active_assertions', 0):,} triples")
