"""Benchmark bulk_load_file on L1Data.ttl (157M triples, 9 GB)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rdf_starbase.store import TripleStore

FILE = os.path.join(os.path.dirname(__file__), 'data', 'gleif', 'L1Data.ttl')
if not os.path.exists(FILE):
    print(f"File not found: {FILE}")
    sys.exit(1)

file_mb = os.path.getsize(FILE) / (1024 * 1024)
print(f"File: {FILE}")
print(f"Size: {file_mb:.1f} MB")

store = TripleStore()
start = time.time()
last_t = [start]

def progress(n):
    now = time.time()
    elapsed = now - start
    rate = n / elapsed if elapsed > 0 else 0
    batch_rate = 2_000_000 / (now - last_t[0]) if (now - last_t[0]) > 0 else 0
    print(f"  {n:>13,} triples | {elapsed:7.1f}s | avg {rate:,.0f} t/s | batch {batch_rate:,.0f} t/s")
    last_t[0] = now

print(f"\nbulk_load_file starting...")
count = store.bulk_load_file(FILE, on_progress=progress)
elapsed = time.time() - start
rate = count / elapsed if elapsed > 0 else 0

print(f"\n{'='*60}")
print(f"  Triples loaded : {count:,}")
print(f"  Wall time      : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
print(f"  Throughput     : {rate:,.0f} triples/sec")
print(f"{'='*60}")

stats = store.stats()
print(f"  Total assertions: {stats.get('total_assertions', 'N/A')}")
