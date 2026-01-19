"""
Synthetic RDF★ Data Generator for Benchmarking.

Generates data according to benchmarks.md specifications:
- Tier A: Synthetic controlled (engineering signal)
- Tier B: Receipts per run (product reality)

Usage:
    python -m benchmarks.data_generator --tier A --entities 100000 --facts 1000000
    python -m benchmarks.data_generator --tier B --runs 1000 --facts-per-run 500
"""

import argparse
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Tuple, List, Optional
import json

import numpy as np


@dataclass
class TierAConfig:
    """Configuration for Tier A synthetic benchmark data."""
    n_entities: int = 100_000
    n_predicates: int = 200
    n_facts: int = 1_000_000
    metadata_rate: float = 0.30  # 30% of facts get metadata
    metadata_per_fact_min: int = 2
    metadata_per_fact_max: int = 5
    n_sources: int = 100
    n_processes: int = 50
    zipf_exponent: float = 1.5  # For predicate distribution
    seed: int = 42


@dataclass
class TierBConfig:
    """Configuration for Tier B receipts-per-run data."""
    n_runs: int = 1_000
    facts_per_run: int = 500
    n_entities: int = 100_000
    n_predicates: int = 100
    entities_per_run: int = 50  # Input artifacts per run
    seed: int = 42


class ZipfDistribution:
    """Zipf distribution for realistic predicate/entity selection."""
    
    def __init__(self, n: int, exponent: float = 1.5, seed: int = 42):
        self.n = n
        self.rng = np.random.default_rng(seed)
        # Precompute weights
        ranks = np.arange(1, n + 1)
        self.weights = 1.0 / (ranks ** exponent)
        self.weights /= self.weights.sum()
        self.cumulative = np.cumsum(self.weights)
    
    def sample(self, size: int = 1) -> np.ndarray:
        """Sample from Zipf distribution."""
        u = self.rng.random(size)
        return np.searchsorted(self.cumulative, u)


def generate_tier_a(config: TierAConfig) -> Iterator[dict]:
    """
    Generate Tier A synthetic benchmark data.
    
    Yields dictionaries with fact information for batch processing.
    """
    rng = np.random.default_rng(config.seed)
    
    # Create distributions
    pred_dist = ZipfDistribution(config.n_predicates, config.zipf_exponent, config.seed)
    entity_dist = ZipfDistribution(config.n_entities, 1.2, config.seed + 1)  # Slightly less skewed
    
    # Predicate URIs
    predicates = [f"http://example.org/pred_{i:03d}" for i in range(config.n_predicates)]
    
    # Metadata predicates
    meta_preds = {
        "derived": "http://www.w3.org/ns/prov#wasDerivedFrom",
        "time": "http://www.w3.org/ns/prov#generatedAtTime",
        "generated": "http://www.w3.org/ns/prov#wasGeneratedBy",
        "confidence": "http://example.org/confidence",
    }
    
    # Sources and processes
    sources = [f"http://example.org/Source_{i:03d}" for i in range(config.n_sources)]
    processes = [f"http://example.org/Process_{i:03d}" for i in range(config.n_processes)]
    
    # Base time for timestamps
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    
    facts_generated = 0
    metadata_generated = 0
    
    while facts_generated < config.n_facts:
        # Generate subject and object
        s_idx = entity_dist.sample(1)[0]
        o_idx = entity_dist.sample(1)[0]
        p_idx = pred_dist.sample(1)[0]
        
        subject = f"http://example.org/Entity_{s_idx:06d}"
        predicate = predicates[p_idx]
        obj = f"http://example.org/Entity_{o_idx:06d}"
        
        # Base fact
        yield {
            "type": "base",
            "s": subject,
            "p": predicate,
            "o": obj,
        }
        facts_generated += 1
        
        # Potentially add metadata
        if rng.random() < config.metadata_rate:
            n_meta = rng.integers(
                config.metadata_per_fact_min, 
                config.metadata_per_fact_max + 1
            )
            
            # Generate quoted triple reference
            qt_ref = {"s": subject, "p": predicate, "o": obj}
            
            # Select which metadata predicates to use
            meta_keys = list(meta_preds.keys())
            rng.shuffle(meta_keys)
            selected_meta = meta_keys[:n_meta]
            
            for meta_key in selected_meta:
                meta_pred = meta_preds[meta_key]
                
                if meta_key == "derived":
                    meta_obj = sources[rng.integers(0, len(sources))]
                elif meta_key == "generated":
                    meta_obj = processes[rng.integers(0, len(processes))]
                elif meta_key == "time":
                    delta = timedelta(seconds=int(rng.integers(0, 86400 * 365)))
                    ts = base_time + delta
                    meta_obj = ts.isoformat()
                elif meta_key == "confidence":
                    conf = round(rng.random() * 0.5 + 0.5, 3)  # 0.5 to 1.0
                    meta_obj = str(conf)
                else:
                    continue
                
                yield {
                    "type": "metadata",
                    "qt": qt_ref,
                    "p": meta_pred,
                    "o": meta_obj,
                }
                metadata_generated += 1
        
        if facts_generated % 100000 == 0:
            print(f"  Generated {facts_generated:,} base facts, {metadata_generated:,} metadata facts...")


def generate_tier_b(config: TierBConfig) -> Iterator[dict]:
    """
    Generate Tier B receipts-per-run data.
    
    Models ETL/inference runs as prov:Activity with produced statements.
    """
    rng = np.random.default_rng(config.seed)
    
    # Predicate URIs
    predicates = [f"http://example.org/pred_{i:03d}" for i in range(config.n_predicates)]
    
    # Metadata predicates
    meta_preds = {
        "generated_by": "http://www.w3.org/ns/prov#wasGeneratedBy",
        "time": "http://www.w3.org/ns/prov#generatedAtTime",
        "derived": "http://www.w3.org/ns/prov#wasDerivedFrom",
        "confidence": "http://example.org/confidence",
    }
    
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    
    total_facts = 0
    total_meta = 0
    
    for run_idx in range(config.n_runs):
        run_uri = f"http://example.org/Run_{run_idx:06d}"
        run_time = base_time + timedelta(hours=run_idx)
        
        # Select input entities for this run
        input_entities = [
            f"http://example.org/Entity_{rng.integers(0, config.n_entities):06d}"
            for _ in range(config.entities_per_run)
        ]
        
        # Yield run activity
        yield {
            "type": "activity",
            "uri": run_uri,
            "inputs": input_entities,
            "time": run_time.isoformat(),
        }
        
        # Generate facts for this run
        for fact_idx in range(config.facts_per_run):
            # Subject from inputs, object random
            s = input_entities[rng.integers(0, len(input_entities))]
            p = predicates[rng.integers(0, len(predicates))]
            o = f"http://example.org/Entity_{rng.integers(0, config.n_entities):06d}"
            
            # Base fact
            yield {
                "type": "base",
                "s": s,
                "p": p,
                "o": o,
            }
            total_facts += 1
            
            # All facts in a run get provenance metadata
            qt_ref = {"s": s, "p": p, "o": o}
            
            # wasGeneratedBy run
            yield {
                "type": "metadata",
                "qt": qt_ref,
                "p": meta_preds["generated_by"],
                "o": run_uri,
            }
            total_meta += 1
            
            # generatedAtTime
            fact_time = run_time + timedelta(seconds=fact_idx)
            yield {
                "type": "metadata",
                "qt": qt_ref,
                "p": meta_preds["time"],
                "o": fact_time.isoformat(),
            }
            total_meta += 1
            
            # wasDerivedFrom (random input entity)
            derived_from = input_entities[rng.integers(0, len(input_entities))]
            yield {
                "type": "metadata",
                "qt": qt_ref,
                "p": meta_preds["derived"],
                "o": derived_from,
            }
            total_meta += 1
            
            # Confidence (random)
            conf = round(rng.random() * 0.4 + 0.6, 3)  # 0.6 to 1.0
            yield {
                "type": "metadata",
                "qt": qt_ref,
                "p": meta_preds["confidence"],
                "o": str(conf),
            }
            total_meta += 1
        
        if (run_idx + 1) % 100 == 0:
            print(f"  Generated {run_idx + 1:,} runs, {total_facts:,} base facts, {total_meta:,} metadata facts...")


def save_config(config: dict, output_dir: Path, tier: str):
    """Save configuration to JSON."""
    config_path = output_dir / f"tier_{tier.lower()}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Saved config to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic RDF★ benchmark data")
    parser.add_argument("--tier", choices=["A", "B"], required=True, help="Data tier to generate")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/data"), help="Output directory")
    
    # Tier A options
    parser.add_argument("--entities", type=int, default=100_000, help="Number of entities (Tier A)")
    parser.add_argument("--predicates", type=int, default=200, help="Number of predicates")
    parser.add_argument("--facts", type=int, default=1_000_000, help="Number of base facts (Tier A)")
    parser.add_argument("--metadata-rate", type=float, default=0.30, help="Fraction of facts with metadata")
    
    # Tier B options
    parser.add_argument("--runs", type=int, default=1_000, help="Number of runs (Tier B)")
    parser.add_argument("--facts-per-run", type=int, default=500, help="Facts per run (Tier B)")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    if args.tier == "A":
        config = TierAConfig(
            n_entities=args.entities,
            n_predicates=args.predicates,
            n_facts=args.facts,
            metadata_rate=args.metadata_rate,
            seed=args.seed,
        )
        print(f"Generating Tier A data:")
        print(f"  Entities: {config.n_entities:,}")
        print(f"  Predicates: {config.n_predicates:,}")
        print(f"  Base facts: {config.n_facts:,}")
        print(f"  Metadata rate: {config.metadata_rate:.0%}")
        
        # Save config
        save_config(vars(config), args.output, "A")
        
        # Generate and yield
        generator = generate_tier_a(config)
        
    else:  # Tier B
        config = TierBConfig(
            n_runs=args.runs,
            facts_per_run=args.facts_per_run,
            n_entities=args.entities,
            n_predicates=args.predicates,
            seed=args.seed,
        )
        print(f"Generating Tier B data:")
        print(f"  Runs: {config.n_runs:,}")
        print(f"  Facts per run: {config.facts_per_run:,}")
        print(f"  Total base facts: {config.n_runs * config.facts_per_run:,}")
        
        save_config(vars(config), args.output, "B")
        generator = generate_tier_b(config)
    
    # Write to JSONL for streaming ingestion
    output_file = args.output / f"tier_{args.tier.lower()}_data.jsonl"
    
    start = time.perf_counter()
    count = 0
    
    with open(output_file, "w") as f:
        for item in generator:
            f.write(json.dumps(item) + "\n")
            count += 1
    
    elapsed = time.perf_counter() - start
    
    print(f"\nGenerated {count:,} items in {elapsed:.2f}s ({count/elapsed:,.0f} items/sec)")
    print(f"Output: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
