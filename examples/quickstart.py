"""
RDF-StarBase Quickstart Example

This demonstrates the core capabilities of RDF-StarBase:
- Adding triples with provenance
- Querying with filters
- Detecting competing claims
- Viewing provenance timelines
"""

from datetime import datetime, timedelta, timezone
from rdf_starbase import TripleStore, ProvenanceContext


def main():
    print("🌟 RDF-StarBase Quickstart\n")
    
    # Create a new triple store
    store = TripleStore()
    print("✓ Created empty triple store\n")
    
    # Example 1: Add triples with provenance
    print("=" * 60)
    print("Example 1: Adding Triples with Provenance")
    print("=" * 60)
    
    crm_prov = ProvenanceContext(
        source="CRM_System",
        confidence=0.85,
        process="nightly_sync"
    )
    
    customer = "http://example.org/customer/12345"
    
    store.add_triple(
        subject=customer,
        predicate="http://xmlns.com/foaf/0.1/name",
        obj="Alice Johnson",
        provenance=crm_prov
    )
    
    store.add_triple(
        subject=customer,
        predicate="http://example.org/age",
        obj=34,
        provenance=crm_prov
    )
    
    print(f"Added 2 triples about customer from CRM")
    print(f"Store now has {len(store)} assertions\n")
    
    # Example 2: Query triples
    print("=" * 60)
    print("Example 2: Querying Triples")
    print("=" * 60)
    
    results = store.get_triples(subject=customer)
    print(f"Found {len(results)} triples about {customer}:")
    print(results[["predicate", "object", "source", "confidence"]])
    print()
    
    # Example 3: Competing claims
    print("=" * 60)
    print("Example 3: Competing Claims (The Key Feature!)")
    print("=" * 60)
    
    # Data Lake has different age
    lake_prov = ProvenanceContext(
        source="DataLake",
        confidence=0.92,
        process="ml_enrichment",
        timestamp=datetime.now(timezone.utc) + timedelta(hours=1)  # More recent
    )
    
    store.add_triple(
        subject=customer,
        predicate="http://example.org/age",
        obj=36,
        provenance=lake_prov
    )
    
    print("Added conflicting age from DataLake")
    print("CRM says: age = 34 (confidence: 0.85)")
    print("DataLake says: age = 36 (confidence: 0.92)")
    print()
    
    # Get competing claims
    claims = store.get_competing_claims(
        subject=customer,
        predicate="http://example.org/age"
    )
    
    print("Competing claims (sorted by confidence & recency):")
    print(claims[["object", "source", "confidence", "timestamp"]])
    print()
    print("👆 This is unique to RDF-StarBase!")
    print("   Traditional databases would just overwrite the value.")
    print("   Here we see BOTH assertions and can choose which to trust.\n")
    
    # Example 4: Provenance timeline
    print("=" * 60)
    print("Example 4: Provenance Timeline")
    print("=" * 60)
    
    # Add status changes over time
    doc = "http://example.org/document/contract_2026_001"
    base_time = datetime(2026, 1, 10, tzinfo=timezone.utc)
    
    statuses = [
        ("draft", "author", 0, 1.0),
        ("review", "workflow", 2, 1.0),
        ("approved", "manager", 5, 1.0),
    ]
    
    for status, source, days_offset, conf in statuses:
        prov = ProvenanceContext(
            source=source,
            timestamp=base_time + timedelta(days=days_offset),
            confidence=conf
        )
        store.add_triple(
            subject=doc,
            predicate="http://example.org/status",
            obj=status,
            provenance=prov
        )
    
    timeline = store.get_provenance_timeline(
        subject=doc,
        predicate="http://example.org/status"
    )
    
    print(f"Timeline for document status:")
    print(timeline[["timestamp", "object", "source"]])
    print()
    print("👆 Full audit trail of how the status evolved!\n")
    
    # Example 5: Confidence filtering
    print("=" * 60)
    print("Example 5: Trust Lens (Filter by Confidence)")
    print("=" * 60)
    
    # Add some low-confidence triples
    uncertain_prov = ProvenanceContext(
        source="ML_Model_v0.1",
        confidence=0.45,
        process="beta_inference"
    )
    
    store.add_triple(
        subject=customer,
        predicate="http://example.org/predictedChurn",
        obj=True,
        provenance=uncertain_prov
    )
    
    print("Added low-confidence prediction (confidence: 0.45)")
    print()
    
    # Filter by confidence
    print("All assertions:")
    all_results = store.get_triples(subject=customer, min_confidence=0.0)
    print(f"  Count: {len(all_results)}")
    
    print("\nHigh-confidence only (>= 0.8):")
    high_conf_results = store.get_triples(subject=customer, min_confidence=0.8)
    print(f"  Count: {len(high_conf_results)}")
    print(high_conf_results[["predicate", "object", "confidence"]])
    print()
    print("👆 Same graph, different 'trust lens'!\n")
    
    # Final stats
    print("=" * 60)
    print("Store Statistics")
    print("=" * 60)
    print(store.stats())
    print()
    
    # Save to disk
    print("=" * 60)
    print("Persistence")
    print("=" * 60)
    store.save("data/quickstart.parquet")
    print("✓ Saved to data/quickstart.parquet")
    print("  (Using Parquet format for blazingly fast queries)")
    
    # Load it back
    loaded_store = TripleStore.load("data/quickstart.parquet")
    print(f"✓ Loaded store with {len(loaded_store)} assertions")
    print()
    
    print("=" * 60)
    print("🎉 Quickstart Complete!")
    print("=" * 60)
    print()
    print("What makes RDF-StarBase different:")
    print("  • Provenance is first-class, not metadata")
    print("  • Competing claims are preserved, not overwritten")
    print("  • Trust is computable via confidence scores")
    print("  • Blazingly fast thanks to Polars + Rust")
    print()
    print("Next: Check out examples/competing_claims.py")


if __name__ == "__main__":
    main()
