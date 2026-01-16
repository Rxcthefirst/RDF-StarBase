"""
RDF-StarBase Example: Assertion Registry

This example shows how to use the Assertion Registry to track
data sources, manage sync runs, and create auditable pipelines.

This is essential for data governance and lineage tracking.
"""

from datetime import datetime, timezone
from rdf_starbase import (
    TripleStore,
    AssertionRegistry,
    SourceType,
    SourceStatus,
    execute_sparql,
)


def main():
    print("📋 RDF-StarBase: Assertion Registry Demo\n")
    
    # Create registry and store
    registry = AssertionRegistry()
    store = TripleStore()
    
    print("=" * 60)
    print("Step 1: Register Data Sources")
    print("=" * 60)
    
    # Register various data sources
    salesforce = registry.register_source(
        name="Salesforce CRM",
        source_type=SourceType.API,
        uri="https://api.salesforce.com/v52",
        description="Customer relationship management system",
        owner="sales-team",
        sync_frequency="hourly",
        tags=["customers", "primary", "verified"]
    )
    print(f"✓ Registered: {salesforce.name} ({salesforce.source_type.value})")
    
    data_warehouse = registry.register_source(
        name="Snowflake DW",
        source_type=SourceType.DATASET,
        uri="snowflake://account.region/db/schema",
        description="Analytics data warehouse",
        owner="data-engineering",
        sync_frequency="daily",
        tags=["analytics", "aggregated"]
    )
    print(f"✓ Registered: {data_warehouse.name} ({data_warehouse.source_type.value})")
    
    customer_mapping = registry.register_source(
        name="Customer Schema Mapping",
        source_type=SourceType.MAPPING,
        description="Maps CRM fields to canonical customer schema",
        owner="data-architecture",
        schema_uri="http://schema.example.org/customer/v2",
        tags=["customers", "schema"]
    )
    print(f"✓ Registered: {customer_mapping.name} ({customer_mapping.source_type.value})")
    
    ml_pipeline = registry.register_source(
        name="Customer Segmentation Model",
        source_type=SourceType.PROCESS,
        description="ML model that predicts customer segments",
        owner="data-science",
        sync_frequency="weekly",
        config={"model_version": "2.1.0", "framework": "scikit-learn"},
        tags=["ml", "predictions", "segments"]
    )
    print(f"✓ Registered: {ml_pipeline.name} ({ml_pipeline.source_type.value})")
    
    # Show registry stats
    stats = registry.get_stats()
    print(f"\n📊 Registry now has {stats['total_sources']} sources:")
    for src_type, count in stats['sources_by_type'].items():
        print(f"   - {src_type}: {count}")
    
    print("\n" + "=" * 60)
    print("Step 2: Simulate Data Sync Pipeline")
    print("=" * 60)
    
    # Simulate syncing from Salesforce
    print(f"\n🔄 Starting sync from {salesforce.name}...")
    sf_run = registry.start_sync(salesforce.id, metadata={"batch_id": "2026-01-15-001"})
    print(f"   Run ID: {sf_run.id}")
    
    # Get provenance context from registry
    sf_prov = registry.create_provenance_context(
        salesforce.id,
        confidence=0.95,
        process="api_sync"
    )
    
    # Add some customer data
    customers = [
        ("customer/1001", "Alice Johnson", "Enterprise"),
        ("customer/1002", "Bob Smith", "SMB"),
        ("customer/1003", "Carol White", "Enterprise"),
    ]
    
    for cid, name, segment in customers:
        store.add_triple(
            f"http://example.org/{cid}",
            "http://xmlns.com/foaf/0.1/name",
            name,
            sf_prov
        )
        store.add_triple(
            f"http://example.org/{cid}",
            "http://example.org/segment",
            segment,
            sf_prov
        )
    
    # Complete the sync
    registry.complete_sync(
        sf_run.id,
        assertions_created=len(customers) * 2,
        status="success"
    )
    print(f"   ✓ Sync complete: {len(customers) * 2} assertions created")
    
    # Now sync ML predictions
    print(f"\n🤖 Starting sync from {ml_pipeline.name}...")
    ml_run = registry.start_sync(ml_pipeline.id, metadata={"model_version": "2.1.0"})
    
    ml_prov = registry.create_provenance_context(
        ml_pipeline.id,
        confidence=0.75,  # Lower confidence for predictions
        process="segment_prediction"
    )
    
    # ML model predicts different segment for Bob
    store.add_triple(
        "http://example.org/customer/1002",
        "http://example.org/predicted_segment",
        "Mid-Market",  # ML thinks Bob should be Mid-Market
        ml_prov
    )
    
    registry.complete_sync(ml_run.id, assertions_created=1, status="success")
    print(f"   ✓ Sync complete: 1 prediction added")
    
    print("\n" + "=" * 60)
    print("Step 3: Query with Source Awareness")
    print("=" * 60)
    
    # Show all assertions
    print("\n📜 All assertions in store:")
    all_triples = store.get_triples()
    print(all_triples[["subject", "predicate", "object", "source", "confidence"]])
    
    # Filter by source
    print(f"\n📜 Assertions from {salesforce.name} only:")
    sf_triples = store.get_triples(source="Salesforce CRM")
    print(sf_triples[["subject", "predicate", "object", "confidence"]])
    
    # Filter by confidence
    print("\n📜 High-confidence assertions only (>= 0.9):")
    high_conf = store.get_triples(min_confidence=0.9)
    print(high_conf[["subject", "predicate", "object", "source", "confidence"]])
    
    print("\n" + "=" * 60)
    print("Step 4: Source Management")
    print("=" * 60)
    
    # Query sources by type
    print("\n🔎 Finding all API sources:")
    apis = registry.get_sources(source_type=SourceType.API)
    for api in apis:
        print(f"   - {api.name}: {api.uri}")
    
    # Query sources by tag
    print("\n🔎 Finding sources tagged 'customers':")
    customer_sources = registry.get_sources(tag="customers")
    for src in customer_sources:
        print(f"   - {src.name} ({src.source_type.value})")
    
    # Check sync history
    print(f"\n📊 Sync history for {salesforce.name}:")
    history = registry.get_sync_history(salesforce.id)
    for row in history.iter_rows(named=True):
        print(f"   - {row['started_at']}: {row['status']} ({row['assertions_created']} created)")
    
    # Deprecate old source
    print(f"\n⚠️  Deprecating {customer_mapping.name} (replaced by v3)")
    registry.deprecate_source(customer_mapping.id)
    updated = registry.get_source(customer_mapping.id)
    print(f"   Status: {updated.status.value}")
    
    print("\n" + "=" * 60)
    print("Step 5: Governance Questions Answered")
    print("=" * 60)
    
    print("""
    With the Assertion Registry, you can answer:
    
    ❓ "Where did this data come from?"
       → Check the source field and look up in registry
    
    ❓ "When was this source last synced?"
       → registry.get_last_sync(source_id)
    
    ❓ "Which systems contribute customer data?"
       → registry.get_sources(tag="customers")
    
    ❓ "What's our sync success rate?"
       → registry.get_stats()['sync_success_rate']
    
    ❓ "Who owns this data pipeline?"
       → source.owner from the registry
    """)
    
    # Final stats
    print("=" * 60)
    print("Final Registry Statistics")
    print("=" * 60)
    final_stats = registry.get_stats()
    print(f"""
    Total Sources: {final_stats['total_sources']}
    Active Sources: {final_stats['sources_by_status'].get('active', 0)}
    Deprecated Sources: {final_stats['sources_by_status'].get('deprecated', 0)}
    Total Sync Runs: {final_stats['total_sync_runs']}
    Sync Success Rate: {final_stats['sync_success_rate']:.0%}
    """)
    
    print("=" * 60)
    print("🎉 Registry Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
