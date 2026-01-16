"""
RDF-StarBase Example: Competing Claims

This example demonstrates how RDF-StarBase handles conflicting data
from multiple sources - a key differentiator from traditional databases.

Scenario: Customer age is reported differently by three systems.
Traditional DBs would overwrite, losing data.
RDF-StarBase keeps ALL claims with full provenance.
"""

from datetime import datetime, timedelta, timezone
from rdf_starbase import (
    TripleStore,
    ProvenanceContext,
    AssertionRegistry,
    SourceType,
    execute_sparql,
)


def main():
    print("⚖️  RDF-StarBase: Competing Claims Demo\n")
    print("=" * 60)
    print("Scenario: Three systems report different customer ages")
    print("=" * 60)
    
    # Create store and registry
    store = TripleStore()
    registry = AssertionRegistry()
    
    # Register our data sources
    crm = registry.register_source(
        name="CRM",
        source_type=SourceType.API,
        owner="sales-team",
        tags=["customer-data", "primary"]
    )
    
    data_lake = registry.register_source(
        name="DataLake",
        source_type=SourceType.DATASET,
        owner="data-engineering",
        tags=["customer-data", "analytics"]
    )
    
    ml_enrichment = registry.register_source(
        name="ML-Enrichment",
        source_type=SourceType.PROCESS,
        owner="data-science",
        tags=["customer-data", "predicted"]
    )
    
    print(f"\n📋 Registered {registry.get_stats()['total_sources']} data sources")
    
    # The customer we're tracking
    customer = "http://example.org/customer/42"
    age_predicate = "http://example.org/schema/age"
    
    # Each source provides different data
    base_time = datetime.now(timezone.utc)
    
    # CRM says 32 (manual entry, high confidence)
    crm_prov = registry.create_provenance_context(crm.id, confidence=0.90)
    store.add_triple(customer, age_predicate, 32, crm_prov)
    print(f"\n🏢 CRM says: age = 32 (confidence: 0.90)")
    
    # Data Lake says 34 (from legacy migration, medium confidence)
    lake_prov = ProvenanceContext(
        source="DataLake",
        confidence=0.75,
        process="legacy_migration",
        timestamp=base_time - timedelta(days=30),  # Older data
    )
    store.add_triple(customer, age_predicate, 34, lake_prov)
    print(f"📊 DataLake says: age = 34 (confidence: 0.75, 30 days old)")
    
    # ML model predicts 33 (recent, but lower confidence)
    ml_prov = ProvenanceContext(
        source="ML-Enrichment",
        confidence=0.65,
        process="age_prediction_model_v2",
        timestamp=base_time,  # Most recent
    )
    store.add_triple(customer, age_predicate, 33, ml_prov)
    print(f"🤖 ML-Enrichment says: age = 33 (confidence: 0.65, predicted)")
    
    # Now let's explore what a traditional DB vs RDF-StarBase shows
    print("\n" + "=" * 60)
    print("Traditional DB Behavior (BAD)")
    print("=" * 60)
    print("❌ Would store: age = 33 (last write wins)")
    print("❌ Lost: CRM's 32 and DataLake's 34")
    print("❌ No audit trail of disagreement")
    
    print("\n" + "=" * 60)
    print("RDF-StarBase Behavior (GOOD)")
    print("=" * 60)
    
    # Get competing claims
    claims = store.get_competing_claims(customer, age_predicate)
    print("\n✅ ALL claims preserved with provenance:\n")
    print(claims[["object", "source", "confidence", "timestamp"]])
    
    # Show different "trust lenses"
    print("\n" + "-" * 60)
    print("🔍 Trust Lens: Highest Confidence")
    print("-" * 60)
    best = claims.filter(
        claims["confidence"] == claims["confidence"].max()
    )
    print(f"Most trusted value: age = {best['object'][0]}")
    print(f"From: {best['source'][0]} (confidence: {best['confidence'][0]})")
    
    print("\n" + "-" * 60)
    print("🔍 Trust Lens: Most Recent")
    print("-" * 60)
    recent = claims.sort("timestamp", descending=True).head(1)
    print(f"Most recent value: age = {recent['object'][0]}")
    print(f"From: {recent['source'][0]} (timestamp: {recent['timestamp'][0]})")
    
    print("\n" + "-" * 60)
    print("🔍 Trust Lens: Exclude Predictions")
    print("-" * 60)
    non_ml = store.get_triples(
        subject=customer,
        predicate=age_predicate,
    ).filter(~store._df["process"].str.contains("prediction"))
    print(f"Values from verified sources only:")
    print(non_ml[["object", "source", "confidence"]])
    
    # SPARQL query example
    print("\n" + "=" * 60)
    print("SPARQL Query: Find all customer attributes")
    print("=" * 60)
    
    # Add more customer data
    name_prov = registry.create_provenance_context(crm.id, confidence=0.99)
    store.add_triple(customer, "http://xmlns.com/foaf/0.1/name", "John Smith", name_prov)
    
    results = execute_sparql(store, """
        SELECT ?p ?o WHERE {
            <http://example.org/customer/42> ?p ?o
        }
    """)
    print("\nAll assertions about customer/42:")
    print(results)
    
    # Business application
    print("\n" + "=" * 60)
    print("📈 Business Application")
    print("=" * 60)
    print("""
    With RDF-StarBase, you can:
    
    1. Show customers a "data quality" score based on agreement
    2. Alert when sources disagree beyond a threshold
    3. Create resolution workflows for conflicting data
    4. Let different teams use different trust policies
    5. Maintain full audit trails for compliance
    
    "The CRM team trusts CRM data."
    "The analytics team trusts the Data Lake."
    "The ML team trusts predictions for some use cases."
    
    ALL are valid perspectives on the same knowledge graph!
    """)
    
    print("=" * 60)
    print("🎉 Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
