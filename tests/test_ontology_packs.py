"""Tests for ontology packs module."""

import pytest
from rdf_starbase.storage.ontology_packs import (
    OntologyPackType,
    PropertyTemplate,
    ClassTemplate,
    OntologyPack,
    OntologyPackManager,
    SchemaGuidance,
    create_prov_o_pack,
    create_dcat_pack,
    create_pav_pack,
    PROV,
    DCAT,
    PAV,
    XSD,
    RDF,
)


# ============================================================================
# PropertyTemplate Tests
# ============================================================================

class TestPropertyTemplate:
    """Tests for PropertyTemplate dataclass."""
    
    def test_create_property_template(self):
        """Test creating a property template."""
        prop = PropertyTemplate(
            uri="http://example.org/name",
            label="name",
            description="The name of the entity",
            range_type="literal",
            datatype=f"{XSD}string",
            required=True,
        )
        
        assert prop.uri == "http://example.org/name"
        assert prop.label == "name"
        assert prop.required is True
        assert prop.datatype == f"{XSD}string"
    
    def test_optional_fields(self):
        """Test optional fields have defaults."""
        prop = PropertyTemplate(
            uri="http://example.org/prop",
            label="prop",
            description="A property",
            range_type=f"{RDF}Resource",
        )
        
        assert prop.required is False
        assert prop.datatype is None
        assert prop.example is None


# ============================================================================
# ClassTemplate Tests
# ============================================================================

class TestClassTemplate:
    """Tests for ClassTemplate dataclass."""
    
    def test_create_class_template(self):
        """Test creating a class template."""
        cls = ClassTemplate(
            uri="http://example.org/Person",
            label="Person",
            description="A human being",
            properties=[
                PropertyTemplate(
                    uri="http://example.org/name",
                    label="name",
                    description="Full name",
                    range_type="literal",
                )
            ],
        )
        
        assert cls.uri == "http://example.org/Person"
        assert len(cls.properties) == 1
    
    def test_parent_class(self):
        """Test class with parent."""
        cls = ClassTemplate(
            uri="http://example.org/Employee",
            label="Employee",
            description="An employee",
            parent_class="http://example.org/Person",
        )
        
        assert cls.parent_class == "http://example.org/Person"


# ============================================================================
# OntologyPack Tests
# ============================================================================

class TestOntologyPack:
    """Tests for OntologyPack dataclass."""
    
    def test_create_pack(self):
        """Test creating an ontology pack."""
        pack = OntologyPack(
            pack_type=OntologyPackType.PROV_O,
            name="Test Pack",
            description="A test pack",
            namespace="http://example.org/",
            prefix="ex",
        )
        
        assert pack.pack_type == OntologyPackType.PROV_O
        assert pack.namespace == "http://example.org/"
        assert pack.prefix == "ex"
    
    def test_get_prefixes(self):
        """Test getting prefix declarations."""
        pack = OntologyPack(
            pack_type=OntologyPackType.DCAT,
            name="Test",
            description="Test",
            namespace="http://example.org/test#",
            prefix="test",
        )
        
        prefixes = pack.get_prefixes()
        assert prefixes == {"test": "http://example.org/test#"}
    
    def test_get_class(self):
        """Test getting a class by local name."""
        pack = OntologyPack(
            pack_type=OntologyPackType.PAV,
            name="Test",
            description="Test",
            namespace="http://example.org/",
            prefix="ex",
            classes=[
                ClassTemplate(
                    uri="http://example.org/Thing",
                    label="Thing",
                    description="A thing",
                )
            ],
        )
        
        cls = pack.get_class("Thing")
        assert cls is not None
        assert cls.label == "Thing"
    
    def test_get_property(self):
        """Test getting a property by local name."""
        pack = OntologyPack(
            pack_type=OntologyPackType.PAV,
            name="Test",
            description="Test",
            namespace="http://example.org/",
            prefix="ex",
            properties=[
                PropertyTemplate(
                    uri="http://example.org/name",
                    label="name",
                    description="The name",
                    range_type="literal",
                )
            ],
        )
        
        prop = pack.get_property("name")
        assert prop is not None
        assert prop.label == "name"


# ============================================================================
# PROV-O Pack Tests
# ============================================================================

class TestProvOPack:
    """Tests for PROV-O ontology pack."""
    
    def test_create_pack(self):
        """Test creating PROV-O pack."""
        pack = create_prov_o_pack()
        
        assert pack.pack_type == OntologyPackType.PROV_O
        assert pack.name == "PROV-O"
        assert pack.namespace == PROV
        assert pack.prefix == "prov"
    
    def test_has_core_classes(self):
        """Test PROV-O has core classes."""
        pack = create_prov_o_pack()
        
        class_uris = [cls.uri for cls in pack.classes]
        assert f"{PROV}Entity" in class_uris
        assert f"{PROV}Activity" in class_uris
        assert f"{PROV}Agent" in class_uris
    
    def test_entity_properties(self):
        """Test Entity class has expected properties."""
        pack = create_prov_o_pack()
        
        entity = pack.get_class("Entity")
        assert entity is not None
        
        prop_uris = [p.uri for p in entity.properties]
        assert f"{PROV}wasGeneratedBy" in prop_uris
        assert f"{PROV}wasDerivedFrom" in prop_uris
        assert f"{PROV}wasAttributedTo" in prop_uris
    
    def test_activity_properties(self):
        """Test Activity class has expected properties."""
        pack = create_prov_o_pack()
        
        activity = pack.get_class("Activity")
        assert activity is not None
        
        prop_uris = [p.uri for p in activity.properties]
        assert f"{PROV}used" in prop_uris
        assert f"{PROV}wasAssociatedWith" in prop_uris
        assert f"{PROV}startedAtTime" in prop_uris
        assert f"{PROV}endedAtTime" in prop_uris
    
    def test_agent_subclasses(self):
        """Test Agent has subclasses."""
        pack = create_prov_o_pack()
        
        person = pack.get_class("Person")
        assert person is not None
        assert person.parent_class == f"{PROV}Agent"
        
        org = pack.get_class("Organization")
        assert org is not None
        assert org.parent_class == f"{PROV}Agent"
        
        software = pack.get_class("SoftwareAgent")
        assert software is not None
        assert software.parent_class == f"{PROV}Agent"


# ============================================================================
# DCAT Pack Tests
# ============================================================================

class TestDcatPack:
    """Tests for DCAT ontology pack."""
    
    def test_create_pack(self):
        """Test creating DCAT pack."""
        pack = create_dcat_pack()
        
        assert pack.pack_type == OntologyPackType.DCAT
        assert pack.name == "DCAT"
        assert pack.namespace == DCAT
        assert pack.prefix == "dcat"
    
    def test_has_core_classes(self):
        """Test DCAT has core classes."""
        pack = create_dcat_pack()
        
        class_uris = [cls.uri for cls in pack.classes]
        assert f"{DCAT}Catalog" in class_uris
        assert f"{DCAT}Dataset" in class_uris
        assert f"{DCAT}Distribution" in class_uris
        assert f"{DCAT}DataService" in class_uris
    
    def test_dataset_has_required_properties(self):
        """Test Dataset class has required title property."""
        pack = create_dcat_pack()
        
        dataset = pack.get_class("Dataset")
        assert dataset is not None
        
        title_props = [p for p in dataset.properties if "title" in p.label]
        assert len(title_props) > 0
        assert title_props[0].required is True
    
    def test_distribution_properties(self):
        """Test Distribution class has expected properties."""
        pack = create_dcat_pack()
        
        dist = pack.get_class("Distribution")
        assert dist is not None
        
        prop_labels = [p.label for p in dist.properties]
        assert "access URL" in prop_labels
        assert "download URL" in prop_labels
        assert "byte size" in prop_labels


# ============================================================================
# PAV Pack Tests
# ============================================================================

class TestPavPack:
    """Tests for PAV ontology pack."""
    
    def test_create_pack(self):
        """Test creating PAV pack."""
        pack = create_pav_pack()
        
        assert pack.pack_type == OntologyPackType.PAV
        assert pack.name == "PAV"
        assert pack.namespace == PAV
        assert pack.prefix == "pav"
    
    def test_has_authoring_properties(self):
        """Test PAV has authoring properties."""
        pack = create_pav_pack()
        
        prop_uris = [p.uri for p in pack.properties]
        assert f"{PAV}authoredBy" in prop_uris
        assert f"{PAV}authoredOn" in prop_uris
        assert f"{PAV}createdBy" in prop_uris
        assert f"{PAV}createdOn" in prop_uris
    
    def test_has_provenance_properties(self):
        """Test PAV has provenance properties."""
        pack = create_pav_pack()
        
        prop_uris = [p.uri for p in pack.properties]
        assert f"{PAV}derivedFrom" in prop_uris
        assert f"{PAV}importedFrom" in prop_uris
        assert f"{PAV}retrievedFrom" in prop_uris
    
    def test_has_versioning_properties(self):
        """Test PAV has versioning properties."""
        pack = create_pav_pack()
        
        prop_uris = [p.uri for p in pack.properties]
        assert f"{PAV}version" in prop_uris
        assert f"{PAV}previousVersion" in prop_uris
        assert f"{PAV}hasCurrentVersion" in prop_uris
    
    def test_datetime_properties_have_datatype(self):
        """Test that datetime properties have correct datatype."""
        pack = create_pav_pack()
        
        # Properties ending in "On" should have dateTime datatype
        for prop in pack.properties:
            if prop.uri.endswith("On"):  # authoredOn, createdOn, etc.
                if prop.range_type == "literal":
                    assert prop.datatype == f"{XSD}dateTime", f"{prop.uri} should have dateTime datatype"


# ============================================================================
# OntologyPackManager Tests
# ============================================================================

class TestOntologyPackManager:
    """Tests for OntologyPackManager."""
    
    def test_create_manager(self):
        """Test creating a pack manager."""
        manager = OntologyPackManager()
        
        assert len(manager.enabled_packs) == 0
        assert len(manager.get_all_packs()) == 3
    
    def test_enable_pack(self):
        """Test enabling a pack."""
        manager = OntologyPackManager()
        
        manager.enable_pack(OntologyPackType.PROV_O)
        
        assert manager.is_enabled(OntologyPackType.PROV_O) is True
        assert manager.is_enabled(OntologyPackType.DCAT) is False
    
    def test_disable_pack(self):
        """Test disabling a pack."""
        manager = OntologyPackManager()
        
        manager.enable_pack(OntologyPackType.PROV_O)
        manager.enable_pack(OntologyPackType.DCAT)
        manager.disable_pack(OntologyPackType.PROV_O)
        
        assert manager.is_enabled(OntologyPackType.PROV_O) is False
        assert manager.is_enabled(OntologyPackType.DCAT) is True
    
    def test_get_pack(self):
        """Test getting a pack by type."""
        manager = OntologyPackManager()
        
        pack = manager.get_pack(OntologyPackType.PROV_O)
        
        assert pack is not None
        assert pack.pack_type == OntologyPackType.PROV_O
    
    def test_get_enabled_packs(self):
        """Test getting all enabled packs."""
        manager = OntologyPackManager()
        
        manager.enable_pack(OntologyPackType.PROV_O)
        manager.enable_pack(OntologyPackType.PAV)
        
        packs = manager.get_enabled_packs()
        assert len(packs) == 2
    
    def test_get_all_prefixes(self):
        """Test getting all prefixes from enabled packs."""
        manager = OntologyPackManager()
        
        manager.enable_pack(OntologyPackType.PROV_O)
        manager.enable_pack(OntologyPackType.DCAT)
        
        prefixes = manager.get_all_prefixes()
        assert "prov" in prefixes
        assert "dcat" in prefixes
        assert prefixes["prov"] == PROV
    
    def test_suggest_classes(self):
        """Test class suggestions."""
        manager = OntologyPackManager()
        manager.enable_pack(OntologyPackType.PROV_O)
        
        suggestions = manager.suggest_classes("ent")
        
        assert len(suggestions) > 0
        assert any("Entity" in s.label for s in suggestions)
    
    def test_suggest_classes_all(self):
        """Test getting all class suggestions."""
        manager = OntologyPackManager()
        manager.enable_pack(OntologyPackType.PROV_O)
        manager.enable_pack(OntologyPackType.DCAT)
        
        suggestions = manager.suggest_classes()
        
        # Should have classes from both packs
        assert len(suggestions) > 5
    
    def test_suggest_properties(self):
        """Test property suggestions."""
        manager = OntologyPackManager()
        manager.enable_pack(OntologyPackType.PAV)
        
        suggestions = manager.suggest_properties(partial="auth")
        
        assert len(suggestions) > 0
        assert any("authored" in s.label.lower() for s in suggestions)
    
    def test_suggest_properties_by_domain(self):
        """Test property suggestions filtered by domain."""
        manager = OntologyPackManager()
        manager.enable_pack(OntologyPackType.PROV_O)
        
        suggestions = manager.suggest_properties(
            domain_class=f"{PROV}Activity"
        )
        
        # Should only include Activity properties
        assert any("used" in s.label.lower() for s in suggestions)
    
    def test_suggest_range_values(self):
        """Test range value suggestions."""
        manager = OntologyPackManager()
        manager.enable_pack(OntologyPackType.PAV)
        
        suggestions = manager.suggest_range_values(f"{PAV}authoredOn")
        
        assert suggestions["type"] == "literal"
        assert suggestions["datatype"] == f"{XSD}dateTime"
        assert "format" in suggestions
    
    def test_suggest_range_values_resource(self):
        """Test range value suggestions for resource type."""
        manager = OntologyPackManager()
        manager.enable_pack(OntologyPackType.PROV_O)
        
        suggestions = manager.suggest_range_values(f"{PROV}wasGeneratedBy")
        
        assert suggestions["type"] == "resource"
        assert suggestions["rangeClass"] == f"{PROV}Activity"
    
    def test_generate_template(self):
        """Test template generation."""
        manager = OntologyPackManager()
        manager.enable_pack(OntologyPackType.PROV_O)
        
        template = manager.generate_template(
            class_uri=f"{PROV}Entity",
            subject_uri="http://example.org/doc1",
        )
        
        assert "@prefix prov:" in template
        assert "http://example.org/doc1" in template
        assert "a prov:Entity" in template
    
    def test_generate_template_required_only(self):
        """Test template generation with required properties only."""
        manager = OntologyPackManager()
        manager.enable_pack(OntologyPackType.DCAT)
        
        template = manager.generate_template(
            class_uri=f"{DCAT}Dataset",
            subject_uri="http://example.org/ds1",
            include_optional=False,
        )
        
        # Should include required title but maybe not all optional
        assert "http://example.org/ds1" in template
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        manager = OntologyPackManager()
        manager.enable_pack(OntologyPackType.PROV_O)
        manager.enable_pack(OntologyPackType.DCAT)
        
        d = manager.to_dict()
        
        assert "enabledPacks" in d
        assert "prov-o" in d["enabledPacks"]
        assert "dcat" in d["enabledPacks"]
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {"enabledPacks": ["prov-o", "pav"]}
        
        manager = OntologyPackManager.from_dict(data)
        
        assert manager.is_enabled(OntologyPackType.PROV_O) is True
        assert manager.is_enabled(OntologyPackType.PAV) is True
        assert manager.is_enabled(OntologyPackType.DCAT) is False
    
    def test_from_dict_invalid_pack(self):
        """Test deserialization ignores invalid pack types."""
        data = {"enabledPacks": ["prov-o", "invalid-pack"]}
        
        manager = OntologyPackManager.from_dict(data)
        
        assert manager.is_enabled(OntologyPackType.PROV_O) is True
        assert len(manager.enabled_packs) == 1
    
    def test_get_summary(self):
        """Test getting pack summary."""
        manager = OntologyPackManager()
        manager.enable_pack(OntologyPackType.PROV_O)
        
        summary = manager.get_summary()
        
        assert summary["enabledCount"] == 1
        assert len(summary["packs"]) == 3
        
        prov_pack = next(p for p in summary["packs"] if p["type"] == "prov-o")
        assert prov_pack["enabled"] is True
        assert prov_pack["classCount"] > 0


# ============================================================================
# SchemaGuidance Tests
# ============================================================================

class TestSchemaGuidance:
    """Tests for SchemaGuidance."""
    
    def test_create_guidance(self):
        """Test creating schema guidance."""
        manager = OntologyPackManager()
        guidance = SchemaGuidance(manager)
        
        assert guidance.pack_manager is manager
        assert guidance.shacl_manager is None
    
    def test_autocomplete_prefix(self):
        """Test autocomplete for prefixed names."""
        manager = OntologyPackManager()
        manager.enable_pack(OntologyPackType.PROV_O)
        
        guidance = SchemaGuidance(manager)
        
        suggestions = guidance.get_autocomplete("prov:Ent", 8)
        
        assert len(suggestions) > 0
        assert any("Entity" in s["label"] for s in suggestions)
    
    def test_autocomplete_after_type(self):
        """Test autocomplete after 'a' keyword."""
        manager = OntologyPackManager()
        manager.enable_pack(OntologyPackType.PROV_O)
        
        guidance = SchemaGuidance(manager)
        
        suggestions = guidance.get_autocomplete("ex:doc a", 8)
        
        # Should suggest classes
        assert len(suggestions) > 0
        assert any(s["kind"] == "class" for s in suggestions)
    
    def test_validate_and_suggest_valid(self):
        """Test validation with valid data."""
        manager = OntologyPackManager()
        manager.enable_pack(OntologyPackType.PROV_O)
        
        guidance = SchemaGuidance(manager)
        
        triples = [
            ("http://example.org/doc", f"{RDF}type", f"{PROV}Entity"),
            ("http://example.org/doc", f"{PROV}wasGeneratedBy", "http://example.org/activity"),
        ]
        
        result = guidance.validate_and_suggest(triples)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_and_suggest_with_suggestions(self):
        """Test validation suggests additional properties."""
        manager = OntologyPackManager()
        manager.enable_pack(OntologyPackType.PROV_O)
        
        guidance = SchemaGuidance(manager)
        
        triples = [
            ("http://example.org/doc", f"{RDF}type", f"{PROV}Entity"),
            # Missing many optional properties
        ]
        
        result = guidance.validate_and_suggest(triples)
        
        # Should suggest Entity properties
        assert len(result["suggestions"]) > 0
    
    def test_validate_and_suggest_missing_required(self):
        """Test validation reports missing required properties."""
        manager = OntologyPackManager()
        manager.enable_pack(OntologyPackType.DCAT)
        
        guidance = SchemaGuidance(manager)
        
        triples = [
            ("http://example.org/ds", f"{RDF}type", f"{DCAT}Dataset"),
            # Missing required title
        ]
        
        result = guidance.validate_and_suggest(triples)
        
        # Should have suggestion for missing required
        missing_required = [s for s in result["suggestions"] if s["type"] == "missing_required"]
        assert len(missing_required) > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestOntologyPacksIntegration:
    """Integration tests for ontology packs."""
    
    def test_all_packs_have_consistent_structure(self):
        """Test all packs have consistent structure."""
        packs = [
            create_prov_o_pack(),
            create_dcat_pack(),
            create_pav_pack(),
        ]
        
        for pack in packs:
            assert pack.name is not None
            assert pack.namespace is not None
            assert pack.prefix is not None
            assert len(pack.namespace) > 0
            
            # All classes should have valid URIs
            for cls in pack.classes:
                assert cls.uri.startswith("http")
                assert cls.label is not None
            
            # All properties should have valid URIs
            for prop in pack.properties:
                assert prop.uri.startswith("http")
                assert prop.label is not None
    
    def test_pack_manager_round_trip(self):
        """Test pack manager serialization round trip."""
        manager1 = OntologyPackManager()
        manager1.enable_pack(OntologyPackType.PROV_O)
        manager1.enable_pack(OntologyPackType.PAV)
        
        data = manager1.to_dict()
        manager2 = OntologyPackManager.from_dict(data)
        
        assert manager2.is_enabled(OntologyPackType.PROV_O) is True
        assert manager2.is_enabled(OntologyPackType.PAV) is True
        assert manager2.is_enabled(OntologyPackType.DCAT) is False
    
    def test_template_generates_valid_turtle_structure(self):
        """Test generated templates have valid structure."""
        manager = OntologyPackManager()
        manager.enable_pack(OntologyPackType.PROV_O)
        
        template = manager.generate_template(
            class_uri=f"{PROV}Activity",
            subject_uri="http://example.org/act1",
        )
        
        # Should have prefix declarations
        assert "@prefix" in template
        
        # Should have subject and type
        assert "http://example.org/act1" in template
        assert "Activity" in template
        
        # Should end with period
        assert template.strip().endswith(".")
    
    def test_guidance_autocomplete_performance(self):
        """Test autocomplete with all packs enabled is performant."""
        manager = OntologyPackManager()
        manager.enable_pack(OntologyPackType.PROV_O)
        manager.enable_pack(OntologyPackType.DCAT)
        manager.enable_pack(OntologyPackType.PAV)
        
        guidance = SchemaGuidance(manager)
        
        # Should return quickly even with all packs
        import time
        start = time.time()
        
        for _ in range(100):
            guidance.get_autocomplete("prov:", 5)
        
        elapsed = time.time() - start
        assert elapsed < 1.0  # 100 calls should take < 1 second
