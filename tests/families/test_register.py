import pytest
from unittest.mock import Mock

# Import the actual registry functions and base class
from rapidroots.families.registry import register_family, get_family, all_families, registry_version
from rapidroots.families.base import FunctionFamily

# Import the registry module to access private state for test isolation
import rapidroots.families.registry as _reg


class TestFunctionFamilyRegistry:
    """Test suite for function family registry following best practices"""
    
    def setup_method(self):
        """Reset registry before each test for independence"""
        # Clear the private state for full test isolation
        _reg._registry.clear()
        _reg._registry_version = 0
    
    def test_register_family_adds_family_to_registry(self):
        """Test that register_family correctly adds a family to the registry"""
        # Arrange
        family = Mock(spec=FunctionFamily)
        family.name = "test_family"
        
        # Act
        register_family(family)
        
        # Assert
        registered_families = all_families()
        assert "test_family" in registered_families
        assert registered_families["test_family"] is family
    
    def test_register_family_increments_version(self):
        """Test that register_family increments the registry version"""
        # Arrange
        family = Mock(spec=FunctionFamily)
        family.name = "test_family"
        initial_version = registry_version()
        
        # Act
        register_family(family)
        
        # Assert
        assert registry_version() == initial_version + 1
    
    def test_register_multiple_families_increments_version_correctly(self):
        """Test that registering multiple families increments version correctly"""
        # Arrange
        family1 = Mock(spec=FunctionFamily)
        family1.name = "family1"
        family2 = Mock(spec=FunctionFamily)
        family2.name = "family2"
        initial_version = registry_version()
        
        # Act
        register_family(family1)
        register_family(family2)
        
        # Assert
        assert registry_version() == initial_version + 2
    
    def test_register_family_overwrites_existing_family(self):
        """Test that registering a family with existing name overwrites it"""
        # Arrange
        family1 = Mock(spec=FunctionFamily)
        family1.name = "duplicate_name"
        family2 = Mock(spec=FunctionFamily)
        family2.name = "duplicate_name"
        register_family(family1)
        
        # Act
        register_family(family2)
        
        # Assert
        registered_families = all_families()
        assert registered_families["duplicate_name"] is family2
        assert registered_families["duplicate_name"] is not family1
    
    def test_get_family_returns_registered_family(self):
        """Test that get_family returns the correct registered family"""
        # Arrange
        family = Mock(spec=FunctionFamily)
        family.name = "test_family"
        register_family(family)
        
        # Act
        retrieved_family = get_family("test_family")
        
        # Assert
        assert retrieved_family is family
    
    def test_get_family_raises_keyerror_for_unregistered_family(self):
        """Test that get_family raises KeyError for unregistered family"""
        # Arrange & Act & Assert
        with pytest.raises(KeyError):
            get_family("nonexistent_family")
    
    def test_all_families_returns_copy_of_registry(self):
        """Test that all_families returns a copy of the registry"""
        # Arrange
        family1 = Mock(spec=FunctionFamily)
        family1.name = "family1"
        family2 = Mock(spec=FunctionFamily)
        family2.name = "family2"
        register_family(family1)
        register_family(family2)
        
        # Act
        families_copy1 = all_families()
        families_copy2 = all_families()
        
        # Assert
        assert families_copy1 == families_copy2
        assert families_copy1 is not families_copy2  # Ensure it's a copy
    
    def test_all_families_modifications_dont_affect_registry(self):
        """Test that modifying the returned dict doesn't affect the registry"""
        # Arrange
        family = Mock(spec=FunctionFamily)
        family.name = "test_family"
        register_family(family)
        initial_families = all_families()
        initial_count = len(initial_families)
        
        # Act
        families_copy = all_families()
        families_copy["new_family_not_in_registry"] = Mock(spec=FunctionFamily)
        
        # Assert
        current_families = all_families()
        assert "new_family_not_in_registry" not in current_families
        assert len(current_families) == initial_count
    
    def test_all_families_returns_empty_dict_when_no_families_registered(self):
        """Test that all_families returns empty dict when no families are registered"""
        # Arrange & Act
        families = all_families()
        
        # Assert
        assert families == {}
        assert isinstance(families, dict)
    
    def test_registry_version_starts_at_zero(self):
        """Test that registry version starts at zero after reset"""
        # Arrange & Act & Assert
        assert registry_version() == 0
    
    def test_registry_independence_between_tests(self):
        """Test that registry state is independent between tests"""
        # This test verifies our setup_method works correctly
        # Arrange & Act & Assert
        assert len(all_families()) == 0
        assert registry_version() == 0
    
    def test_multiple_registrations_same_family_updates_version(self):
        """Test that re-registering the same family name still increments version"""
        # Arrange
        family1 = Mock(spec=FunctionFamily)
        family1.name = "same_name"
        family2 = Mock(spec=FunctionFamily)
        family2.name = "same_name"
        
        # Act
        register_family(family1)
        version_after_first = registry_version()
        register_family(family2)
        version_after_second = registry_version()
        
        # Assert
        assert version_after_second == version_after_first + 1
        assert len(all_families()) == 1  # Only one entry despite two registrations


class TestRegistryEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Reset registry before each test for independence"""
        # Clear the private state for full test isolation
        _reg._registry.clear()
        _reg._registry_version = 0
    
    def test_register_family_with_none_name(self):
        """Test behavior when family has None as name"""
        # Arrange
        family = Mock(spec=FunctionFamily)
        family.name = None
        
        # Act
        register_family(family)
        
        # Assert
        registered_families = all_families()
        assert None in registered_families
        assert registered_families[None] is family
    
    def test_register_family_with_empty_string_name(self):
        """Test behavior when family has empty string as name"""
        # Arrange
        family = Mock(spec=FunctionFamily)
        family.name = ""
        
        # Act
        register_family(family)
        
        # Assert
        registered_families = all_families()
        assert "" in registered_families
        assert registered_families[""] is family
    
    def test_get_family_with_none_key(self):
        """Test get_family behavior with None key"""
        # Arrange
        family = Mock(spec=FunctionFamily)
        family.name = None
        register_family(family)
        
        # Act
        retrieved_family = get_family(None)
        
        # Assert
        assert retrieved_family is family
    
    def test_get_family_with_empty_string_key(self):
        """Test get_family behavior with empty string key"""
        # Arrange
        family = Mock(spec=FunctionFamily)
        family.name = ""
        register_family(family)
        
        # Act
        retrieved_family = get_family("")
        
        # Assert
        assert retrieved_family is family


# Additional integration-style test
class TestRegistryIntegration:
    """Integration tests for registry functionality"""
    
    def setup_method(self):
        """Reset registry before each test for independence"""
        # Clear the private state for full test isolation
        _reg._registry.clear()
        _reg._registry_version = 0
    
    def test_complete_registry_workflow(self):
        """Test complete workflow: register, retrieve, list, version tracking"""
        # Arrange
        families = []
        for i in range(3):
            family = Mock(spec=FunctionFamily)
            family.name = f"integration_test_family_{i}"
            families.append(family)
        
        # Act & Assert - Register families
        for i, family in enumerate(families):
            register_family(family)
            assert registry_version() == i + 1
        
        # Act & Assert - Retrieve families
        for family in families:
            retrieved = get_family(family.name)
            assert retrieved is family
        
        # Act & Assert - List all families
        all_registered = all_families()
        assert len(all_registered) == 3
        for family in families:
            assert family.name in all_registered
            assert all_registered[family.name] is family
        
        # Final version check
        assert registry_version() == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])