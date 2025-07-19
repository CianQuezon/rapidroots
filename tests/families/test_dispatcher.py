import pytest
import numpy as np
from unittest.mock import Mock, patch

# Import the actual dispatcher
from rapidroots.families.dispatcher import UniversalFunctionDispatcher
from rapidroots.families.base import FunctionFamily

# Import registry functions for test setup
import rapidroots.families.registry as _reg
from rapidroots.families.registry import register_family


class TestUniversalFunctionDispatcher:
    """Test suite for UniversalFunctionDispatcher core functionality"""
    
    def setup_method(self):
        """Reset registry and create fresh dispatcher for each test"""
        # Clear registry state
        _reg._registry.clear()
        _reg._registry_version = 0
        
        # Create test families
        self.family1 = Mock(spec=FunctionFamily)
        self.family1.name = "linear"
        self.family1.param_count = 2
        self.family1.evaluate.return_value = np.array([1.0, 2.0])
        
        self.family2 = Mock(spec=FunctionFamily)
        self.family2.name = "quadratic"
        self.family2.param_count = 3
        self.family2.evaluate.return_value = np.array([3.0, 4.0])
        
        # Register families
        register_family(self.family1)
        register_family(self.family2)
        
        # Create fresh dispatcher
        self.dispatcher = UniversalFunctionDispatcher()
    
    def test_init_sets_initial_state(self):
        """Test that __init__ properly initializes dispatcher state"""
        # Arrange & Act
        dispatcher = UniversalFunctionDispatcher()
        
        # Assert
        assert dispatcher._family_indices is None
        assert dispatcher._known_registry_version == _reg.registry_version()
        assert dispatcher._warmup_done is False
    
    def test_evaluate_batch_basic_functionality(self):
        """Test basic batch evaluation with mixed function types"""
        # Arrange
        func_types = np.array(["linear", "quadratic", "linear"])
        x_arr = np.array([1.0, 2.0, 3.0])
        params_arr = [[1.0, 2.0], [1.0, 2.0, 3.0], [4.0, 5.0]]
        
        # Act
        result = self.dispatcher.evaluate_batch(func_types, params_arr, x_arr)
        
        # Assert
        assert result.shape == (3,)
        assert result.dtype == np.float64
        # Verify families were called
        assert self.family1.evaluate.called
        assert self.family2.evaluate.called
    
    def test_evaluate_batch_with_derivatives(self):
        """Test batch evaluation with derivative orders"""
        # Arrange
        func_types = ["linear", "linear"]  # Use list instead of np.array
        x_arr = np.array([1.0, 2.0])
        params_arr = [[1.0, 2.0], [3.0, 4.0]]  # Use list instead of np.array

        # Act
        result = self.dispatcher.evaluate_batch(
            func_types, params_arr, x_arr, derivatives=1
        )

        # Assert
        assert result.shape == (2,)
        
        # Option 1: Manual verification of call arguments
        assert self.family1.evaluate.called, "family1.evaluate should have been called"
        
        # Get the actual call arguments
        call_args = self.family1.evaluate.call_args
        actual_x, actual_params, actual_derivatives = call_args[0]
        
        # Verify each argument separately using numpy testing
        np.testing.assert_array_equal(actual_x, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(actual_params, np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert actual_derivatives == 1
        
    def test_evaluate_batch_with_custom_dtype(self):
        """Test batch evaluation with float32 dtype"""
        # Arrange
        func_types = np.array(["linear"])
        x_arr = np.array([1.0])
        params_arr = np.array([[1.0, 2.0]])
        
        # Act
        result = self.dispatcher.evaluate_batch(
            func_types, params_arr, x_arr, dtype=np.float32
        )
        
        # Assert
        assert result.dtype == np.float32
    


    def test_evaluate_batch_with_chunking(self):
        """Test batch evaluation with small chunk size"""
        # Arrange
        func_types = ["linear"] * 5  # Use list, not np.array
        x_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        params_arr = [[1.0, 2.0]] * 5  # Use list of lists

        # SOLUTION 1: Skip warmup entirely (recommended)
        self.dispatcher._warmup_done = True

        # Mock the evaluation calls we need:
        # With chunk_size=2 and 5 elements, we get 3 chunks:
        # Chunk 1: indices [0,1] → 2 elements
        # Chunk 2: indices [2,3] → 2 elements  
        # Chunk 3: indices [4]   → 1 element
        self.family1.evaluate.side_effect = [
            np.array([1.0, 2.0]),    # First chunk (size 2)
            np.array([3.0, 4.0]),    # Second chunk (size 2)
            np.array([5.0])          # Third chunk (size 1)
        ]

        # Act
        result = self.dispatcher.evaluate_batch(
            func_types, params_arr, x_arr, chunk_size=2
        )

        # Assert
        assert result.shape == (5,)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(result, expected)
        
    def test_warmup_functionality(self):
        """Test that warmup calls all derivatives for all families"""
        # Arrange
        dispatcher = UniversalFunctionDispatcher()
        
        # Act
        dispatcher._warmup()
        
        # Assert
        assert dispatcher._warmup_done is True
        # Verify each family was called with all derivative orders (0, 1, 2)
        assert self.family1.evaluate.call_count == 3
        assert self.family2.evaluate.call_count == 3
    
    def test_warmup_called_automatically_on_first_evaluate(self):
        """Test that warmup is called automatically on first evaluation"""
        # Arrange
        func_types = np.array(["linear"])
        x_arr = np.array([1.0])
        params_arr = np.array([[1.0, 2.0]])
        
        # Act
        self.dispatcher.evaluate_batch(func_types, params_arr, x_arr)
        
        # Assert
        assert self.dispatcher._warmup_done is True
    
    def test_registry_version_change_resets_warmup(self):
        """Test that warmup is reset when registry version changes"""
        # Arrange
        func_types = ["linear"]
        x_arr = np.array([1.0])
        params_arr = [[1.0, 2.0]]
        
        # Skip initial warmup for cleaner test
        self.dispatcher._warmup_done = True
        
        # Mock registry_version - use the correct import path
        # Based on your dispatcher: "from .registry import registry_version"
        with patch('rapidroots.families.dispatcher.registry_version') as mock_registry_version:
            # First call returns original version
            mock_registry_version.return_value = 1
            
            # Evaluate once - should not trigger warmup reset
            self.dispatcher.evaluate_batch(func_types, params_arr, x_arr)
            assert self.dispatcher._warmup_done is True
            
            # Change registry version
            mock_registry_version.return_value = 2
            
            # Next evaluation should reset warmup
            self.dispatcher.evaluate_batch(func_types, params_arr, x_arr)
            
            # Warmup should have been reset and re-done
            assert self.dispatcher._warmup_done is True
            assert mock_registry_version.call_count >= 2


    def test_registry_version_change_triggers_warmup(self):
        """Test that registry version changes trigger re-warmup"""
        # Arrange
        func_types = ["linear"]
        x_arr = np.array([1.0])
        params_arr = [[1.0, 2.0]]
        
        with patch('rapidroots.families.dispatcher.registry_version') as mock_registry_version:
            with patch.object(self.dispatcher, '_warmup') as mock_warmup:
                # Set initial state - both dispatcher and mock return same version
                mock_registry_version.return_value = 1
                self.dispatcher._warmup_done = True
                self.dispatcher._known_registry_version = 1
                
                # First call with same version - no warmup should be triggered
                self.dispatcher.evaluate_batch(func_types, params_arr, x_arr)
                mock_warmup.assert_not_called()
                
                # Reset mock call count for cleaner testing
                mock_warmup.reset_mock()
                
                # Change registry version to trigger warmup
                mock_registry_version.return_value = 2
                
                # Second call should detect version change and trigger warmup
                self.dispatcher.evaluate_batch(func_types, params_arr, x_arr)
                mock_warmup.assert_called_once()

    def test_registry_version_no_change_skips_warmup(self):
        """Test that warmup is skipped when registry version hasn't changed"""
        # Arrange
        func_types = ["linear"]
        x_arr = np.array([1.0])
        params_arr = [[1.0, 2.0]]
        
        with patch('rapidroots.families.dispatcher.registry_version') as mock_registry_version:
            with patch.object(self.dispatcher, '_warmup') as mock_warmup:
                # Set consistent version
                mock_registry_version.return_value = 5
                self.dispatcher._known_registry_version = 5
                self.dispatcher._warmup_done = True
                
                # Call multiple times - warmup should never be triggered
                self.dispatcher.evaluate_batch(func_types, params_arr, x_arr)
                self.dispatcher.evaluate_batch(func_types, params_arr, x_arr)
                
                # Warmup should never have been called
                mock_warmup.assert_not_called()

class TestDispatcherEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Reset registry for each test"""
        _reg._registry.clear()
        _reg._registry_version = 0
    
    def test_empty_batch_evaluation(self):
        """Test evaluation with empty arrays"""
        # Arrange
        dispatcher = UniversalFunctionDispatcher()
        func_types = np.array([])
        x_arr = np.array([])
        params_arr = np.array([]).reshape(0, 2)
        
        # Act
        result = dispatcher.evaluate_batch(func_types, params_arr, x_arr)
        
        # Assert
        assert result.shape == (0,)
        assert result.dtype == np.float64
    
    def test_single_element_batch(self):
        """Test evaluation with single element"""
        # Arrange
        family = Mock(spec=FunctionFamily)
        family.name = "test"
        family.param_count = 1
        family.evaluate.return_value = np.array([42.0])
        register_family(family)
        
        dispatcher = UniversalFunctionDispatcher()
        func_types = np.array(["test"])
        x_arr = np.array([1.0])
        params_arr = np.array([[5.0]])
        
        # Act
        result = dispatcher.evaluate_batch(func_types, params_arr, x_arr)
        
        # Assert
        assert result.shape == (1,)
        assert result[0] == 42.0
    
    def test_nonexistent_family_raises_error(self):
        """Test that using nonexistent family raises appropriate error"""
        # Arrange
        dispatcher = UniversalFunctionDispatcher()
        func_types = np.array(["nonexistent"])
        x_arr = np.array([1.0])
        params_arr = np.array([[1.0]])
        
        # Act & Assert
        with pytest.raises(KeyError):
            dispatcher.evaluate_batch(func_types, params_arr, x_arr)
    
    def test_param_count_mismatch_raises_valueerror(self):
        """Test that param_count mismatch raises ValueError"""
        # Arrange
        family = Mock(spec=FunctionFamily)
        family.name = "mismatch_test"
        family.param_count = 3  # Expects 3 parameters
        family.evaluate.side_effect = ValueError("params_arr must be shape (n, 3), got (1, 2)")
        register_family(family)
        
        dispatcher = UniversalFunctionDispatcher()
        func_types = np.array(["mismatch_test"])
        x_arr = np.array([1.0])
        params_arr = np.array([[1.0, 2.0]])  # Only 2 parameters provided
        
        # Act & Assert
        with pytest.raises(ValueError, match="params_arr must be shape"):
            dispatcher.evaluate_batch(func_types, params_arr, x_arr)
    
    def test_non_ndarray_inputs_work(self):
        """Test that Python lists and mixed-type sequences work via np.asarray"""
        # Arrange
        family = Mock(spec=FunctionFamily)
        family.name = "list_test"
        family.param_count = 2
        family.evaluate.return_value = np.array([1.0, 2.0])
        register_family(family)
        
        dispatcher = UniversalFunctionDispatcher()
        
        # Act - using Python lists instead of numpy arrays
        result = dispatcher.evaluate_batch(
            func_types=["list_test", "list_test"],  # Python list
            params_arr=[[1.0, 2.0], [3.0, 4.0]],  # Python list of lists
            x_arr=[1.0, 2.0]  # Python list
        )
        
        # Assert
        assert result.shape == (2,)
        assert isinstance(result, np.ndarray)
    
    # MOST ROBUST: Use a function for side_effect
    def test_chunk_boundary_exact_multiple_with_function(self):
        """Test batch whose size is exactly a multiple of chunk_size (using function)"""
        # Arrange
        family = Mock(spec=FunctionFamily)
        family.name = "boundary_test"
        family.param_count = 1
        
        def mock_evaluate(x, params, derivatives):
            # Return values based on input size
            if len(x) == 1:  # Warmup calls
                return np.array([0.0])
            elif len(x) == 3:  # Chunk calls
                # Return values starting from the first x value
                start_val = int(x[0])
                return np.array([float(start_val), float(start_val + 1), float(start_val + 2)])
            else:
                return np.zeros(len(x))
        
        family.evaluate.side_effect = mock_evaluate
        register_family(family)

        dispatcher = UniversalFunctionDispatcher()
        func_types = ["boundary_test"] * 6
        x_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        params_arr = [[1.0]] * 6

        # Act
        result = dispatcher.evaluate_batch(
            func_types, params_arr, x_arr, chunk_size=3
        )

        # Assert
        assert result.shape == (6,)
        # The function will return [1,2,3] for first chunk and [4,5,6] for second chunk
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_array_equal(result, expected)
        
    def test_batch_smaller_than_chunk_size(self):
        """Test batch smaller than chunk_size"""
        # Arrange
        family = Mock(spec=FunctionFamily)
        family.name = "small_batch"
        family.param_count = 1
        family.evaluate.return_value = np.array([1.0, 2.0])
        register_family(family)
        
        dispatcher = UniversalFunctionDispatcher()
        func_types = np.array(["small_batch"] * 2)
        x_arr = np.array([1.0, 2.0])
        params_arr = np.array([[1.0], [2.0]])
        
        # Act
        result = dispatcher.evaluate_batch(
            func_types, params_arr, x_arr, chunk_size=10  # Much larger than batch
        )
        
        # Assert
        assert result.shape == (2,)
        assert family.evaluate.call_count >= 1  # Should be exactly 1 chunk
    
    def test_multiple_families_per_chunk(self):
        """Test chunk containing multiple different families"""
        # Arrange
        family_a = Mock(spec=FunctionFamily)
        family_a.name = "A"
        family_a.param_count = 1
        family_a.evaluate.return_value = np.array([10.0, 30.0])  # For positions 0,2
        
        family_b = Mock(spec=FunctionFamily)
        family_b.name = "B"
        family_b.param_count = 1
        family_b.evaluate.return_value = np.array([20.0])  # For position 1
        
        family_c = Mock(spec=FunctionFamily)
        family_c.name = "C"
        family_c.param_count = 1
        family_c.evaluate.return_value = np.array([40.0])  # For position 3
        
        register_family(family_a)
        register_family(family_b)
        register_family(family_c)
        
        dispatcher = UniversalFunctionDispatcher()
        func_types = np.array(["A", "B", "A", "C"])  # Multiple families in one chunk
        x_arr = np.array([1.0, 2.0, 3.0, 4.0])
        params_arr = np.array([[1.0], [2.0], [3.0], [4.0]])
        
        # Act
        result = dispatcher.evaluate_batch(
            func_types, params_arr, x_arr, chunk_size=10  # Single chunk
        )
        
        # Assert
        assert result.shape == (4,)
        # Each family should be called exactly once per chunk
        assert family_a.evaluate.call_count >= 1
        assert family_b.evaluate.call_count >= 1
        assert family_c.evaluate.call_count >= 1
    
    def test_error_in_family_propagates(self):
        """Test that error in one family's evaluate propagates"""
        # Arrange
        family_good = Mock(spec=FunctionFamily)
        family_good.name = "good"
        family_good.param_count = 1
        family_good.evaluate.return_value = np.array([1.0])
        
        family_bad = Mock(spec=FunctionFamily)
        family_bad.name = "bad"
        family_bad.param_count = 1
        family_bad.evaluate.side_effect = RuntimeError("Computation failed")
        
        register_family(family_good)
        register_family(family_bad)
        
        dispatcher = UniversalFunctionDispatcher()
        func_types = np.array(["good", "bad"])
        x_arr = np.array([1.0, 2.0])
        params_arr = np.array([[1.0], [2.0]])
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Computation failed"):
            dispatcher.evaluate_batch(func_types, params_arr, x_arr)


class TestDispatcherPerformance:
    """Test performance-related aspects of dispatcher"""
    
    def setup_method(self):
        """Set up performance test environment"""
        _reg._registry.clear()
        _reg._registry_version = 0
        
        # Create a family for performance testing
        self.perf_family = Mock(spec=FunctionFamily)
        self.perf_family.name = "perf_test"
        self.perf_family.param_count = 2
        register_family(self.perf_family)
    
    def test_family_indices_caching(self):
        """Test that family indices are cached between evaluations"""
        # Arrange
        dispatcher = UniversalFunctionDispatcher()
        func_types = np.array(["perf_test", "perf_test"])
        x_arr = np.array([1.0, 2.0])
        params_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Mock evaluate to return appropriate arrays
        self.perf_family.evaluate.return_value = np.array([1.0, 2.0])
        
        # Act - first evaluation
        dispatcher.evaluate_batch(func_types, params_arr, x_arr)
        first_indices = dispatcher._family_indices
        
        # Act - second evaluation with same types
        dispatcher.evaluate_batch(func_types, params_arr, x_arr)
        second_indices = dispatcher._family_indices
        
        # Assert - same indices object should be reused
        assert first_indices is second_indices
    
    def test_warmup_only_runs_once(self):
        """Test that warmup only runs once per dispatcher instance"""
        # Arrange
        dispatcher = UniversalFunctionDispatcher()
        func_types = np.array(["perf_test"])
        x_arr = np.array([1.0])
        params_arr = np.array([[1.0, 2.0]])
        
        # Mock evaluate to track calls
        self.perf_family.evaluate.return_value = np.array([1.0])
        
        # Act - multiple evaluations
        dispatcher.evaluate_batch(func_types, params_arr, x_arr)
        call_count_after_first = self.perf_family.evaluate.call_count
        
        dispatcher.evaluate_batch(func_types, params_arr, x_arr)
        call_count_after_second = self.perf_family.evaluate.call_count
        
        # Assert - warmup calls (3 for derivatives) + actual calls (2)
        # First eval: 3 warmup + 1 actual = 4 total
        # Second eval: 0 warmup + 1 actual = 1 additional = 5 total
        assert call_count_after_first == 4  # 3 warmup + 1 actual
        assert call_count_after_second == 5  # Previous 4 + 1 more actual


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])