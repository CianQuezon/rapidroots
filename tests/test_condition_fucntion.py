import pytest
import numpy as np
import math
from numpy.testing import assert_array_equal, assert_allclose
from rapidroots.utils.function_utils import condition_function  # Replace with actual import


class TestConditionFunction:
    
    def test_normal_values_passthrough(self):
        """Test that normal values pass through unchanged."""
        def simple_func(x):
            return x * 2
        
        conditioned = condition_function(simple_func)
        result = conditioned([1.0, 2.0, 3.0])
        
        assert_array_equal(result, [2.0, 4.0, 6.0])
    
    def test_exception_handling(self):
        """Test that exceptions return NaN array."""
        def failing_func(x):
            raise ValueError("Function failed")
        
        conditioned = condition_function(failing_func)
        result = conditioned([1.0, 2.0])
        
        assert len(result) == 2
        assert np.all(np.isnan(result))
    
    def test_infinity_replacement(self):
        """Test infinity replacement with default values."""
        def inf_func(x):
            x = np.asarray(x)
            return np.array([np.inf, -np.inf, 5.0])
        
        conditioned = condition_function(inf_func)
        result = conditioned([1.0, 2.0, 3.0])
        
        expected = [1e10, -1e10, 5.0]
        assert_array_equal(result, expected)
    
    def test_custom_infinity_replacement(self):
        """Test custom infinity replacement value."""
        def inf_func(x):
            return np.array([np.inf, -np.inf])
        
        conditioned = condition_function(inf_func, inf_replacement=999.0)
        result = conditioned([1.0, 2.0])
        
        expected = [999.0, -999.0]
        assert_array_equal(result, expected)
    
    def test_zero_threshold_snapping(self):
        """Test small values get snapped to zero."""
        def small_func(x):
            x = np.asarray(x)
            # Use where to maintain shape and simulate small values
            return np.where(x == 1.0, 1e-16, 
                   np.where(x == 2.0, -1e-16,
                   np.where(x == 3.0, 1e-14, 0.001)))
        
        conditioned = condition_function(small_func, zero_threshold=1e-15)
        result = conditioned([1.0, 2.0, 3.0, 4.0])
        
        expected = [0.0, 0.0, 1e-14, 0.001]  # Only values < threshold become 0
        assert_array_equal(result, expected)
    
    def test_nan_preservation(self):
        """Test that NaN values are preserved."""
        def nan_func(x):
            x = np.asarray(x)
            # Return same shape with one NaN
            result = x * 2  # Normal operation
            result[1] = np.nan  # Inject NaN at index 1
            return result
        
        conditioned = condition_function(nan_func)
        result = conditioned([1.0, 2.0, 3.0])
        
        assert result[0] == 2.0  # 1*2
        assert np.isnan(result[1])  # Injected NaN
        assert result[2] == 6.0  # 3*2
    
    def test_shape_preservation(self):
        """Test that output shape matches input shape."""
        def ones_func(x):
            return np.ones_like(x)  # Same shape as input
        
        conditioned = condition_function(ones_func)
        x_input = np.array([[1, 2, 3], [4, 5, 6]])
        result = conditioned(x_input)
        
        assert result.shape == (2, 3)
        assert_array_equal(result, np.ones((2, 3)))
    
    def test_scalar_input(self):
        """Test function works with scalar inputs."""
        def square_func(x):
            return x ** 2
        
        conditioned = condition_function(square_func)
        result = conditioned(5.0)
        
        assert result.shape == ()  # Explicitly require 0-d array
        assert result.item() == 25.0
    
    def test_combined_issues(self):
        """Test function handling multiple issues simultaneously."""
        def problematic_func(x):
            x = np.asarray(x)
            # Return same shape with various problematic values
            result = np.zeros_like(x)
            if len(result) >= 5:
                result[0] = np.inf
                result[1] = -1e-16  
                result[2] = np.nan
                result[3] = -np.inf
                result[4] = 42.0
            return result
        
        conditioned = condition_function(
            problematic_func, 
            inf_replacement=100.0, 
            zero_threshold=1e-15
        )
        result = conditioned([1.0, 2.0, 3.0, 4.0, 5.0])
        
        assert result[0] == 100.0    # +inf replaced
        assert result[1] == 0.0      # small value snapped
        assert np.isnan(result[2])   # NaN preserved
        assert result[3] == -100.0   # -inf replaced
        assert result[4] == 42.0     # normal value unchanged
    
    def test_list_input_conversion(self):
        """Test that list inputs are converted to arrays."""
        def identity_func(x):
            return x
        
        conditioned = condition_function(identity_func)
        result = conditioned([1, 2, 3])  # List input
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == float
        assert_array_equal(result, [1.0, 2.0, 3.0])
    
    def test_division_by_zero_case(self):
        """Test real-world case: division by zero."""
        def divide_func(x):
            x = np.asarray(x)
            with np.errstate(divide='ignore'):
                return 1.0 / x  # Will create inf when x contains 0
        
        conditioned = condition_function(divide_func, inf_replacement=999.0)
        result = conditioned([2.0, 0.0, -1.0])
        
        expected = [0.5, 999.0, -1.0]  # 1/0 -> inf -> 999.0
        assert_allclose(result, expected)
    
    def test_edge_case_empty_array(self):
        """Test edge case with empty input."""
        def identity_func(x):
            return x
        
        conditioned = condition_function(identity_func)
        result = conditioned([])
        
        assert len(result) == 0
        assert isinstance(result, np.ndarray)
    
    def test_integer_input_conversion(self):
        """Test that integer inputs are converted to float arrays."""
        def identity_func(x):
            return x
        
        conditioned = condition_function(identity_func)
        result = conditioned([1, 2, 3])  # Integer list
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == float
        assert_array_equal(result, [1.0, 2.0, 3.0])
    
    def test_batch_exception_semantics(self):
        """Test that single exception affects entire batch (by design)."""
        def partially_failing_func(x):
            x = np.asarray(x)
            if x[0] == 999:  # Trigger exception based on input
                raise ValueError("Partial failure")
            return x * 2
        
        conditioned = condition_function(partially_failing_func)
        
        # Normal case works
        result1 = conditioned([1.0, 2.0])
        assert_array_equal(result1, [2.0, 4.0])
        
        # Exception case returns all NaN (intentional all-or-nothing behavior)
        result2 = conditioned([999.0, 2.0])
        assert len(result2) == 2
        assert np.all(np.isnan(result2))
    
    def test_function_returns_list(self):
        """Test wrapper handles functions that return Python lists."""
        def list_func(x):
            return x.tolist()  # Returns Python list
        
        conditioned = condition_function(list_func)
        result = conditioned([1.0, 2.0, 3.0])
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == float
        assert_array_equal(result, [1.0, 2.0, 3.0])
    
    def test_large_array_performance(self):
        """Smoke test for large arrays - no O(N²) regressions."""
        def identity_func(x):
            return x
        
        import time
        conditioned = condition_function(identity_func)
        
        large_input = np.ones(100000)  # 100k elements
        start_time = time.time()
        result = conditioned(large_input)
        elapsed = time.time() - start_time
        
        assert elapsed < 0.1  # Should complete in <100ms
        assert len(result) == 100000
        assert_array_equal(result, large_input)