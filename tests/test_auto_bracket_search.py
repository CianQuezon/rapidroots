import pytest
import numpy as np
import sympy as sp
from numpy.testing import assert_allclose, assert_array_equal
from unittest.mock import Mock, patch

# Assuming the imports from your actual module
from rapidroots.core.dispatcher import UniversalFunctionSympyDispatcher
from rapidroots.utils.utils import _VectorBracketFinder  # Replace with actual import


class TestVectorBracketFinder:
    
    @pytest.fixture
    def mock_dispatcher(self):
        """Create a mock dispatcher for controlled testing."""
        dispatcher = Mock(spec=UniversalFunctionSympyDispatcher)
        return dispatcher
    
    @pytest.fixture
    def real_dispatcher(self):
        """Create real dispatcher with test functions."""
        dispatcher = UniversalFunctionSympyDispatcher()
        x, a, b = sp.symbols('x a b')
        
        # Simple linear: f(x) = a*x + b
        dispatcher.register_symbolic_family('linear', a*x + b, (a, b))
        
        # Quadratic with known roots: f(x) = (x-2)(x-5) = x^2 - 7x + 10
        dispatcher.register_symbolic_family('quadratic', x**2 - 7*x + 10, ())
        
        # Simple sine: f(x) = sin(x) (has roots at multiples of π)
        dispatcher.register_symbolic_family('sine', sp.sin(x), ())
        
        return dispatcher
    
    def test_basic_bracket_finding_linear(self, real_dispatcher):
        """Test basic bracket finding for linear function with known root."""
        # f(x) = 2*x - 4, root at x = 2
        func_types = np.array(['linear'])
        params_array = np.array([[2.0, -4.0]])
        initial_guess = np.array([0.0])
        
        left, right = _VectorBracketFinder.auto_bracket_search(
            real_dispatcher, func_types, params_array, initial_guess
        )
        
        # Verify we found valid brackets
        assert len(left) == 1
        assert len(right) == 1
        assert left[0] < 2.0 < right[0]  # Root should be between brackets
        
        # Verify sign change across bracket
        f_left = real_dispatcher.evaluate_batch(func_types, params_array, left)
        f_right = real_dispatcher.evaluate_batch(func_types, params_array, right)
        assert f_left[0] * f_right[0] < 0  # Opposite signs
    
    def test_multiple_points_same_function(self, real_dispatcher):
        """Test bracket finding for multiple initial points with same function."""
        # f(x) = 2*x - 4, root at x = 2
        n_points = 3
        func_types = np.full(n_points, 'linear')
        params_array = np.full((n_points, 2), [2.0, -4.0])
        initial_guess = np.array([-1.0, 0.0, 5.0])
        
        left, right = _VectorBracketFinder.auto_bracket_search(
            real_dispatcher, func_types, params_array, initial_guess
        )
        
        assert len(left) == n_points
        assert len(right) == n_points
        
        # All should bracket the root at x=2
        for i in range(n_points):
            assert left[i] < 2.0 < right[i]
    
    def test_quadratic_with_two_roots(self, real_dispatcher):
        """Test bracket finding for quadratic with two roots."""
        # f(x) = x^2 - 7x + 10 = (x-2)(x-5), roots at x=2 and x=5
        func_types = np.array(['quadratic', 'quadratic'])
        params_array = np.empty((2, 0))  # No parameters for this function
        initial_guess = np.array([1.0, 6.0])  # Near each root
        
        left, right = _VectorBracketFinder.auto_bracket_search(
            real_dispatcher, func_types, params_array, initial_guess
        )
        
        # Should find brackets around the nearest roots
        assert left[0] < 2.0 < right[0]  # First point brackets root at 2
        assert left[1] < 5.0 < right[1]  # Second point brackets root at 5
    
    @pytest.mark.parametrize("initial_guess,expected_shape", [
        (0.0, ()),                    # Scalar
        ([1.0, 2.0], (2,)),          # 1D array
        ([[1.0, 2.0], [3.0, 4.0]], (2, 2)),  # 2D array
    ])
    def test_shape_preservation(self, mock_dispatcher, initial_guess, expected_shape):
        """Test that output shapes match input shapes."""
        initial_guess = np.array(initial_guess)
        flat_size = initial_guess.size
        
        # Mock function that returns sign-changing values
        mock_dispatcher.evaluate_batch.side_effect = [
            np.ones(flat_size),    # f0 (positive)
            -np.ones(flat_size),   # fa (negative) - creates ab bracket
            np.ones(flat_size)     # fb (positive)
        ]
        
        func_types = np.full(flat_size, 'test')
        params_array = np.zeros((flat_size, 1))
        
        left, right = _VectorBracketFinder.auto_bracket_search(
            mock_dispatcher, func_types, params_array, initial_guess
        )
        
        assert left.shape == expected_shape
        assert right.shape == expected_shape
    
    def test_concat_threshold_behavior(self, mock_dispatcher):
        """Test behavior with different concat_threshold values."""
        n = 5
        func_types = np.full(n, 'test')
        params_array = np.zeros((n, 1))
        initial_guess = np.zeros(n)
        
        # Mock to create ab brackets (sign change between a and b)
        mock_dispatcher.evaluate_batch.side_effect = [
            np.ones(n),     # f0
            -np.ones(n),    # fa (when single call)
            np.ones(n),     # fb (when single call)
            np.concatenate([-np.ones(n), np.ones(n)])  # combined call
        ]
        
        # Test with threshold higher than n (should use separate calls)
        left1, right1 = _VectorBracketFinder.auto_bracket_search(
            mock_dispatcher, func_types, params_array, initial_guess,
            concat_threshold=10
        )
        
        # Reset mock
        mock_dispatcher.reset_mock()
        mock_dispatcher.evaluate_batch.side_effect = [
            np.ones(n),     # f0
            np.concatenate([-np.ones(n), np.ones(n)])  # combined call
        ]
        
        # Test with threshold lower than n (should use combined call)
        left2, right2 = _VectorBracketFinder.auto_bracket_search(
            mock_dispatcher, func_types, params_array, initial_guess,
            concat_threshold=3
        )
        
        # Results should be identical regardless of threshold
        assert_allclose(left1, left2)
        assert_allclose(right1, right2)
    
    def test_expansion_parameters(self, mock_dispatcher):
        """Test different expansion parameters."""
        func_types = np.array(['test'])
        params_array = np.zeros((1, 1))
        initial_guess = np.array([0.0])
        
        # Mock that requires multiple iterations
        mock_dispatcher.evaluate_batch.side_effect = [
            np.array([1.0]),   # f0
            np.array([1.0]),   # fa (iteration 1, no bracket)
            np.array([1.0]),   # fb (iteration 1, no bracket)
            np.array([-1.0]),  # fa (iteration 2, creates bracket)
            np.array([1.0])    # fb (iteration 2)
        ]
        
        left, right = _VectorBracketFinder.auto_bracket_search(
            mock_dispatcher, func_types, params_array, initial_guess,
            base_step=0.5,
            expansion_factor=2.0
        )
        
        # Should find brackets with custom parameters
        assert len(left) == 1
        assert len(right) == 1
        
        # Verify multiple calls were made (due to expansion)
        assert mock_dispatcher.evaluate_batch.call_count > 2
    
    def test_max_range_limiting(self, mock_dispatcher):
        """Test that max_range parameter limits search."""
        func_types = np.array(['test'])
        params_array = np.zeros((1, 1))
        initial_guess = np.array([0.0])
        
        # Mock that never finds a bracket
        mock_dispatcher.evaluate_batch.return_value = np.array([1.0])
        
        with pytest.raises(ValueError, match="Failed to bracket"):
            _VectorBracketFinder.auto_bracket_search(
                mock_dispatcher, func_types, params_array, initial_guess,
                max_range=0.5,  # Very restrictive
                max_iterations=3
            )
    
    def test_input_validation(self, mock_dispatcher):
        """Test input validation and error handling."""
        # Mismatched array sizes
        func_types = np.array(['test', 'test'])
        params_array = np.zeros((1, 1))  # Wrong size
        initial_guess = np.array([0.0])
        
        with pytest.raises(ValueError, match="func_types and params_array must match"):
            _VectorBracketFinder.auto_bracket_search(
                mock_dispatcher, func_types, params_array, initial_guess
            )
    
    def test_edge_case_zero_guess(self, real_dispatcher):
        """Test bracket finding with zero as initial guess."""
        # f(x) = x - 1, root at x = 1
        func_types = np.array(['linear'])
        params_array = np.array([[1.0, -1.0]])
        initial_guess = np.array([0.0])  # Exactly at a function value
        
        left, right = _VectorBracketFinder.auto_bracket_search(
            real_dispatcher, func_types, params_array, initial_guess
        )
        
        # Should still find valid brackets
        assert left[0] < 1.0 < right[0]
    
    def test_empty_input(self, mock_dispatcher):
        """Test handling of empty input arrays."""
        func_types = np.array([])
        params_array = np.empty((0, 1))
        initial_guess = np.array([])
        
        mock_dispatcher.evaluate_batch.return_value = np.array([])
        
        left, right = _VectorBracketFinder.auto_bracket_search(
            mock_dispatcher, func_types, params_array, initial_guess
        )
        
        assert len(left) == 0
        assert len(right) == 0
    
    @pytest.mark.parametrize("bracket_type", ['ab', 'oa', 'ob'])
    def test_different_bracket_types(self, mock_dispatcher, bracket_type):
        """Test detection of different bracket types."""
        func_types = np.array(['test'])
        params_array = np.zeros((1, 1))
        initial_guess = np.array([0.0])
        
        if bracket_type == 'ab':
            # f(a) and f(b) have opposite signs
            mock_dispatcher.evaluate_batch.side_effect = [
                np.array([1.0]),   # f0
                np.array([-1.0]),  # fa
                np.array([1.0])    # fb
            ]
        elif bracket_type == 'oa':
            # f0 and f(a) have opposite signs
            mock_dispatcher.evaluate_batch.side_effect = [
                np.array([1.0]),   # f0
                np.array([-1.0]),  # fa
                np.array([1.0])    # fb (same sign as f0)
            ]
        elif bracket_type == 'ob':
            # f0 and f(b) have opposite signs
            mock_dispatcher.evaluate_batch.side_effect = [
                np.array([1.0]),   # f0
                np.array([1.0]),   # fa (same sign as f0)
                np.array([-1.0])   # fb
            ]
        
        left, right = _VectorBracketFinder.auto_bracket_search(
            mock_dispatcher, func_types, params_array, initial_guess
        )
        
        assert len(left) == 1
        assert len(right) == 1
    
    def test_chunk_size_parameter(self, real_dispatcher):
        """Test that chunk_size parameter is passed correctly."""
        func_types = np.array(['linear'])
        params_array = np.array([[1.0, -2.0]])
        initial_guess = np.array([0.0])
        
        # This mainly tests that the parameter is accepted
        left, right = _VectorBracketFinder.auto_bracket_search(
            real_dispatcher, func_types, params_array, initial_guess,
            chunk_size=1000
        )
        
        assert len(left) == 1
        assert len(right) == 1
    
    def test_performance_with_large_arrays(self, mock_dispatcher):
        """Test performance with larger arrays."""
        n = 1000
        func_types = np.full(n, 'test')
        params_array = np.zeros((n, 1))
        initial_guess = np.zeros(n)
        
        # Mock to immediately find brackets
        mock_dispatcher.evaluate_batch.side_effect = [
            np.ones(n),      # f0
            -np.ones(n),     # fa
            np.ones(n)       # fb
        ]
        
        left, right = _VectorBracketFinder.auto_bracket_search(
            mock_dispatcher, func_types, params_array, initial_guess
        )
        
        assert len(left) == n
        assert len(right) == n
        # Should complete quickly with immediate bracketing


# Integration test
class TestBracketFinderIntegration:
    """Integration tests with real mathematical functions."""
    
    def test_sin_function_brackets(self):
        """Test bracket finding for sine function."""
        dispatcher = UniversalFunctionSympyDispatcher()
        x = sp.Symbol('x')
        dispatcher.register_symbolic_family('sine', sp.sin(x), ())
        
        # Test points near π (where sin(x) = 0)
        func_types = np.array(['sine'])
        params_array = np.empty((1, 0))
        initial_guess = np.array([3.0])  # Near π
        
        left, right = _VectorBracketFinder.auto_bracket_search(
            dispatcher, func_types, params_array, initial_guess
        )
        
        # Should bracket the root near π
        assert left[0] < np.pi < right[0]
        
        # Verify sign change
        f_left = dispatcher.evaluate_batch(func_types, params_array, left)
        f_right = dispatcher.evaluate_batch(func_types, params_array, right)
        assert f_left[0] * f_right[0] < 0
    
    def test_polynomial_multiple_roots(self):
        """Test with polynomial having multiple roots."""
        dispatcher = UniversalFunctionSympyDispatcher()
        x = sp.Symbol('x')
        # f(x) = x(x-1)(x-2) = x³ - 3x² + 2x (roots at 0, 1, 2)
        dispatcher.register_symbolic_family('cubic', x**3 - 3*x**2 + 2*x, ())
        
        func_types = np.array(['cubic', 'cubic', 'cubic'])
        params_array = np.empty((3, 0))
        initial_guess = np.array([-0.5, 0.5, 1.5])  # Near each root
        
        left, right = _VectorBracketFinder.auto_bracket_search(
            dispatcher, func_types, params_array, initial_guess
        )
        
        # Should bracket SOME root for each initial guess (not necessarily nearest)
        all_roots = [0.0, 1.0, 2.0]
        for i in range(len(initial_guess)):
            # Verify we found a valid bracket (sign change)
            f_left = dispatcher.evaluate_batch(
                func_types[i:i+1], params_array[i:i+1], left[i:i+1]
            )
            f_right = dispatcher.evaluate_batch(
                func_types[i:i+1], params_array[i:i+1], right[i:i+1]
            )
            assert f_left[0] * f_right[0] < 0, f"No sign change in bracket {i}"
            
            # Check that the bracket contains at least one of the known roots
            bracket_contains_root = any(
                left[i] < root < right[i] for root in all_roots
            )
            assert bracket_contains_root, f"Bracket {i} [{left[i]}, {right[i]}] contains no known root"
