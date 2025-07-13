import pytest
import numpy as np
from unittest.mock import Mock
from numpy.testing import assert_array_equal
from rapidroots.utils.function_utils import validate_bracket  # Replace with actual import


class TestValidateBracket:
    
    @pytest.fixture
    def mock_dispatcher(self):
        """Create mock dispatcher for controlled testing."""
        mock = Mock()
        return mock
    
    def test_valid_bracket_opposite_signs(self, mock_dispatcher):
        """Test basic valid bracket with opposite signs."""
        # f(a) = -1, f(b) = 1 → valid bracket
        mock_dispatcher.evaluate_batch.side_effect = [
            np.array([-1.0]),  # f(a)
            np.array([1.0])    # f(b)
        ]
        
        is_valid, messages = validate_bracket(
            mock_dispatcher,
            np.array(['test']),
            np.array([[1.0]]),
            np.array([0.0]),
            np.array([1.0])
        )
        
        assert is_valid[0] == True
        assert messages[0] == "Valid bracket"
    
    def test_invalid_bracket_same_signs(self, mock_dispatcher):
        """Test invalid bracket with same signs."""
        # f(a) = 1, f(b) = 2 → invalid bracket (both positive)
        mock_dispatcher.evaluate_batch.side_effect = [
            np.array([1.0]),   # f(a)
            np.array([2.0])    # f(b)
        ]
        
        is_valid, messages = validate_bracket(
            mock_dispatcher,
            np.array(['test']),
            np.array([[1.0]]),
            np.array([0.0]),
            np.array([1.0])
        )
        
        assert is_valid[0] == False
        assert "No sign change" in messages[0]
        assert "f(0)" in messages[0] and "f(1)" in messages[0]
    
    def test_valid_bracket_with_zero(self, mock_dispatcher):
        """Test valid bracket when one endpoint is exactly zero."""
        # f(a) = 0, f(b) = 1 → valid bracket
        mock_dispatcher.evaluate_batch.side_effect = [
            np.array([0.0]),   # f(a)
            np.array([1.0])    # f(b)
        ]
        
        is_valid, messages = validate_bracket(
            mock_dispatcher,
            np.array(['test']),
            np.array([[1.0]]),
            np.array([0.0]),
            np.array([1.0])
        )
        
        assert is_valid[0] == True
        assert messages[0] == "Valid bracket"
    
    def test_invalid_bracket_non_finite_values(self, mock_dispatcher):
        """Test invalid bracket with NaN/inf values."""
        # f(a) = nan, f(b) = 1 → invalid bracket
        mock_dispatcher.evaluate_batch.side_effect = [
            np.array([np.nan]), # f(a)
            np.array([1.0])     # f(b)
        ]
        
        is_valid, messages = validate_bracket(
            mock_dispatcher,
            np.array(['test']),
            np.array([[1.0]]),
            np.array([0.0]),
            np.array([1.0])
        )
        
        assert is_valid[0] == False
        assert messages[0] == "Function values at endpoints are not finite"
    
    def test_multiple_brackets_mixed_validity(self, mock_dispatcher):
        """Test multiple brackets with mixed validity."""
        # First bracket: valid (-1, 1)
        # Second bracket: invalid (2, 3) 
        # Third bracket: valid with zero (0, -2)
        mock_dispatcher.evaluate_batch.side_effect = [
            np.array([-1.0, 2.0, 0.0]),  # f(a) values
            np.array([1.0, 3.0, -2.0])   # f(b) values
        ]
        
        is_valid, messages = validate_bracket(
            mock_dispatcher,
            np.array(['test1', 'test2', 'test3']),
            np.array([[1.0], [1.0], [1.0]]),
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 2.0, 3.0])
        )
        
        expected_valid = [True, False, True]
        assert_array_equal(is_valid, expected_valid)
        assert messages[0] == "Valid bracket"
        assert "No sign change" in messages[1]
        assert messages[2] == "Valid bracket"
    
    def test_subnormal_numbers_handling(self, mock_dispatcher):
        """Test bracket validation with very small subnormal numbers."""
        # Test the fix for subnormal multiplication underflow
        tiny_positive = 1e-308  # Near machine epsilon
        tiny_negative = -1e-308
        
        mock_dispatcher.evaluate_batch.side_effect = [
            np.array([tiny_negative]),  # f(a)
            np.array([tiny_positive])   # f(b)
        ]
        
        is_valid, messages = validate_bracket(
            mock_dispatcher,
            np.array(['test']),
            np.array([[1.0]]),
            np.array([0.0]),
            np.array([1.0])
        )
        
        # Should detect sign change despite tiny values
        assert is_valid[0] == True
        assert messages[0] == "Valid bracket"
    
    def test_dispatcher_exception_handling(self, mock_dispatcher):
        """Test graceful handling of dispatcher evaluation failures."""
        mock_dispatcher.evaluate_batch.side_effect = RuntimeError("Evaluation failed")
        
        is_valid, messages = validate_bracket(
            mock_dispatcher,
            np.array(['test']),
            np.array([[1.0]]),
            np.array([0.0]),
            np.array([1.0])
        )
        
        assert is_valid[0] == False
        assert "Function evaluation failed" in messages[0]
    
    def test_derivative_parameter(self, mock_dispatcher):
        """Test validation works with different derivative orders."""
        mock_dispatcher.evaluate_batch.side_effect = [
            np.array([-2.0]),  # f'(a)
            np.array([3.0])    # f'(b)
        ]
        
        is_valid, messages = validate_bracket(
            mock_dispatcher,
            np.array(['test']),
            np.array([[1.0]]),
            np.array([0.0]),
            np.array([1.0]),
            derivative=1  # Test first derivative
        )
        
        assert is_valid[0] == True
        # Verify dispatcher was called with correct derivative parameter
        calls = mock_dispatcher.evaluate_batch.call_args_list
        assert calls[0].kwargs['derivative'] == 1
        assert calls[1].kwargs['derivative'] == 1
    
    def test_chunk_size_parameter(self, mock_dispatcher):
        """Test chunk_size parameter is passed correctly."""
        mock_dispatcher.evaluate_batch.side_effect = [
            np.array([-1.0]),
            np.array([1.0])
        ]
        
        validate_bracket(
            mock_dispatcher,
            np.array(['test']),
            np.array([[1.0]]),
            np.array([0.0]),
            np.array([1.0]),
            chunk_size=5000
        )
        
        # Verify chunk_size was passed to dispatcher
        calls = mock_dispatcher.evaluate_batch.call_args_list
        assert calls[0].kwargs['chunk_size'] == 5000
        assert calls[1].kwargs['chunk_size'] == 5000
    
    def test_auto_chunk_size_selection(self, mock_dispatcher):
        """Test automatic chunk size selection for different array sizes."""
        mock_dispatcher.evaluate_batch.side_effect = [
            np.array([-1.0]),
            np.array([1.0])
        ]
        
        # Small array should use array size as chunk_size
        validate_bracket(
            mock_dispatcher,
            np.array(['test']),
            np.array([[1.0]]),
            np.array([0.0]),
            np.array([1.0]),
            chunk_size=None  # Auto-select
        )
        
        # Should work without error (exact chunk_size depends on implementation)
        assert mock_dispatcher.evaluate_batch.called
    
    def test_edge_case_both_zeros(self, mock_dispatcher):
        """Test edge case where both endpoints are zero."""
        mock_dispatcher.evaluate_batch.side_effect = [
            np.array([0.0]),   # f(a)
            np.array([0.0])    # f(b)
        ]
        
        is_valid, messages = validate_bracket(
            mock_dispatcher,
            np.array(['test']),
            np.array([[1.0]]),
            np.array([0.0]),
            np.array([1.0])
        )
        
        # Both zeros should be considered valid (has_zero condition)
        assert is_valid[0] == True
        assert messages[0] == "Valid bracket"
    
    def test_input_array_conversion(self, mock_dispatcher):
        """Test that inputs are properly converted to arrays."""
        mock_dispatcher.evaluate_batch.side_effect = [
            np.array([-1.0]),
            np.array([1.0])
        ]
        
        # Pass scalar inputs - should be converted to arrays
        is_valid, messages = validate_bracket(
            mock_dispatcher,
            np.array(['test']),
            np.array([[1.0]]),
            0.0,  # Scalar
            1.0   # Scalar
        )
        
        assert is_valid[0] == True
        assert len(is_valid) == 1
        assert len(messages) == 1