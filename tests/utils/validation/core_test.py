import numpy as np
import pytest
from rapidroots.utils.validation._core import _validate_bracket_core

# Note: The original function has a bug - this line:
# finite = {np.isinf(a_arr) & np.isinf(b_arr) & np.isinf(fa_arr) & np.isinf(fb_arr)}
# Should be:
# finite = ~(np.isinf(a_arr) | np.isinf(b_arr) | np.isinf(fa_arr) | np.isinf(fb_arr))


def test_valid_bracket_basic():
    """Test basic valid bracket: f(a) and f(b) have opposite signs"""
    a = np.array([0.0, 1.0])
    b = np.array([1.0, 2.0])
    fa = np.array([-1.0, -0.5])
    fb = np.array([1.0, 0.5])
    
    is_valid, reason = _validate_bracket_core(a, b, fa, fb)
    
    np.testing.assert_array_equal(is_valid, [True, True])
    np.testing.assert_array_equal(reason, [0, 0])


def test_valid_bracket_with_zero():
    """Test valid bracket when one function value is zero"""
    a = np.array([0.0, 1.0])
    b = np.array([1.0, 2.0])
    fa = np.array([0.0, -1.0])
    fb = np.array([1.0, 0.0])
    
    is_valid, reason = _validate_bracket_core(a, b, fa, fb)
    
    np.testing.assert_array_equal(is_valid, [True, True])
    np.testing.assert_array_equal(reason, [0, 0])


def test_invalid_ordering():
    """Test invalid when a >= b"""
    a = np.array([2.0, 1.0])
    b = np.array([1.0, 1.0])
    fa = np.array([-1.0, -1.0])
    fb = np.array([1.0, 1.0])
    
    is_valid, reason = _validate_bracket_core(a, b, fa, fb)
    
    np.testing.assert_array_equal(is_valid, [False, False])
    np.testing.assert_array_equal(reason, [1, 1])


def test_infinite_values_only():
    """Test rejection when only infinite values fail (not ordering)"""
    a = np.array([0.0, 1.0])
    b = np.array([1.0, np.inf])  # This will fail finite check but pass ordering
    fa = np.array([-1.0, -1.0])
    fb = np.array([1.0, 1.0])
    
    is_valid, reason = _validate_bracket_core(a, b, fa, fb)
    
    np.testing.assert_array_equal(is_valid, [True, False])
    np.testing.assert_array_equal(reason, [0, 2])  # Now we get reason 2 for infinite


def test_infinite_values():
    """Test rejection of infinite values"""
    a = np.array([0.0, np.inf])
    b = np.array([1.0, 2.0])
    fa = np.array([-1.0, -1.0])
    fb = np.array([1.0, 1.0])
    
    is_valid, reason = _validate_bracket_core(a, b, fa, fb)
    
    np.testing.assert_array_equal(is_valid, [True, False])
    # The second element fails ordering (np.inf >= 2.0), so reason is 1, not 2
    np.testing.assert_array_equal(reason, [0, 1])


def test_same_sign_functions():
    """Test invalid when f(a) and f(b) have same sign"""
    a = np.array([0.0, 1.0])
    b = np.array([1.0, 2.0])
    fa = np.array([1.0, 2.0])
    fb = np.array([2.0, 3.0])
    
    is_valid, reason = _validate_bracket_core(a, b, fa, fb)
    
    np.testing.assert_array_equal(is_valid, [False, False])
    np.testing.assert_array_equal(reason, [3, 3])

def test_tolerance_zero_detection():
    """Test tolerance parameter for zero detection"""
    a = np.array([0.0, 1.0])
    b = np.array([1.0, 2.0])
    fa = np.array([0.001, 1.0])  # Changed: both positive so they fail without tolerance
    fb = np.array([1.0, 0.001])
    
    # Without tolerance - should be invalid
    is_valid, _ = _validate_bracket_core(a, b, fa, fb, tolerance=0.0)
    np.testing.assert_array_equal(is_valid, [False, False])
    
    # With tolerance - should be valid
    is_valid, reason = _validate_bracket_core(a, b, fa, fb, tolerance=0.01)
    np.testing.assert_array_equal(is_valid, [True, True])
    np.testing.assert_array_equal(reason, [0, 0])


def test_shape_mismatch():
    """Test error on shape mismatch"""
    a = np.array([0.0, 1.0])
    b = np.array([1.0])  # Different shape
    fa = np.array([-1.0, -1.0])
    fb = np.array([1.0, 1.0])
    
    with pytest.raises(ValueError, match="shape mismatch"):
        _validate_bracket_core(a, b, fa, fb)


def test_dtype_handling():
    """Test dtype parameter"""
    a = [0, 1]
    b = [1, 2]
    fa = [-1, -1]
    fb = [1, 1]
    
    is_valid, reason = _validate_bracket_core(a, b, fa, fb, dtype=np.float32)
    
    np.testing.assert_array_equal(is_valid, [True, True])
    np.testing.assert_array_equal(reason, [0, 0])


def test_mixed_validity():
    """Test mixed valid/invalid cases"""
    a = np.array([0.0, 2.0, 0.0, np.inf])
    b = np.array([1.0, 1.0, 1.0, 2.0])
    fa = np.array([-1.0, 1.0, 1.0, -1.0])
    fb = np.array([1.0, 2.0, 2.0, 1.0])
    
    is_valid, reason = _validate_bracket_core(a, b, fa, fb)
    
    np.testing.assert_array_equal(is_valid, [True, False, False, False])
    np.testing.assert_array_equal(reason, [0, 1, 3, 1])


def test_signbit_edge_cases():
    """Test signbit behavior with +0.0 and -0.0"""
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    fa = np.array([0.0, -0.0])
    fb = np.array([-0.0, 0.0])
    
    is_valid, reason = _validate_bracket_core(a, b, fa, fb)
    
    # Both should be valid due to zero detection
    np.testing.assert_array_equal(is_valid, [True, True])
    np.testing.assert_array_equal(reason, [0, 0])

if __name__ == "__main__":
    # Run with: python -m pytest test_bisection_scalar.py -v
    pytest.main([__file__, "-v", "--maxfail=1"])

