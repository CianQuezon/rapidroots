import numpy as np
import pytest
import sympy as sp
from numpy.testing import assert_allclose, assert_array_equal
from rapidroots.families.sympy_family import SympyFamily


class TestSympyFamily:
    """Professional test suite for SympyFamily class"""

    def test_basic_polynomial_creation(self):
        """Test basic polynomial function creation and properties"""
        x = sp.Symbol('x')
        a, b = sp.symbols('a b')
        expr = a * x**2 + b * x + 1
        
        family = SympyFamily("quadratic", expr, (a, b))
        
        assert family.name == "quadratic"
        assert family.param_count == 2

    def test_parameter_shape_validation(self):
        """Test parameter array shape validation"""
        x = sp.Symbol('x')
        a = sp.symbols('a')
        expr = a * x + 1
        
        family = SympyFamily("linear", expr, (a,))
        
        # Wrong shape should raise ValueError
        with pytest.raises(ValueError, match="params_arr must be shape"):
            family.evaluate([1.0, 2.0], [[1.0, 2.0]])  # Wrong param shape
        
        # Correct shape should work
        result = family.evaluate([1.0, 2.0], [[1.0], [2.0]])
        assert result.shape == (2,)

    def test_function_evaluation_accuracy(self):
        """Test numerical accuracy of function evaluation"""
        x = sp.Symbol('x')
        a, b = sp.symbols('a b')
        expr = a * x**2 + b
        
        family = SympyFamily("simple_quad", expr, (a, b))
        
        # Test known values
        x_vals = np.array([0.0, 1.0, 2.0])
        params = np.array([[2.0, 3.0]])  # a=2, b=3
        
        result = family.evaluate(x_vals, params)
        expected = np.array([3.0, 5.0, 11.0])  # 2*x^2 + 3
        
        assert_allclose(result, expected, rtol=1e-14)

    def test_derivative_evaluation(self):
        """Test first and second derivative evaluation"""
        x = sp.Symbol('x')
        a = sp.symbols('a')
        expr = a * x**3
        
        family = SympyFamily("cubic", expr, (a,))
        
        x_vals = np.array([1.0, 2.0])
        params = np.array([[2.0]])  # a=2
        
        # Function: 2*x^3
        f_result = family.evaluate(x_vals, params, derivative=0)
        expected_f = np.array([2.0, 16.0])
        assert_allclose(f_result, expected_f, rtol=1e-14)
        
        # First derivative: 6*x^2
        df_result = family.evaluate(x_vals, params, derivative=1)
        expected_df = np.array([6.0, 24.0])
        assert_allclose(df_result, expected_df, rtol=1e-14)
        
        # Second derivative: 12*x
        d2f_result = family.evaluate(x_vals, params, derivative=2)
        expected_d2f = np.array([12.0, 24.0])
        assert_allclose(d2f_result, expected_d2f, rtol=1e-14)

    def test_vectorized_multiple_parameter_sets(self):
        """Test vectorized evaluation with multiple parameter sets (element-wise)"""
        x = sp.Symbol('x')
        a, b = sp.symbols('a b')
        expr = a * x + b
        
        family = SympyFamily("linear", expr, (a, b))
        
        x_vals = np.array([1.0, 2.0, 3.0])
        params = np.array([
            [1.0, 0.0],  # y = x     → f(1) = 1
            [2.0, 1.0],  # y = 2x+1  → f(2) = 5  
            [0.0, 5.0]   # y = 5     → f(3) = 5
        ])
        
        result = family.evaluate(x_vals, params)
        expected = np.array([1.0, 5.0, 5.0])  # Element-wise evaluation
        
        assert_allclose(result, expected, rtol=1e-14)

    def test_transcendental_functions(self):
        """Test transcendental function evaluation"""
        x = sp.Symbol('x')
        a, w = sp.symbols('a w')
        expr = a * sp.sin(w * x)
        
        family = SympyFamily("sine_wave", expr, (a, w))
        
        x_vals = np.array([0.0, np.pi/2, np.pi])
        params = np.array([[1.0, 1.0]])  # a=1, w=1
        
        result = family.evaluate(x_vals, params)
        expected = np.array([0.0, 1.0, 0.0])  # sin(x)
        
        assert_allclose(result, expected, atol=1e-14)

    def test_dtype_consistency(self):
        """Test dtype parameter handling"""
        x = sp.Symbol('x')
        a = sp.symbols('a')
        expr = a * x**2
        
        family = SympyFamily("square", expr, (a,))
        
        x_vals = [1.0, 2.0]
        params = [[2.0]]
        
        # Test float64 (default)
        result_f64 = family.evaluate(x_vals, params, dtype=np.float64)
        assert result_f64.dtype == np.float64
        
        # Test float32
        result_f32 = family.evaluate(x_vals, params, dtype=np.float32)
        assert result_f32.dtype == np.float32
        
        # Values should be close
        assert_allclose(result_f64, result_f32, rtol=1e-6)

    def test_complex_expression_with_multiple_terms(self):
        """Test complex expression with multiple mathematical operations"""
        x = sp.Symbol('x')
        a, b, c = sp.symbols('a b c')
        expr = a * sp.exp(b * x) + c * sp.cos(x)
        
        family = SympyFamily("complex_expr", expr, (a, b, c))
        
        assert family.param_count == 3
        
        x_vals = np.array([0.0])
        params = np.array([[1.0, 1.0, 1.0]])  # a=1, b=1, c=1
        
        result = family.evaluate(x_vals, params)
        expected = 1.0 * np.exp(0.0) + 1.0 * np.cos(0.0)  # 1 + 1 = 2
        
        assert_allclose(result, [expected], rtol=1e-14)

    def test_edge_case_zero_parameters(self):
        """Test function with no parameters"""
        x = sp.Symbol('x')
        expr = x**2 + 1  # No parameters
        
        family = SympyFamily("no_params", expr, ())
        
        assert family.param_count == 0
        
        x_vals = np.array([1.0, 2.0])
        params = np.empty((2, 0))  # Empty parameter array
        
        result = family.evaluate(x_vals, params)
        expected = np.array([2.0, 5.0])  # x^2 + 1
        
        assert_allclose(result, expected, rtol=1e-14)

    def test_numba_compilation_warmup(self):
        """Test that Numba compilation works correctly"""
        x = sp.Symbol('x')
        a = sp.symbols('a')
        expr = a * x
        
        family = SympyFamily("linear_simple", expr, (a,))
        
        x_vals = np.array([1.0])
        params = np.array([[1.0]])
        
        # First call triggers Numba compilation
        result1 = family.evaluate(x_vals, params)
        
        # Second call should use compiled version
        result2 = family.evaluate(x_vals, params)
        
        # Results should be identical
        assert_array_equal(result1, result2)
        assert_allclose(result1, [1.0], rtol=1e-14)

    def test_large_array_performance(self):
        """Test performance with larger arrays"""
        x = sp.Symbol('x')
        a, b = sp.symbols('a b')
        expr = a * x**2 + b
        
        family = SympyFamily("perf_test", expr, (a, b))
        
        # Large arrays
        n = 10000
        x_vals = np.linspace(0, 10, n)
        params = np.array([[1.0, 0.0]])  # y = x^2
        
        result = family.evaluate(x_vals, params)
        
        # Verify shape and basic correctness
        assert result.shape == (n,)
        assert_allclose(result[0], 0.0, atol=1e-14)  # f(0) = 0
        assert_allclose(result[-1], 100.0, rtol=1e-12)  # f(10) = 100

    def test_invalid_derivative_order(self):
        """Test invalid derivative order handling"""
        x = sp.Symbol('x')
        a = sp.symbols('a')
        expr = a * x
        
        family = SympyFamily("linear", expr, (a,))
        
        x_vals = np.array([1.0])
        params = np.array([[1.0]])
        
        # Invalid derivative order should raise KeyError from ufuncs dict
        with pytest.raises(KeyError):
            family.evaluate(x_vals, params, derivative=3)

if __name__ == "__main__":
    # Run with: python -m pytest test_bisection_scalar.py -v
    pytest.main([__file__, "-v", "--maxfail=1"])