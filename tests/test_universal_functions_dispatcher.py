import pytest
import numpy as np
import sympy as sp
from numpy.testing import assert_allclose
from rapidroots.core.dispatcher import UniversalFunctionSympyDispatcher

class TestUniversalFunctionSympyDispatcher:
    
    @pytest.fixture
    def dispatcher(self):
        """Create a fresh dispatcher instance for each test."""
        return UniversalFunctionSympyDispatcher()
    
    @pytest.fixture
    def sample_functions(self, dispatcher):
        """Register common test functions."""
        x, a, b = sp.symbols('x a b')
        
        # Linear function: f(x) = a*x + b
        dispatcher.register_symbolic_family(
            'linear', a*x + b, (a, b)
        )
        
        # Quadratic function: f(x) = a*x^2 + b*x
        dispatcher.register_symbolic_family(
            'quadratic', a*x**2 + b*x, (a, b)
        )
        
        # Exponential function: f(x) = a*exp(b*x)
        dispatcher.register_symbolic_family(
            'exponential', a*sp.exp(b*x), (a, b)
        )
        
        return dispatcher
    
    def test_registration_basic(self, dispatcher):
        """Test basic function registration."""
        x, a = sp.symbols('x a')
        expr = a * x
        
        dispatcher.register_symbolic_family('test_func', expr, (a,))
        
        assert 'test_func' in dispatcher.compiled_functions
        assert 'f' in dispatcher.compiled_functions['test_func']
        assert 'df' in dispatcher.compiled_functions['test_func']
        assert 'd2f' in dispatcher.compiled_functions['test_func']
        assert dispatcher.compiled_functions['test_func']['param_count'] == 1
    
    @pytest.mark.parametrize("func_name,x_val,params,expected", [
        ('linear', 2.0, [3.0, 1.0], 7.0),      # f(x) = 3*x + 1, f(2) = 7
        ('quadratic', 3.0, [2.0, 1.0], 21.0),  # f(x) = 2*x² + x, f(3) = 21
        ('linear', 0.0, [5.0, -2.0], -2.0),    # f(x) = 5*x - 2, f(0) = -2
    ])
    def test_single_evaluations(self, sample_functions, func_name, x_val, params, expected):
        """Test single function evaluations with different parameters."""
        func_types = np.array([func_name])
        params_array = np.array([params])
        x_array = np.array([x_val])
        
        result = sample_functions.evaluate_batch(func_types, params_array, x_array)
        
        assert_allclose(result, [expected], rtol=1e-10)
    
    @pytest.mark.parametrize("derivative", [0, 1, 2])
    def test_derivatives_linear(self, sample_functions, derivative):
        """Test derivatives of linear function: f(x) = 2*x + 3."""
        func_types = np.array(['linear'])
        params_array = np.array([[2.0, 3.0]])
        x_array = np.array([5.0])
        
        result = sample_functions.evaluate_batch(
            func_types, params_array, x_array, derivative=derivative
        )
        
        expected = {
            0: 13.0,  # f(5) = 2*5 + 3 = 13
            1: 2.0,   # f'(x) = 2
            2: 0.0    # f''(x) = 0
        }
        
        assert_allclose(result, [expected[derivative]], rtol=1e-10)
    
    def test_batch_processing(self, sample_functions):
        """Test batch processing with mixed function types."""
        # Mix of different functions
        func_types = np.array(['linear', 'quadratic', 'linear'])
        params_array = np.array([
            [2.0, 1.0],   # 2*x + 1
            [1.0, 0.0],   # x²
            [0.5, 2.0]    # 0.5*x + 2
        ])
        x_array = np.array([1.0, 2.0, 4.0])
        
        result = sample_functions.evaluate_batch(func_types, params_array, x_array)
        
        expected = np.array([3.0, 4.0, 4.0])  # [3, 4, 4]
        assert_allclose(result, expected, rtol=1e-10)
    
    def test_chunked_processing(self, sample_functions):
        """Test chunked processing with large arrays."""
        # Large array that will trigger chunking
        n = 25000  # Larger than default chunk_size of 10000
        func_types = np.full(n, 'linear')
        params_array = np.full((n, 2), [1.0, 0.0])  # f(x) = x
        x_array = np.linspace(0, 1, n)
        
        result = sample_functions.evaluate_batch(
            func_types, params_array, x_array, chunk_size=5000
        )
        
        # For f(x) = x, result should equal x_array
        assert_allclose(result, x_array, rtol=1e-10)
        assert len(result) == n
    
    def test_exponential_function(self, sample_functions):
        """Test exponential function evaluation and derivatives."""
        func_types = np.array(['exponential'])
        params_array = np.array([[1.0, 1.0]])  # f(x) = e^x
        x_array = np.array([0.0])
        
        # Test function value: e^0 = 1
        result_f = sample_functions.evaluate_batch(
            func_types, params_array, x_array, derivative=0
        )
        assert_allclose(result_f, [1.0], rtol=1e-10)
        
        # Test first derivative: d/dx(e^x) = e^x, at x=0 = 1
        result_df = sample_functions.evaluate_batch(
            func_types, params_array, x_array, derivative=1
        )
        assert_allclose(result_df, [1.0], rtol=1e-10)
    
    def test_empty_input_handling(self, sample_functions):
        """Test handling of empty input arrays."""
        func_types = np.array([])
        params_array = np.empty((0, 2))
        x_array = np.array([])
        
        result = sample_functions.evaluate_batch(func_types, params_array, x_array)
        
        assert len(result) == 0
        assert result.dtype == np.float64
    
    def test_error_handling_invalid_derivative(self, sample_functions):
        """Test error handling for invalid derivative order."""
        func_types = np.array(['linear'])
        params_array = np.array([[1.0, 0.0]])
        x_array = np.array([1.0])
        
        with pytest.raises(KeyError):
            sample_functions.evaluate_batch(
                func_types, params_array, x_array, derivative=3
            )
    
    def test_performance_consistency(self, sample_functions):
        """Test that results are consistent regardless of chunk size."""
        n = 12000
        func_types = np.full(n, 'quadratic')
        params_array = np.full((n, 2), [2.0, 1.0])  # f(x) = 2x² + x
        x_array = np.linspace(-5, 5, n)
        
        # Test with different chunk sizes
        result1 = sample_functions.evaluate_batch(
            func_types, params_array, x_array, chunk_size=1000
        )
        result2 = sample_functions.evaluate_batch(
            func_types, params_array, x_array, chunk_size=5000
        )
        
        assert_allclose(result1, result2, rtol=1e-14)
    
    @pytest.mark.parametrize("chunk_size", [1, 100, 1000, 10000])
    def test_chunk_size_variations(self, sample_functions, chunk_size):
        """Test different chunk sizes produce identical results."""
        n = 2000
        func_types = np.full(n, 'linear')
        params_array = np.full((n, 2), [3.0, -1.0])
        x_array = np.arange(n, dtype=float)
        
        result = sample_functions.evaluate_batch(
            func_types, params_array, x_array, chunk_size=chunk_size
        )
        
        # Expected: f(x) = 3x - 1
        expected = 3.0 * x_array - 1.0
        assert_allclose(result, expected, rtol=1e-10)


# Additional integration test
def test_full_workflow():
    """Integration test of the complete workflow."""
    dispatcher = UniversalFunctionSympyDispatcher()
    
    # Register a trigonometric function
    x, a, b = sp.symbols('x a b')
    dispatcher.register_symbolic_family('sinusoid', a * sp.sin(b * x), (a, b))
    
    # Test with multiple periods
    func_types = np.array(['sinusoid'] * 100)
    params_array = np.full((100, 2), [2.0, np.pi])  # f(x) = 2*sin(π*x)
    x_array = np.linspace(0, 2, 100)
    
    result = dispatcher.evaluate_batch(func_types, params_array, x_array)
    
    # At x=0.5: sin(π*0.5) = sin(π/2) = 1, so f(0.5) = 2*1 = 2
    # At x=1.5: sin(π*1.5) = sin(3π/2) = -1, so f(1.5) = 2*(-1) = -2
    closest_to_half = np.argmin(np.abs(x_array - 0.5))
    closest_to_one_half = np.argmin(np.abs(x_array - 1.5))
    
    assert_allclose(result[closest_to_half], 2.0, rtol=0.1)
    assert_allclose(result[closest_to_one_half], -2.0, rtol=0.1)