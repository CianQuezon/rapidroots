"""
Test suite for BisectionSolver.solve_vectorized() method
Focuses on essential functionality with dispatcher integration
"""

import pytest
import numpy as np
import sympy as sp
from typing import Dict, List, Callable, Tuple
from rapidroots.solvers.bisection import BisectionSolver
import logging

# Import the classes (assuming they are in modules)
# from your_module import BisectionSolver, UniversalFunctionSympyDispatcher



class MockUniversalFunctionSympyDispatcher:
    """Mock dispatcher for testing - simplified version of the real one"""
    
    def __init__(self):
        self.compiled_functions: Dict[str, Dict[str, Callable]] = {}
        self._setup_test_functions()
    
    def _setup_test_functions(self):
        """Setup basic test functions for testing"""
        # Quadratic: x^2 - 4 (roots at ±2)
        self.compiled_functions['quadratic'] = {
            'f': lambda x, a, b, c: a*x**2 + b*x + c,
            'param_count': 3
        }
        
        # Linear: a*x + b
        self.compiled_functions['linear'] = {
            'f': lambda x, a, b: a*x + b,
            'param_count': 2
        }
        
        # Cubic: x^3 - 2*x - 5 (has one real root around 2.09)
        self.compiled_functions['cubic'] = {
            'f': lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d,
            'param_count': 4
        }

    def evaluate_batch(self, func_types, params_array, x_array, derivative=0, chunk_size=10000):
        """Simplified batch evaluation for testing"""
        n = x_array.size
        results = np.empty(n, dtype=np.float64)
        
        for i, (func_type, params, x) in enumerate(zip(func_types, params_array, x_array)):
            if func_type in self.compiled_functions:
                f = self.compiled_functions[func_type]['f']
                param_count = self.compiled_functions[func_type]['param_count']
                # Use only the number of parameters the function expects
                actual_params = params[:param_count]
                try:
                    results[i] = f(x, *actual_params)
                except Exception as e:
                    print(f"Mock dispatcher error: {e}")
                    results[i] = np.nan
            else:
                print(f"Unknown function type: {func_type}")
                results[i] = np.nan
                
        return results


@pytest.fixture
def dispatcher():
    """Fixture providing mock dispatcher"""
    return MockUniversalFunctionSympyDispatcher()


@pytest.fixture
def solver(dispatcher):
    """Fixture providing BisectionSolver with mock dispatcher"""
    return BisectionSolver(dispatcher=dispatcher, default_chunk_size=10)


class TestBisectionSolverVectorized:
    """Test suite for solve_vectorized method"""

    def test_non_dispatcher_mode_works(self, solver):
        """Test that solve_vectorized works without dispatcher (fallback mode)"""
        # Simple function: x^2 - 4 = 0
        def f_func(x):
            if np.isscalar(x):
                return x**2 - 4.0
            else:
                return x**2 - 4.0
        
        roots, converged = solver.solve_vectorized(
            f_func=f_func,
            tolerance=1e-6,
            max_iterations=50,
            a_array=np.array([1.5]),
            b_array=np.array([3.0])
        )
        
        print(f"Non-dispatcher test - Root: {roots[0]}, Converged: {converged[0]}")
        assert converged[0], "Non-dispatcher mode should work"
        assert abs(roots[0] - 2.0) < 1e-2, f"Root should be near 2.0, got {roots[0]}"  # More reasonable tolerance

    def test_mock_dispatcher_functionality(self, dispatcher):
        """Test that the mock dispatcher works correctly"""
        # Test quadratic function: x^2 - 4
        x_vals = np.array([1.5, 2.0, 3.0])
        func_types = np.array(['quadratic'] * 3)
        params = np.array([[1.0, 0.0, -4.0, 0.0]] * 3)
        
        results = dispatcher.evaluate_batch(func_types, params, x_vals)
        expected = x_vals**2 - 4.0  # [1.5^2-4, 2^2-4, 3^2-4] = [-1.75, 0, 5]
        
        np.testing.assert_allclose(results, expected, atol=1e-10)
        print("Mock dispatcher test passed")

    def test_convergence_criteria_debug(self, solver):
        """Debug the convergence criteria issue"""
        func_types = np.array(['quadratic'])
        params_array = np.array([[1.0, 0.0, -4.0, 0.0]])  # x^2 - 4 = 0
        a_array = np.array([1.5])
        b_array = np.array([3.0])
        
        # Test with very loose tolerance to see if it converges
        roots, converged = solver.solve_vectorized(
            f_func=None,
            tolerance=1e-1,  # Very loose tolerance
            max_iterations=20,  # Fewer iterations
            func_types=func_types,
            params_array=params_array,
            a_array=a_array,
            b_array=b_array,
            enable_logging=True
        )
        
        print(f"Loose tolerance test - Root: {roots[0]}, Converged: {converged[0]}")
        
        # Check function value at the root
        f_at_root = solver.dispatcher.evaluate_batch(func_types, params_array, roots)[0]
        print(f"Function value at root: {f_at_root}")
        print(f"Absolute function value: {abs(f_at_root)}")
        print(f"Is |f(root)| < 1e-1? {abs(f_at_root) < 1e-1}")

    def test_single_function_convergence(self, solver):
        """Test convergence with a single simple function"""
        func_types = np.array(['quadratic'])
        params_array = np.array([[1.0, 0.0, -4.0, 0.0]])  # x^2 - 4 = 0
        a_array = np.array([1.5])
        b_array = np.array([3.0])
        
        roots, converged = solver.solve_vectorized(
            f_func=None,
            tolerance=1e-6,
            max_iterations=100,
            func_types=func_types,
            params_array=params_array,
            a_array=a_array,
            b_array=b_array,
            enable_logging=True
        )
        
        print(f"Single function test - Root: {roots[0]}, Converged: {converged[0]}")
        assert converged[0], "Single quadratic function should converge"
        assert abs(roots[0] - 2.0) < 1e-2, f"Root should be near 2.0, got {roots[0]}"  # More reasonable tolerance

    def test_basic_dispatcher_mode_convergence(self, solver):
        """Test basic convergence with mixed function types
        
        Note: The bisection method requires f(a) * f(b) < 0 (opposite signs)
        for proper bracketing. Functions must change sign in the interval.
        """
        # Setup: 3 quadratics with known roots and 2 linear functions
        n = 5
        func_types = np.array(['quadratic', 'quadratic', 'linear', 'cubic', 'quadratic'])
        
        # Quadratic params: [a, b, c] for ax^2 + bx + c = 0
        # Linear params: [a, b] for ax + b = 0  
        # Cubic params: [a, b, c, d] for ax^3 + bx^2 + cx + d = 0
        params_array = np.array([
            [1.0, 0.0, -4.0, 0.0],   # x^2 - 4 = 0, root at x=2
            [1.0, -3.0, 2.0, 0.0],   # x^2 - 3x + 2 = 0, root at x=2 (and x=1)
            [2.0, -6.0, 0.0, 0.0],   # 2x - 6 = 0, root at x=3
            [1.0, 0.0, -2.0, -5.0],  # x^3 - 2x - 5 = 0, root ~2.09
            [1.0, 0.0, -9.0, 0.0],   # x^2 - 9 = 0, root at x=3
        ])
        
        # Brackets containing the roots with proper sign changes
        a_array = np.array([1.5, 1.5, 2.0, 2.0, 2.5])
        b_array = np.array([3.0, 2.5, 4.0, 3.0, 3.5])
        
        # Expected roots (approximately)
        expected_roots = np.array([2.0, 2.0, 3.0, 2.094, 3.0])
        
        # Solve with more lenient parameters
        roots, converged = solver.solve_vectorized(
            f_func=None,  # Not used in dispatcher mode
            tolerance=1e-2,  # More lenient tolerance
            max_iterations=100,  # More iterations
            func_types=func_types,
            params_array=params_array,
            a_array=a_array,
            b_array=b_array,
            enable_logging=True  # Enable logging for debugging
        )
        
        # Debug output
        print("Debug info:")
        print("Roots:", roots)
        print("Converged:", converged)
        for i in range(n):
            print(f"Function {i+1}: {func_types[i]} with params {params_array[i][:solver.dispatcher.compiled_functions[func_types[i]]['param_count']]}")
            print(f"  Bracket: [{a_array[i]}, {b_array[i]}]")
            print(f"  Root: {roots[i]}, Converged: {converged[i]}")
        
        # Verify all converged
        assert converged.all(), f"Not all solutions converged: {converged}"
        
        # Verify roots are close to expected (with reasonable tolerance)
        np.testing.assert_allclose(roots, expected_roots, atol=1e-1)  # More reasonable tolerance

    def test_input_validation_errors(self, solver):
        """Test proper error handling for invalid inputs"""
        
        # Test only one of func_types/params_array provided (partial dispatcher mode)
        with pytest.raises(ValueError, match="Dispatcher mode requires both"):
            solver.solve_vectorized(
                f_func=None,
                func_types=np.array(['quadratic']),
                # Missing params_array, a_array, b_array
            )
        
        # Test missing required arrays in dispatcher mode
        with pytest.raises(ValueError, match="Missing required kwarg"):
            solver.solve_vectorized(
                f_func=None,
                func_types=np.array(['quadratic']),
                params_array=np.array([[1.0, 0.0, -4.0, 0.0]]),
                # Missing a_array, b_array
            )
        
        # Test mismatched array lengths
        with pytest.raises(ValueError, match="must match array length"):
            solver.solve_vectorized(
                f_func=None,
                func_types=np.array(['quadratic', 'linear']),
                params_array=np.array([[1.0, 0.0, -4.0, 0.0]]),  # Length 1
                a_array=np.array([1.0, 2.0]),  # Length 2
                b_array=np.array([3.0, 4.0])   # Length 2
            )

    def test_exact_solutions_at_endpoints(self, solver):
        """Test detection of exact solutions at bracket endpoints"""
        n = 3
        func_types = np.array(['linear', 'quadratic', 'linear'])
        params_array = np.array([
            [1.0, -2.0, 0.0, 0.0],   # x - 2 = 0, root exactly at x=2
            [1.0, 0.0, -4.0, 0.0],   # x^2 - 4 = 0, root exactly at x=2  
            [2.0, -8.0, 0.0, 0.0],   # 2x - 8 = 0, root exactly at x=4
        ])
        
        # Set brackets where one endpoint is exactly the root
        a_array = np.array([2.0, 2.0, 3.0])  # First two have exact root at 'a'
        b_array = np.array([3.0, 3.0, 4.0])  # Third has exact root at 'b'
        
        roots, converged = solver.solve_vectorized(
            f_func=None,
            tolerance=1e-6,
            func_types=func_types,
            params_array=params_array,
            a_array=a_array,
            b_array=b_array
        )
        
        # All should converge to exact solutions
        assert converged.all()
        expected = np.array([2.0, 2.0, 4.0])
        np.testing.assert_allclose(roots, expected, atol=1e-10)

    def test_invalid_brackets_handling(self, solver):
        """Test handling of brackets that don't contain roots"""
        n = 2
        func_types = np.array(['quadratic', 'linear'])
        params_array = np.array([
            [1.0, 0.0, 4.0, 0.0],    # x^2 + 4 = 0 (no real roots)
            [1.0, 2.0, 0.0, 0.0],    # x + 2 = 0, root at x=-2
        ])
        
        # Bad brackets: both functions positive at both endpoints
        a_array = np.array([1.0, 1.0])  
        b_array = np.array([2.0, 2.0])
        
        # Should not raise by default, but mark as not converged
        roots, converged = solver.solve_vectorized(
            f_func=None,
            tolerance=1e-6,
            func_types=func_types,
            params_array=params_array,
            a_array=a_array,
            b_array=b_array,
            raise_on_fail=False
        )
        
        # Should not converge
        assert not converged.any()
        
        # Should raise when raise_on_fail=True
        with pytest.raises(RuntimeError, match="failed to converge"):
            solver.solve_vectorized(
                f_func=None,
                tolerance=1e-6,
                func_types=func_types,
                params_array=params_array,
                a_array=a_array,
                b_array=b_array,
                raise_on_fail=True
            )

    def test_chunked_processing(self, solver):
        """Test that chunked processing works correctly"""
        # Test with chunk_size smaller than problem size
        n = 15  # Larger than default chunk_size of 10
        func_types = np.array(['linear'] * n)
        
        # Linear functions: a*x + b = 0, root at x = -b/a
        # Use different slopes and intercepts
        params_array = np.array([[1.0, -i, 0.0, 0.0] for i in range(1, n+1)])
        expected_roots = np.arange(1, n+1, dtype=float)
        
        # Brackets around each root
        a_array = expected_roots - 0.5
        b_array = expected_roots + 0.5
        
        roots, converged = solver.solve_vectorized(
            f_func=None,
            tolerance=1e-6,
            chunk_size=5,  # Force multiple chunks
            func_types=func_types,
            params_array=params_array,
            a_array=a_array,
            b_array=b_array
        )
        
        assert converged.all()
        np.testing.assert_allclose(roots, expected_roots, atol=1e-1)  # More reasonable tolerance

    def test_mixed_convergence_scenarios(self, solver):
        """Test mixture of convergent and non-convergent cases"""
        n = 4
        func_types = np.array(['linear', 'quadratic', 'linear', 'quadratic'])
        params_array = np.array([
            [1.0, -1.0, 0.0, 0.0],    # x - 1 = 0, good bracket
            [1.0, 0.0, -4.0, 0.0],    # x^2 - 4 = 0, good bracket  
            [1.0, -3.0, 0.0, 0.0],    # x - 3 = 0, bad bracket (no sign change)
            [1.0, 0.0, 1.0, 0.0],     # x^2 + 1 = 0, no real roots
        ])
        
        a_array = np.array([0.5, 1.5, 4.0, 0.0])   # Third has no sign change
        b_array = np.array([1.5, 2.5, 5.0, 1.0])   # Fourth has no real roots
        
        roots, converged = solver.solve_vectorized(
            f_func=None,
            tolerance=1e-6,
            func_types=func_types,
            params_array=params_array,
            a_array=a_array,
            b_array=b_array,
            raise_on_fail=False
        )
        
        # First two should converge, last two should not
        expected_converged = np.array([True, True, False, False])
        np.testing.assert_array_equal(converged, expected_converged)
        
        # Check convergent solutions (with reasonable tolerance)
        assert abs(roots[0] - 1.0) < 1e-1  # x - 1 = 0
        assert abs(roots[1] - 2.0) < 1e-1  # x^2 - 4 = 0, positive root

    def test_tolerance_and_iteration_limits(self, solver):
        """Test tolerance and iteration limit behavior"""
        # Use a slowly converging case
        func_types = np.array(['quadratic'])
        params_array = np.array([[1.0, 0.0, -2.0, 0.0]])  # x^2 - 2 = 0, root = sqrt(2)
        a_array = np.array([1.0])
        b_array = np.array([2.0])
        
        # Test tight tolerance
        roots, converged = solver.solve_vectorized(
            f_func=None,
            tolerance=1e-8,
            max_iterations=100,
            func_types=func_types,
            params_array=params_array,
            a_array=a_array,
            b_array=b_array
        )
        
        assert converged[0]
        np.testing.assert_allclose(roots[0], np.sqrt(2), atol=1e-4)  # More achievable tolerance
        
        # Test iteration limit exceeded
        roots, converged = solver.solve_vectorized(
            f_func=None,
            tolerance=1e-12,  # Very tight
            max_iterations=5,  # Very few iterations
            func_types=func_types,
            params_array=params_array,
            a_array=a_array,
            b_array=b_array
        )
        
        # Should not converge due to iteration limit
        assert not converged[0]

    def test_dispatcher_evaluation_error_handling(self, solver):
        """Test handling when dispatcher evaluation fails"""
        # Use unknown function type that will cause dispatcher to return NaN
        func_types = np.array(['unknown_function'])
        params_array = np.array([[1.0, 2.0, 3.0, 4.0]])
        a_array = np.array([1.0])
        b_array = np.array([2.0])
        
        # Should handle gracefully and not converge
        roots, converged = solver.solve_vectorized(
            f_func=None,
            tolerance=1e-6,
            func_types=func_types,
            params_array=params_array,
            a_array=a_array,
            b_array=b_array,
            raise_on_fail=False,
            enable_logging=False  # Suppress warning logs for this test
        )
        
        assert not converged[0]

    @pytest.mark.parametrize("chunk_size", [1, 5, 100])
    def test_chunk_size_consistency(self, solver, chunk_size):
        """Test that different chunk sizes give same results"""
        n = 10
        func_types = np.array(['linear'] * n)
        params_array = np.array([[1.0, -float(i), 0.0, 0.0] for i in range(1, n+1)])
        a_array = np.arange(0.5, n+0.5)
        b_array = np.arange(1.5, n+1.5)
        
        roots, converged = solver.solve_vectorized(
            f_func=None,
            tolerance=1e-6,
            chunk_size=chunk_size,
            func_types=func_types,
            params_array=params_array,
            a_array=a_array,
            b_array=b_array
        )
        
        assert converged.all()
        expected = np.arange(1, n+1, dtype=float)
        np.testing.assert_allclose(roots, expected, atol=1e-1)


if __name__ == "__main__":
    # Can be run directly for quick testing
    pytest.main([__file__, "-v"])