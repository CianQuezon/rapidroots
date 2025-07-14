"""
Test suite for BisectionSolver._solve_scalar method and related functionality.

Organized into:
- Successful convergence cases  
- Error and edge cases
- Vectorized functionality
- Mixin pipeline components
"""

import pytest
import numpy as np
import math
from unittest.mock import Mock, patch
from rapidroots.solvers.bisection import BisectionOptions, BisectionSolver
# Assuming the BisectionSolver class is imported
# from your_module import BisectionSolver, BisectionOptions


# ===== SHARED FIXTURES AND HELPERS =====

@pytest.fixture
def solver():
    """Create a BisectionSolver instance for testing."""
    return BisectionSolver()


def assert_convergence(root, converged, expected_root, tolerance_margin=None):
    """Helper to assert successful convergence with expected root."""
    assert converged, "Expected convergence but got convergence failure"
    if tolerance_margin is None:
        tolerance_margin = 0.1  # Default margin
    assert abs(root - expected_root) < tolerance_margin, f"Root {root} not within {tolerance_margin} of expected {expected_root}"


def assert_no_convergence(root, converged, expected_root=None):
    """Helper to assert failed convergence."""
    assert not converged, "Expected convergence failure but got success"
    if expected_root is not None:
        assert abs(root - expected_root) < 0.01, f"Expected root {expected_root} but got {root}"


# ===== SUCCESSFUL CONVERGENCE TESTS =====

class TestBisectionSuccessfulConvergence:
    """Test cases where bisection should successfully find roots."""
    
    @pytest.mark.parametrize("func,a,b,expected_root,tolerance,description", [
        # Basic polynomial functions
        (lambda x: x - 2, 0, 4, 2.0, 0.01, "linear function"),
        (lambda x: x**2 - 4, 1, 3, 2.0, 0.01, "quadratic positive root"),
        (lambda x: x**2 - 4, -3, -1, -2.0, 0.01, "quadratic negative root"),
        (lambda x: x**3 - x - 2, 1, 2, 1.52138, 0.1, "cubic function"),
        (lambda x: 2*x - 6, 2, 4, 3.0, 0.01, "scaled linear"),
        (lambda x: x**2 - x - 6, 2, 4, 3.0, 0.01, "factored quadratic"),
        
        # Transcendental functions
        (lambda x: math.sin(x), 3, 4, math.pi, 0.01, "sin function near π"),
        (lambda x: math.exp(x) - 2, 0, 1, math.log(2), 0.01, "exponential function"),
        
        # Functions with different slopes
        (lambda x: 1000 * (x - 2), 1, 3, 2.0, 0.01, "steep slope function"),
        (lambda x: 0.001 * (x - 2), 1, 3, 2.0, 0.2, "flat slope function"),
        
        # Different interval sizes
        (lambda x: x - 1.0001, 1.0, 1.001, 1.0001, 0.01, "very small interval"),
        (lambda x: x - 500, -1000, 1000, 500, 0.01, "very large interval"),
    ])
    def test_basic_convergence_cases(self, solver, func, a, b, expected_root, tolerance, description):
        """Test successful convergence for various function types."""
        root, converged = solver._solve_scalar(func, a, b, tolerance, max_iterations=50, raise_on_fail=False)
        assert_convergence(root, converged, expected_root, tolerance * 10)
    
    def test_exact_root_at_left_endpoint(self, solver):
        """Test detection of exact root at left endpoint."""
        def f(x):
            return x - 3
        
        root, converged = solver._solve_scalar(f, 3, 5, tolerance=0.01, max_iterations=50, raise_on_fail=False)
        assert converged
        assert root == 3.0
    
    def test_exact_root_at_right_endpoint(self, solver):
        """Test detection of exact root at right endpoint.""" 
        def f(x):
            return x - 7
        
        root, converged = solver._solve_scalar(f, 5, 7, tolerance=0.01, max_iterations=50, raise_on_fail=False)
        assert converged
        assert root == 7.0
    
    @pytest.mark.parametrize("tolerance", [0.1, 0.01, 0.001, 1e-6])
    def test_tolerance_scaling(self, solver, tolerance):
        """Test that different tolerances work correctly."""
        def f(x):
            return x - 2.5
        
        root, converged = solver._solve_scalar(f, 2, 3, tolerance, max_iterations=100, raise_on_fail=False)
        assert_convergence(root, converged, 2.5, tolerance * 2)
    
    def test_interval_width_convergence(self, solver):
        """Test convergence based on interval width rather than function value."""
        def f(x):
            return 0.001 * (x - 5)  # Very flat function
        
        # Use tolerance much smaller than function scale to force interval-width convergence
        root, converged = solver._solve_scalar(f, 4, 6, tolerance=0.0001, max_iterations=50, raise_on_fail=False)
        assert_convergence(root, converged, 5.0, 0.01)
    
    def test_convergence_rate_within_theory(self, solver):
        """Test that convergence happens within expected iteration count."""
        def f(x):
            return x - 5
        
        # For interval [0, 10] with tolerance 0.01, should need at most:
        # ceil(log2((10-0)/0.01)) = ceil(log2(1000)) ≈ 10 iterations
        root, converged = solver._solve_scalar(f, 0, 10, tolerance=0.01, max_iterations=15, raise_on_fail=False)
        assert_convergence(root, converged, 5.0, 0.01)
        
        # TODO: If iteration count is exposed, add explicit check:
        # assert actual_iterations <= 10


# ===== ERROR AND EDGE CASES =====

class TestBisectionErrorAndEdgeCases:
    """Test cases where bisection should fail gracefully or raise errors."""
    
    @pytest.mark.parametrize("a,b,expected_root,description", [
        (np.nan, 2, np.nan, "NaN left endpoint"),
        (1, np.inf, np.nan, "infinite right endpoint"),
        (np.inf, np.nan, np.nan, "both endpoints invalid"),
    ])
    def test_invalid_bracket_endpoints(self, solver, a, b, expected_root, description):
        """Test behavior with invalid bracket endpoints."""
        def f(x):
            return x - 1
        
        root, converged = solver._solve_scalar(f, a, b, tolerance=0.01, max_iterations=50, raise_on_fail=False)
        assert_no_convergence(root, converged)
        if np.isnan(expected_root):
            assert np.isnan(root)
        else:
            assert root == expected_root
    
    @pytest.mark.parametrize("a,b,expected_midpoint", [
        (3, 1, 2.0),  # a > b
        (2, 2, 2.0),  # a == b
    ])
    def test_invalid_bracket_order_returns_midpoint(self, solver, a, b, expected_midpoint):
        """Test behavior when bracket order is invalid - should return midpoint."""
        def f(x):
            return x - 1
        
        root, converged = solver._solve_scalar(f, a, b, tolerance=0.01, max_iterations=50, raise_on_fail=False)
        assert_no_convergence(root, converged, expected_midpoint)
    
    def test_same_sign_brackets_returns_midpoint(self, solver):
        """Test behavior when f(a) and f(b) have same sign - should return midpoint."""
        def f(x):
            return x**2 + 1  # Always positive, no real roots
        
        root, converged = solver._solve_scalar(f, -2, 2, tolerance=0.01, max_iterations=50, raise_on_fail=False)
        assert_no_convergence(root, converged, 0.0)
    
    def test_function_evaluation_error_handling(self, solver):
        """Test behavior when function raises exception during evaluation."""
        def f(x):
            if abs(x - 1.5) < 0.01:  # Error near expected root
                raise ZeroDivisionError("Division by zero")
            return x - 1.5
        
        root, converged = solver._solve_scalar(f, 1, 2, tolerance=0.01, max_iterations=50, raise_on_fail=False)
        assert_no_convergence(root, converged)
    
    def test_max_iterations_exceeded_behavior(self, solver):
        """Test behavior when max iterations is exceeded."""
        def f(x):
            return x - 1.7320508  # sqrt(3), not a nice fraction - won't hit exact midpoints
        
        # Use very small max_iterations to force timeout
        root, converged = solver._solve_scalar(f, 1, 2, tolerance=1e-10, max_iterations=2, raise_on_fail=False)
        assert_no_convergence(root, converged)
        assert 1 <= root <= 2, "Root should be within original bracket"
    
    # Tests with raise_on_fail=True
    def test_raise_on_fail_invalid_endpoints(self, solver):
        """Test that raise_on_fail=True raises for invalid endpoints."""
        def f(x):
            return x - 1
        
        with pytest.raises(ValueError, match="Invalid bracket endpoints"):
            solver._solve_scalar(f, np.nan, 2, tolerance=0.01, max_iterations=50, raise_on_fail=True)
    
    def test_raise_on_fail_invalid_order(self, solver):
        """Test that raise_on_fail=True raises for invalid bracket order."""
        def f(x):
            return x - 1
        
        with pytest.raises(ValueError, match="Invalid bracket order"):
            solver._solve_scalar(f, 3, 1, tolerance=0.01, max_iterations=50, raise_on_fail=True)
    
    def test_raise_on_fail_same_sign_condition(self, solver):
        """Test that raise_on_fail=True raises for same sign condition."""
        def f(x):
            return x**2 + 1  # Always positive
        
        with pytest.raises(ValueError, match="f\\(a\\) and f\\(b\\) must have opposite signs"):
            solver._solve_scalar(f, -2, 2, tolerance=0.01, max_iterations=50, raise_on_fail=True)
    
    def test_raise_on_fail_function_error(self, solver):
        """Test that raise_on_fail=True raises for function evaluation errors."""
        def f(x):
            raise RuntimeError("Function evaluation failed")
        
        with pytest.raises(RuntimeError, match="Function evaluation failed"):
            solver._solve_scalar(f, 1, 2, tolerance=0.01, max_iterations=50, raise_on_fail=True)
    
    def test_raise_on_fail_max_iterations(self, solver):
        """Test that raise_on_fail=True raises when max iterations exceeded."""
        def f(x):
            return x - 1.7320508  # sqrt(3), not a nice fraction - won't hit exact midpoints
        
        with pytest.raises(RuntimeError, match="Bisection failed to converge"):
            solver._solve_scalar(f, 1, 2, tolerance=1e-10, max_iterations=2, raise_on_fail=True)


# ===== DISPATCH FUNCTIONALITY TESTS =====

class TestBisectionDispatch:
    """Test dispatch functionality for scalar vs array inputs."""
    
    def test_solve_dispatches_scalar_correctly(self, solver):
        """Test that _bisect_dispatch correctly handles scalar inputs."""
        def f(x):
            return x - 2
        
        # Mock _solve_scalar to verify it gets called
        with patch.object(solver, '_solve_scalar', return_value=(2.0, True)) as mock_scalar:
            root, converged = solver._bisect_dispatch(f, 1, 3, 0.01, 50, False)
            
            mock_scalar.assert_called_once_with(f, 1, 3, 0.01, 50, False)
            assert root == 2.0
            assert converged == True
    
    def test_solve_dispatches_array_to_base_class(self, solver):
        """Test that _bisect_dispatch correctly dispatches array inputs to base class."""
        def f(x):
            return x - np.array([1, 2])
        
        a_array = np.array([0, 1])
        b_array = np.array([2, 3])
        
        # Mock the parent's solve_vectorized method
        with patch('rapidroots.solvers.base.BracketingMethodBase.solve_vectorized', 
                   return_value=(np.array([1.0, 2.0]), np.array([True, True]))) as mock_vectorized:
            
            roots, converged = solver._bisect_dispatch(f, a_array, b_array, 0.01, 50, False)
            
            mock_vectorized.assert_called_once()
            np.testing.assert_array_equal(roots, [1.0, 2.0])
            np.testing.assert_array_equal(converged, [True, True])


# ===== MIXIN PIPELINE TESTS =====

class TestBracketPreparationMixin:
    """Test the bracket preparation mixin components."""
    
    def test_apply_conditioning_with_conditioning_enabled(self, solver):
        """Test that apply_conditioning works when conditioning is enabled."""
        def original_func(x):
            return x - 2
        
        # Test with conditioning enabled (use positional arguments)
        conditioned_func = solver.apply_conditioning(original_func, True)
        
        # Should return a function that doesn't throw (basic smoke test)
        result = conditioned_func(1.5)
        # Conditioning might wrap function to return numpy arrays for vectorization
        assert isinstance(result, (int, float, np.number, np.ndarray))
        # Verify the mathematical result is still correct
        if isinstance(result, np.ndarray):
            assert np.isclose(result, -0.5)  # 1.5 - 2 = -0.5
        else:
            assert abs(result - (-0.5)) < 1e-10
    
    def test_apply_conditioning_with_conditioning_disabled(self, solver):
        """Test that apply_conditioning returns original function when disabled."""
        def original_func(x):
            return x - 2
        
        # Test with conditioning disabled (use positional arguments)
        result_func = solver.apply_conditioning(original_func, False)
        
        # Should return the same function
        assert result_func is original_func
    
    def test_validate_all_brackets_success_case(self, solver):
        """Test validate_all_brackets with valid brackets."""
        def f(x):
            return x - 2
        
        # Should not raise with valid brackets (use positional arguments)
        try:
            solver.validate_all_brackets(
                f, 1, 3, True,  # f_func, a, b, validate
                None, None, True  # func_types, params_array, raise_on_fail
            )
        except Exception as e:
            pytest.fail(f"validate_all_brackets raised unexpectedly: {e}")
    
    def test_validate_all_brackets_invalid_case(self, solver):
        """Test validate_all_brackets with invalid brackets."""
        def f(x):
            return x**2 + 1  # Always positive, invalid bracket
        
        # Should raise with invalid brackets when validate=True and raise_on_fail=True
        with pytest.raises(ValueError):
            solver.validate_all_brackets(
                f, -1, 1, True,  # f_func, a, b, validate
                None, None, True  # func_types, params_array, raise_on_fail
            )
    
    def test_validate_all_brackets_disabled(self, solver):
        """Test validate_all_brackets when validation is disabled."""
        def f(x):
            return x**2 + 1  # Always positive, would be invalid
        
        # When validate=False, basic bracket checks still run but raise_on_fail=False
        # should cause it to log warnings instead of raising exceptions
        try:
            solver.validate_all_brackets(
                f, -1, 1, False,  # f_func, a, b, validate
                None, None, False  # func_types, params_array, raise_on_fail=False
            )
            # Should succeed (with warning logged) when raise_on_fail=False
        except Exception as e:
            pytest.fail(f"validate_all_brackets raised unexpectedly when raise_on_fail=False: {e}")
    
    def test_validate_all_brackets_disabled_with_valid_brackets(self, solver):
        """Test validate_all_brackets disabled vs enabled with valid brackets."""
        def f(x):
            return x - 1.5  # Root at x = 1.5
        
        # Use interval [1, 2] which properly brackets the root
        # f(1) = 1 - 1.5 = -0.5 (negative)
        # f(2) = 2 - 1.5 = 0.5 (positive)  
        # Sign change exists, so this is valid bracketing
        
        try:
            # Test with validation disabled
            solver.validate_all_brackets(
                f, 1, 2, False,   # f_func, a, b, validate=False
                None, None, True  # func_types, params_array, raise_on_fail=True
            )
            
            # Test with validation enabled  
            solver.validate_all_brackets(
                f, 1, 2, True,    # f_func, a, b, validate=True
                None, None, True  # func_types, params_array, raise_on_fail=True
            )
        except Exception as e:
            pytest.fail(f"validate_all_brackets failed with valid brackets: {e}")
    
    def test_find_missing_brackets_no_auto(self, solver):
        """Test find_missing_brackets when auto=False."""
        a, b = 1, 3
        
        # Use positional arguments as shown in the original code
        result_a, result_b = solver.find_missing_brackets(
            a, b, False,  # a, b, auto
            None, None,   # func_types, params_array
            None, None,   # initial_guess, preferred_domain
            False         # raise_on_fail
        )
        
        # Should return original values when auto=False
        assert result_a == a
        assert result_b == b


# ===== INTEGRATION TESTS =====

class TestBisectionIntegration:
    """Integration tests for the full solve() pipeline."""
    
    def test_full_solve_pipeline_success(self, solver):
        """Test the complete solve() pipeline for successful case."""
        def f(x):
            return x - 2
        
        opts = BisectionOptions(condition=True, validate=True, auto=False)
        
        root, converged = solver.solve(
            f, a=1, b=3, tolerance=0.01, max_iterations=50, 
            raise_on_fail=False, opts=opts
        )
        
        assert_convergence(root, converged, 2.0, 0.01)
    
    def test_full_solve_pipeline_with_defaults(self, solver):
        """Test solve() with default options."""
        def f(x):
            return x - 5
        
        # Test with default options (None should create BisectionOptions())
        root, converged = solver.solve(f, a=4, b=6, tolerance=0.01, max_iterations=50, raise_on_fail=False)
        
        assert_convergence(root, converged, 5.0, 0.01)


if __name__ == "__main__":
    # Run with: python -m pytest test_bisection_scalar.py -v
    pytest.main([__file__, "-v"])