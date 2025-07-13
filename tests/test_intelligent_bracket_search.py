import pytest
import numpy as np
from unittest.mock import Mock, patch
from numpy.testing import assert_array_equal, assert_allclose
from rapidroots.utils.intelligent_bracket import IntelligentBracketFinder


class TestIntelligentBracketFinder:
    
    @pytest.fixture
    def mock_dispatcher(self):
        """Create mock dispatcher."""
        return Mock()
    
    @pytest.fixture
    def finder(self, mock_dispatcher):
        """Create IntelligentBracketFinder instance."""
        return IntelligentBracketFinder(mock_dispatcher)
    
    @patch('rapidroots.utils.intelligent_bracket.BulkBracketFinder.auto_bracket_search')
    def test_local_search_success(self, mock_search, finder):
        """Test successful local search strategy."""
        # Mock successful local search
        mock_search.return_value = (
            np.array([0.0, 1.0]),  # left brackets
            np.array([1.0, 2.0])   # right brackets
        )
        
        func_types = np.array(['linear', 'quadratic'])
        params_array = np.array([[1.0, 0.0], [2.0, 1.0]])
        initial_guess = np.array([0.5, 1.5])
        
        left, right = finder.find_brackets(
            func_types, params_array, initial_guess=initial_guess
        )
        
        # Should find brackets via local search
        assert_array_equal(left, [0.0, 1.0])
        assert_array_equal(right, [1.0, 2.0])
        mock_search.assert_called_once()
    
    @patch('rapidroots.utils.intelligent_bracket.GridBracketScanner.find_all_brackets')
    @patch('rapidroots.utils.intelligent_bracket.BulkBracketFinder.auto_bracket_search')
    def test_grid_search_fallback(self, mock_search, mock_grid, finder):
        """Test fallback to grid search when local search fails."""
        # Mock failed local search
        mock_search.return_value = (
            np.array([np.nan, np.nan]),  # Failed local search
            np.array([np.nan, np.nan])
        )
        
        # Mock successful grid search
        mock_grid.return_value = [(0.0, 1.0)]
        
        func_types = np.array(['linear', 'linear'])
        params_array = np.array([[1.0, 0.0], [2.0, 1.0]])
        
        left, right = finder.find_brackets(
            func_types, params_array,
            initial_guess=np.array([0.5, 1.5]),
            preferred_domain=(-5, 5)
        )
        
        # Should use grid search for failed points
        assert not np.isnan(left[0])  # First point found by grid
        assert not np.isnan(right[0])
        mock_grid.assert_called()
    
    @patch('rapidroots.utils.intelligent_bracket.GridBracketScanner.find_all_brackets')
    def test_global_fallback_domains(self, mock_grid, finder):
        """Test global fallback with multiple domains."""
        # Mock grid search that succeeds on second domain
        mock_grid.side_effect = [
            [],  # First domain fails
            [(0.0, 1.0)]  # Second domain succeeds
        ]
        
        func_types = np.array(['linear'])
        params_array = np.array([[1.0, 0.0]])
        
        left, right = finder.find_brackets(
            func_types, params_array, global_fallback=True
        )
        
        # Should eventually find bracket via global fallback
        assert not np.isnan(left[0])
        assert not np.isnan(right[0])
        # Should have tried multiple domains
        assert mock_grid.call_count >= 2
    
    def test_strict_mode_failure(self, finder):
        """Test strict mode raises error on failure."""
        func_types = np.array(['linear'])
        params_array = np.array([[1.0, 0.0]])
        
        # All strategies fail (no mocking = no success)
        with pytest.raises(ValueError, match="No valid brackets found"):
            finder.find_brackets(
                func_types, params_array, strict=True
            )
    
    def test_non_strict_mode_returns_nan(self, finder):
        """Test non-strict mode returns NaN for failures."""
        func_types = np.array(['linear'])
        params_array = np.array([[1.0, 0.0]])
        
        # All strategies fail
        left, right = finder.find_brackets(
            func_types, params_array, strict=False
        )
        
        # Should return NaN without raising
        assert np.isnan(left[0])
        assert np.isnan(right[0])
    
    def test_return_report_functionality(self, finder):
        """Test diagnostic report when return_report=True."""
        func_types = np.array(['linear'])
        params_array = np.array([[1.0, 0.0]])
        
        left, right, report = finder.find_brackets(
            func_types, params_array, 
            strict=False, 
            return_report=True
        )
        
        # Should return report with expected keys
        expected_keys = {
            'total_points', 'local_search_success', 
            'grid_search_success', 'global_fallback_success', 
            'final_failures'
        }
        assert set(report.keys()) == expected_keys
        assert report['total_points'] == 1
        assert report['final_failures'] >= 0
    
    @patch('rapidroots.utils.intelligent_bracket.BulkBracketFinder.auto_bracket_search')
    def test_initial_guess_broadcasting(self, mock_search, finder):
        """Test scalar initial_guess is broadcast to all points."""
        mock_search.return_value = (
            np.array([0.0, 1.0]),
            np.array([1.0, 2.0])
        )
        
        func_types = np.array(['linear', 'linear'])
        params_array = np.array([[1.0, 0.0], [2.0, 1.0]])
        
        finder.find_brackets(
            func_types, params_array, 
            initial_guess=0.5  # Scalar should broadcast
        )
        
        # Should have called with broadcast guess
        call_args = mock_search.call_args
        initial_guess_used = call_args.kwargs['initial_guess']
        assert_array_equal(initial_guess_used, [0.5, 0.5])
    
    @patch('rapidroots.utils.intelligent_bracket.GridBracketScanner.find_all_brackets')
    def test_closest_bracket_selection(self, mock_grid, finder):
        """Test selection of bracket closest to initial guess."""
        # Mock multiple brackets found
        mock_grid.return_value = [
            (5.0, 6.0),   # Far from initial guess
            (0.0, 1.0),   # Close to initial guess (0.5)
            (10.0, 11.0)  # Very far
        ]
        
        func_types = np.array(['linear'])
        params_array = np.array([[1.0, 0.0]])
        initial_guess = np.array([0.5])
        
        left, right = finder.find_brackets(
            func_types, params_array,
            initial_guess=initial_guess,
            preferred_domain=(-10, 10)
        )
        
        # Should select closest bracket (0.0, 1.0)
        assert_allclose(left[0], 0.0)
        assert_allclose(right[0], 1.0)
    
    def test_mixed_success_strategies(self, finder):
        """Test mixed success across different strategies."""
        with patch('rapidroots.utils.intelligent_bracket.BulkBracketFinder.auto_bracket_search') as mock_search, \
             patch('rapidroots.utils.intelligent_bracket.GridBracketScanner.find_all_brackets') as mock_grid:
            
            # Local search succeeds for first point only
            mock_search.return_value = (
                np.array([0.0, np.nan]),
                np.array([1.0, np.nan])
            )
            
            # Grid search succeeds for second point
            mock_grid.return_value = [(2.0, 3.0)]
            
            func_types = np.array(['linear', 'quadratic'])
            params_array = np.array([[1.0, 0.0], [2.0, 1.0]])
            
            left, right, report = finder.find_brackets(
                func_types, params_array,
                initial_guess=np.array([0.5, 2.5]),
                preferred_domain=(-5, 5),
                return_report=True
            )
            
            # First point from local search, second from grid search
            assert not np.isnan(left[0]) and not np.isnan(right[0])
            assert not np.isnan(left[1]) and not np.isnan(right[1])
            assert report['local_search_success'] >= 1
            assert report['grid_search_success'] >= 1
    
    def test_exception_handling_in_strategies(self, finder):
        """Test graceful handling of exceptions in strategies."""
        with patch('rapidroots.utils.intelligent_bracket.BulkBracketFinder.auto_bracket_search') as mock_search:
            # Local search raises exception
            mock_search.side_effect = RuntimeError("Search failed")
            
            func_types = np.array(['linear'])
            params_array = np.array([[1.0, 0.0]])
            
            # Should not crash, continue to next strategy
            left, right = finder.find_brackets(
                func_types, params_array,
                initial_guess=np.array([0.5]),
                strict=False
            )
            
            # Should handle exception gracefully
            assert isinstance(left, np.ndarray)
            assert isinstance(right, np.ndarray)
    
    def test_chunk_size_parameter_passing(self, finder):
        """Test chunk_size is passed to underlying methods."""
        with patch('rapidroots.utils.intelligent_bracket.BulkBracketFinder.auto_bracket_search') as mock_search:
            mock_search.return_value = (
                np.array([0.0]), np.array([1.0])
            )
            
            func_types = np.array(['linear'])
            params_array = np.array([[1.0, 0.0]])
            
            finder.find_brackets(
                func_types, params_array,
                initial_guess=np.array([0.5]),
                chunk_size=25000
            )
            
            # Should pass chunk_size to underlying method
            mock_search.assert_called_with(
                finder.dispatcher, func_types, params_array,
                initial_guess=np.array([0.5]), chunk_size=25000
            )