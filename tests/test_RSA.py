import pytest
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
import sys
import os

# Add the parent directory to the path to import yescarpenter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from yescarpenter.RSA import construct_RDM, do_RSA, variance_partitioning, standardize_rdms

class TestConstructRDM:
    """Test cases for construct_RDM function"""
    
    def setup_method(self):
        """Set up test data"""
        # Simple 3x2 test data
        self.test_data = np.array([[1, 2], [3, 4], [5, 6]])
        self.n_target = 3
        
        # Test data with NaN/inf values
        self.data_with_nan = np.array([[1, 2], [np.nan, 4], [5, 6], [7, 8]])
        self.data_with_inf = np.array([[1, 2], [np.inf, 4], [5, 6], [7, 8]])
        
        # DataFrame test data
        self.df_data = pd.DataFrame({
            'feature1': [1, 3, 5],
            'feature2': [2, 4, 6]
        })
    
    def test_construct_rdm_basic_euclidean(self):
        """Test basic Euclidean distance RDM construction"""
        rdm = construct_RDM(self.test_data, self.n_target, method="euclidean", draw=False)
        
        # Check shape
        assert rdm.shape == (3, 3), f"Expected shape (3, 3), got {rdm.shape}"
        
        # Check diagonal is zero
        assert np.allclose(np.diag(rdm), 0), "Diagonal should be zero"
        
        # Check symmetry
        assert np.allclose(rdm, rdm.T), "RDM should be symmetric"
        
        # Check specific distances
        expected_dist = np.sqrt((1-3)**2 + (2-4)**2)  # Distance between first two points
        assert np.isclose(rdm[0, 1], expected_dist), f"Expected distance {expected_dist}, got {rdm[0, 1]}"
    
    def test_construct_rdm_cosine(self):
        """Test cosine distance RDM construction"""
        rdm = construct_RDM(self.test_data, self.n_target, method="cosine", draw=False)
        
        assert rdm.shape == (3, 3), f"Expected shape (3, 3), got {rdm.shape}"
        assert np.allclose(np.diag(rdm), 0), "Diagonal should be zero"
        assert np.allclose(rdm, rdm.T), "RDM should be symmetric"
    
    def test_construct_rdm_cityblock(self):
        """Test Manhattan (cityblock) distance RDM construction"""
        rdm = construct_RDM(self.test_data, self.n_target, method="cityblock", draw=False)
        
        assert rdm.shape == (3, 3), f"Expected shape (3, 3), got {rdm.shape}"
        assert np.allclose(np.diag(rdm), 0), "Diagonal should be zero"
        assert np.allclose(rdm, rdm.T), "RDM should be symmetric"
        
        # Check specific Manhattan distance
        expected_dist = abs(1-3) + abs(2-4)  # Manhattan distance between first two points
        assert np.isclose(rdm[0, 1], expected_dist), f"Expected distance {expected_dist}, got {rdm[0, 1]}"
    
    def test_construct_rdm_spearman(self):
        """Test Spearman correlation-based RDM construction"""
        rdm = construct_RDM(self.test_data, self.n_target, method="spearman", draw=False)
        
        assert rdm.shape == (3, 3), f"Expected shape (3, 3), got {rdm.shape}"
        assert np.allclose(np.diag(rdm), 0), "Diagonal should be zero for perfect self-correlation"
        assert np.allclose(rdm, rdm.T), "RDM should be symmetric"
    
    def test_construct_rdm_single_feature(self):
        """Test RDM construction with single feature (edge case)"""
        single_feature_data = np.array([[1], [2], [3]])
        rdm = construct_RDM(single_feature_data, 3, method="euclidean", draw=False)
        
        # For single feature case with spearmanr
        assert rdm.shape == (3, 3), f"Expected shape (3, 3), got {rdm.shape}"
        assert np.allclose(np.diag(rdm), 0), "Diagonal should be zero"
    
    def test_construct_rdm_dataframe_input(self):
        """Test RDM construction with DataFrame input"""
        rdm = construct_RDM(self.df_data, self.n_target, method="euclidean", draw=False)
        
        assert rdm.shape == (3, 3), f"Expected shape (3, 3), got {rdm.shape}"
        assert np.allclose(np.diag(rdm), 0), "Diagonal should be zero"
        assert np.allclose(rdm, rdm.T), "RDM should be symmetric"
    
    def test_construct_rdm_nan_handling(self):
        """Test RDM construction with NaN values"""
        # Should remove rows with NaN values
        rdm = construct_RDM(self.data_with_nan, 3, method="euclidean", draw=False)  # Expecting 3 clean rows
        
        assert rdm.shape == (3, 3), f"Expected shape (3, 3) after NaN removal, got {rdm.shape}"
        assert np.allclose(np.diag(rdm), 0), "Diagonal should be zero"
        assert not np.any(np.isnan(rdm)), "RDM should not contain NaN values"
    
    def test_construct_rdm_inf_handling(self):
        """Test RDM construction with infinite values"""
        # Should remove rows with infinite values
        rdm = construct_RDM(self.data_with_inf, 3, method="euclidean", draw=False)  # Expecting 3 clean rows
        
        assert rdm.shape == (3, 3), f"Expected shape (3, 3) after inf removal, got {rdm.shape}"
        assert np.allclose(np.diag(rdm), 0), "Diagonal should be zero"
        assert not np.any(np.isinf(rdm)), "RDM should not contain infinite values"
    
    def test_construct_rdm_transposed_data(self):
        """Test RDM construction when data needs to be transposed"""
        transposed_data = self.test_data.T  # Shape will be (2, 3)
        rdm = construct_RDM(transposed_data, self.n_target, method="euclidean", draw=False)
        
        assert rdm.shape == (3, 3), f"Expected shape (3, 3), got {rdm.shape}"
    
    def test_construct_rdm_invalid_method(self):
        """Test error handling for invalid method"""
        with pytest.raises(ValueError, match="Unsupported method"):
            construct_RDM(self.test_data, self.n_target, method="invalid_method", draw=False)
    
    def test_construct_rdm_wrong_dimensions(self):
        """Test error handling for wrong number of targets"""
        with pytest.raises(ValueError, match="does not have"):
            construct_RDM(self.test_data, 5, method="euclidean", draw=False)  # Wrong n_target
    
    def test_construct_rdm_3d_data_error(self):
        """Test error handling for 3D input data"""
        data_3d = np.random.rand(3, 4, 5)
        with pytest.raises(ValueError, match="Data must be 1D or 2D"):
            construct_RDM(data_3d, 3, method="euclidean", draw=False)


class TestDoRSA:
    """Test cases for do_RSA function"""
    
    def setup_method(self):
        """Set up test data"""
        # Create two similar RDMs for testing
        np.random.seed(42)  # For reproducibility
        n = 5
        
        # Create base data
        data1 = np.random.randn(n, 3)
        data2 = data1 + 0.1 * np.random.randn(n, 3)  # Similar to data1 with noise
        
        self.rdm1 = cdist(data1, data1, metric='euclidean')
        self.rdm2 = cdist(data2, data2, metric='euclidean')
        
        # Create dissimilar RDMs
        data3 = np.random.randn(n, 3)
        self.rdm3 = cdist(data3, data3, metric='euclidean')
        
        # Create identical RDMs
        self.rdm_identical = self.rdm1.copy()
    
    def test_do_rsa_basic(self):
        """Test basic RSA functionality"""
        perm_r, obs_r, p_val = do_RSA(self.rdm1, self.rdm2, n_permutations=100, 
                                      random_state=42, plot_histogram=False)
        
        # Check return types and shapes
        assert isinstance(perm_r, np.ndarray), "Permuted correlations should be numpy array"
        assert isinstance(obs_r, (float, np.floating)), "Observed correlation should be float"
        assert isinstance(p_val, (float, np.floating)), "P-value should be float"
        
        assert len(perm_r) == 100, f"Expected 100 permutations, got {len(perm_r)}"
        assert 0 <= p_val <= 1, f"P-value should be between 0 and 1, got {p_val}"
        assert -1 <= obs_r <= 1, f"Correlation should be between -1 and 1, got {obs_r}"
    
    def test_do_rsa_identical_matrices(self):
        """Test RSA with identical matrices"""
        perm_r, obs_r, p_val = do_RSA(self.rdm1, self.rdm_identical, n_permutations=100, 
                                      random_state=42, plot_histogram=False)
        
        # Identical matrices should have correlation = 1 and very low p-value
        assert np.isclose(obs_r, 1.0, atol=1e-10), f"Identical matrices should have r=1, got {obs_r}"
        assert p_val <= 0.05, f"Identical matrices should have significant p-value, got {p_val}"
    
    def test_do_rsa_dissimilar_matrices(self):
        """Test RSA with dissimilar matrices"""
        perm_r, obs_r, p_val = do_RSA(self.rdm1, self.rdm3, n_permutations=100, 
                                      random_state=42, plot_histogram=False)
        
        # Dissimilar matrices should have lower correlation and higher p-value
        assert abs(obs_r) < 0.9, f"Dissimilar matrices should have lower correlation, got {obs_r}"
    
    def test_do_rsa_reproducibility(self):
        """Test that results are reproducible with same random seed"""
        perm_r1, obs_r1, p_val1 = do_RSA(self.rdm1, self.rdm2, n_permutations=50, 
                                          random_state=123, plot_histogram=False)
        perm_r2, obs_r2, p_val2 = do_RSA(self.rdm1, self.rdm2, n_permutations=50, 
                                          random_state=123, plot_histogram=False)
        
        assert np.allclose(perm_r1, perm_r2), "Results should be reproducible with same seed"
        assert np.isclose(obs_r1, obs_r2), "Observed correlation should be identical"
        assert np.isclose(p_val1, p_val2), "P-value should be identical"
    
    def test_do_rsa_different_seeds(self):
        """Test that different seeds give different permutation results"""
        perm_r1, obs_r1, p_val1 = do_RSA(self.rdm1, self.rdm2, n_permutations=50, 
                                          random_state=123, plot_histogram=False)
        perm_r2, obs_r2, p_val2 = do_RSA(self.rdm1, self.rdm2, n_permutations=50, 
                                          random_state=456, plot_histogram=False)
        
        # Observed correlation should be the same (deterministic)
        assert np.isclose(obs_r1, obs_r2), "Observed correlation should be identical"
        
        # Permutation results should be different
        assert not np.allclose(perm_r1, perm_r2), "Different seeds should give different permutations"
    
    def test_do_rsa_matrix_validation(self):
        """Test matrix validation"""
        # Test non-square matrix
        non_square = np.random.randn(3, 4)
        with pytest.raises(AssertionError, match="Matrices must have the same dimensions"):
            do_RSA(non_square, self.rdm2, n_permutations=10, plot_histogram=False)
        
        # Test different sized matrices
        small_rdm = np.random.randn(3, 3)
        with pytest.raises(AssertionError, match="Matrices must have the same dimensions"):
            do_RSA(small_rdm, self.rdm1, n_permutations=10, plot_histogram=False)

    def test_do_rsa_permutation_range(self):
        """Test that permutation correlations are within valid range"""
        perm_r, obs_r, p_val = do_RSA(self.rdm1, self.rdm2, n_permutations=50, 
                                      random_state=42, plot_histogram=False)
        
        # All correlations should be between -1 and 1
        assert np.all(perm_r >= -1), f"Some permutation correlations < -1: {perm_r.min()}"
        assert np.all(perm_r <= 1), f"Some permutation correlations > 1: {perm_r.max()}"
        assert not np.any(np.isnan(perm_r)), "Permutation correlations should not contain NaN"
    
    def test_do_rsa_p_value_calculation(self):
        """Test p-value calculation logic"""
        # For identical matrices, p-value should be very small
        perm_r, obs_r, p_val = do_RSA(self.rdm1, self.rdm_identical, n_permutations=100, 
                                      random_state=42, plot_histogram=False)
        
        # Count how many permutations are >= observed correlation
        n_greater = np.sum(perm_r >= obs_r)
        expected_p = (n_greater + 1) / (100 + 1)
        
        assert np.isclose(p_val, expected_p), f"P-value calculation incorrect: expected {expected_p}, got {p_val}"


class TestVariancePartitioning:
    """Test cases for variance_partitioning function"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)  # For reproducibility
        
        # Create synthetic data for testing
        n_stimuli = 6
        n_features = 4
        
        # Create base feature matrices
        conceptual_features = np.random.randn(n_stimuli, n_features)
        physical_features = np.random.randn(n_stimuli, n_features)
        
        # Create RDMs from features
        self.conceptual_rdm = cdist(conceptual_features, conceptual_features, metric='euclidean')
        self.physical_rdm = cdist(physical_features, physical_features, metric='euclidean')
        
        # Create a dependent variable RDM that's influenced by both predictors
        combined_features = 0.6 * conceptual_features + 0.4 * physical_features
        noise = 0.1 * np.random.randn(n_stimuli, n_features)
        self.dv_rdm = cdist(combined_features + noise, combined_features + noise, metric='euclidean')
        
        # Create another DV RDM that's primarily influenced by conceptual features
        conceptual_dominant = 0.8 * conceptual_features + 0.2 * physical_features
        self.conceptual_dominant_rdm = cdist(conceptual_dominant, conceptual_dominant, metric='euclidean')
        
        # Create predictor dictionary
        self.rdm_dict = {
            'Conceptual': self.conceptual_rdm,
            'Physical': self.physical_rdm
        }
        
        # Create DV dictionary
        self.dv_rdms = {
            'Mixed': self.dv_rdm,
            'Conceptual_Dominant': self.conceptual_dominant_rdm
        }
        
        # Create a single DV case for simpler testing
        self.single_dv = {'TestDV': self.dv_rdm}
        
        # Create identical RDMs for testing perfect correlation
        self.identical_rdms = {
            'Identical1': self.conceptual_rdm,
            'Identical2': self.conceptual_rdm.copy()
        }
    
    def test_variance_partitioning_basic(self):
        """Test basic variance partitioning functionality"""
        results_df = variance_partitioning(
            self.single_dv, self.rdm_dict, 
            plot_title='Test', print_results=False
        )
        
        # Check return type and structure
        assert isinstance(results_df, pd.DataFrame), "Results should be a pandas DataFrame"
        assert results_df.index.name == 'DV', "Index should be named 'DV'"
        assert 'TestDV' in results_df.index, "TestDV should be in the index"
        
        # Check required columns exist
        required_columns = [
            'Full R-squared',
            'Conceptual Exclusive Contribution',
            'Physical Exclusive Contribution',
            'Conceptual P-value',
            'Physical P-value',
            'Overlapped'
        ]
        
        for col in required_columns:
            assert col in results_df.columns, f"Column '{col}' should be present"
    
    def test_variance_partitioning_multiple_dvs(self):
        """Test variance partitioning with multiple dependent variables"""
        results_df = variance_partitioning(
            self.dv_rdms, self.rdm_dict,
            plot_title='Multiple DVs Test', print_results=False
        )
        
        # Check that all DVs are included
        assert len(results_df) == 2, "Should have 2 rows for 2 DVs"
        assert 'Mixed' in results_df.index, "Mixed DV should be in results"
        assert 'Conceptual_Dominant' in results_df.index, "Conceptual_Dominant DV should be in results"
        
        # Check that R-squared values are reasonable
        assert all(results_df['Full R-squared'] >= 0), "All R-squared values should be non-negative"
        assert all(results_df['Full R-squared'] <= 1), "All R-squared values should be <= 1"
    
    def test_variance_partitioning_contribution_sum(self):
        """Test that exclusive contributions + overlapped = full R-squared"""
        results_df = variance_partitioning(
            self.single_dv, self.rdm_dict,
            plot_title='Contribution Sum Test', print_results=False
        )
        
        row = results_df.iloc[0]
        full_r2 = row['Full R-squared']
        conceptual_contrib = row['Conceptual Exclusive Contribution']
        physical_contrib = row['Physical Exclusive Contribution']
        overlapped = row['Overlapped']
        
        # Check that components sum to full R-squared (within tolerance)
        total_contrib = conceptual_contrib + physical_contrib + overlapped
        assert np.isclose(total_contrib, full_r2, atol=1e-10), \
            f"Contributions don't sum to full R-squared: {total_contrib} != {full_r2}"
    
    def test_variance_partitioning_positive_contributions(self):
        """Test that exclusive contributions are non-negative"""
        results_df = variance_partitioning(
            self.single_dv, self.rdm_dict,
            plot_title='Positive Contributions Test', print_results=False
        )
        
        row = results_df.iloc[0]
        conceptual_contrib = row['Conceptual Exclusive Contribution']
        physical_contrib = row['Physical Exclusive Contribution']
        
        assert conceptual_contrib >= -1e-10, f"Conceptual contribution should be non-negative: {conceptual_contrib}"
        assert physical_contrib >= -1e-10, f"Physical contribution should be non-negative: {physical_contrib}"
    
    def test_variance_partitioning_p_values_range(self):
        """Test that p-values are in valid range [0, 1]"""
        results_df = variance_partitioning(
            self.single_dv, self.rdm_dict,
            plot_title='P-values Test', print_results=False
        )
        
        row = results_df.iloc[0]
        conceptual_p = row['Conceptual P-value']
        physical_p = row['Physical P-value']
        
        assert 0 <= conceptual_p <= 1, f"Conceptual p-value out of range: {conceptual_p}"
        assert 0 <= physical_p <= 1, f"Physical p-value out of range: {physical_p}"
    
    def test_variance_partitioning_identical_predictors(self):
        """Test variance partitioning with identical predictor RDMs"""
        results_df = variance_partitioning(
            self.single_dv, self.identical_rdms,
            plot_title='Identical Predictors Test', print_results=False
        )
        
        row = results_df.iloc[0]
        
        # With identical predictors, exclusive contributions should be very small
        # (most variance should be in overlap)
        identical1_contrib = row['Identical1 Exclusive Contribution']
        identical2_contrib = row['Identical2 Exclusive Contribution']
        overlapped = row['Overlapped']
        
        # Overlapped should be much larger than exclusive contributions
        assert overlapped > identical1_contrib, "Overlap should dominate with identical predictors"
        assert overlapped > identical2_contrib, "Overlap should dominate with identical predictors"
    
    def test_variance_partitioning_single_predictor(self):
        """Test variance partitioning with a single predictor"""
        single_predictor = {'Conceptual': self.conceptual_rdm}
        
        results_df = variance_partitioning(
            self.single_dv, single_predictor,
            plot_title='Single Predictor Test', print_results=False
        )
        
        row = results_df.iloc[0]
        
        # With only one predictor, there should be no overlap
        assert np.isclose(row['Overlapped'], 0, atol=1e-10), "No overlap expected with single predictor"
        
        # Exclusive contribution should equal full R-squared
        exclusive = row['Conceptual Exclusive Contribution']
        full_r2 = row['Full R-squared']
        assert np.isclose(exclusive, full_r2, atol=1e-10), \
            "Exclusive contribution should equal full R-squared with single predictor"
    
    def test_variance_partitioning_empty_input_validation(self):
        """Test error handling for empty inputs"""
        
        # Test empty DV dictionary
        with pytest.raises((ValueError, KeyError, IndexError)):
            variance_partitioning({}, self.rdm_dict, plot_title='Empty DV Test', print_results=False)
        
        # Test empty predictor dictionary
        with pytest.raises((ValueError, KeyError, IndexError, AttributeError)):
            variance_partitioning(self.single_dv, {}, plot_title='Empty Predictor Test', print_results=False)
        
        # Additional test: Both empty
        with pytest.raises((ValueError, KeyError, IndexError, AttributeError)):
            variance_partitioning({}, {}, plot_title='Both Empty Test', print_results=False)
    
    def test_variance_partitioning_mismatched_rdm_sizes(self):
        """Test error handling for RDMs of different sizes"""
        # Create a smaller RDM
        small_rdm = np.random.rand(3, 3)
        mismatched_predictors = {
            'Normal': self.conceptual_rdm,  # 6x6
            'Small': small_rdm              # 3x3
        }
        
        # This should raise an error during standardization
        with pytest.raises((ValueError, IndexError)):
            variance_partitioning(
                self.single_dv, mismatched_predictors,
                plot_title='Mismatched Sizes Test', print_results=False
            )
    
    def test_variance_partitioning_with_nan_rdm(self):
        """Test handling of RDMs containing NaN values"""
        nan_rdm = self.conceptual_rdm.copy()
        nan_rdm[0, 1] = np.nan
        nan_rdm[1, 0] = np.nan
        
        nan_predictors = {
            'Normal': self.physical_rdm,
            'WithNaN': nan_rdm
        }
        
        # The function should handle NaN values through clean_data_df
        # It might reduce the data size but shouldn't crash
        try:
            results_df = variance_partitioning(
                self.single_dv, nan_predictors,
                plot_title='NaN Test', print_results=False
            )
            # If it succeeds, check that we get valid results
            assert isinstance(results_df, pd.DataFrame), "Should return valid DataFrame"
            assert not results_df.empty, "Results should not be empty"
        except ValueError:
            # It's also acceptable if the function raises a clear error
            pass
    
    def test_variance_partitioning_print_results_flag(self):
        """Test that print_results flag doesn't affect results"""
        results_no_print = variance_partitioning(
            self.single_dv, self.rdm_dict,
            plot_title='No Print Test', print_results=False
        )
        
        results_with_print = variance_partitioning(
            self.single_dv, self.rdm_dict,
            plot_title='With Print Test', print_results=True
        )
        
        # Results should be identical regardless of print flag
        pd.testing.assert_frame_equal(results_no_print, results_with_print)
    
    def test_variance_partitioning_different_plot_titles(self):
        """Test that different plot titles don't affect results"""
        results1 = variance_partitioning(
            self.single_dv, self.rdm_dict,
            plot_title='Title 1', print_results=False
        )
        
        results2 = variance_partitioning(
            self.single_dv, self.rdm_dict,
            plot_title='Title 2', print_results=False
        )
        
        # Results should be identical regardless of plot title
        pd.testing.assert_frame_equal(results1, results2)
    
    def test_standardize_rdms_helper(self):
        """Test the standardize_rdms helper function directly"""
        standardized = standardize_rdms(self.rdm_dict)
        
        # Check that we get the expected keys
        assert 'Conceptual' in standardized, "Conceptual should be in standardized results"
        assert 'Physical' in standardized, "Physical should be in standardized results"
        
        # Check that standardized values have approximately zero mean and unit variance
        # (allowing for small numerical errors)
        for name, vec in standardized.items():
            if len(vec) > 1:  # Skip if too few values to standardize
                assert np.isclose(np.mean(vec), 0, atol=1e-10), f"{name} should have zero mean"
                assert np.isclose(np.std(vec), 1, atol=1e-10), f"{name} should have unit variance"
    
    def test_standardize_rdms_constant_values(self):
        """Test standardize_rdms with constant RDM values"""
        # Create an RDM with all identical upper triangle values
        constant_rdm = np.ones((4, 4))
        np.fill_diagonal(constant_rdm, 0)  # Keep diagonal at zero
        
        constant_dict = {'Constant': constant_rdm}
        standardized = standardize_rdms(constant_dict)
        
        # Should return zeros for constant values
        assert np.allclose(standardized['Constant'], 0), "Constant values should be standardized to zeros"


# Helper function to run just the variance partitioning tests
def run_variance_partitioning_tests():
    """Run only the variance partitioning tests"""
    pytest.main([__file__ + "::TestVariancePartitioning", "-v"])


def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()