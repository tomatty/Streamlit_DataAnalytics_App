"""
Tests for sample size calculator module.
"""
import numpy as np
from scipy import stats
import pytest


class TestSampleSizeCalculations:
    """Test sample size calculation formulas."""

    def test_mean_estimation_basic(self):
        """Test basic sample size calculation for mean estimation."""
        confidence_level = 95
        margin_of_error = 0.5
        std_dev = 1.0

        alpha = 1 - (confidence_level / 100)
        z_score = stats.norm.ppf(1 - alpha / 2)

        n = ((z_score * std_dev) / margin_of_error) ** 2
        n = int(np.ceil(n))

        # For 95% confidence, z ≈ 1.96, so n ≈ (1.96 * 1 / 0.5)^2 ≈ 15.37 -> 16
        assert n >= 15
        assert n <= 16

    def test_mean_estimation_with_larger_std_dev(self):
        """Test sample size increases with larger standard deviation."""
        confidence_level = 95
        margin_of_error = 0.5
        alpha = 1 - (confidence_level / 100)
        z_score = stats.norm.ppf(1 - alpha / 2)

        std_dev_small = 1.0
        std_dev_large = 2.0

        n_small = ((z_score * std_dev_small) / margin_of_error) ** 2
        n_large = ((z_score * std_dev_large) / margin_of_error) ** 2

        # Larger std dev should require larger sample size
        assert n_large > n_small

    def test_mean_estimation_with_smaller_margin(self):
        """Test sample size increases with smaller margin of error."""
        confidence_level = 95
        std_dev = 1.0
        alpha = 1 - (confidence_level / 100)
        z_score = stats.norm.ppf(1 - alpha / 2)

        margin_large = 0.5
        margin_small = 0.25

        n_large_margin = ((z_score * std_dev) / margin_large) ** 2
        n_small_margin = ((z_score * std_dev) / margin_small) ** 2

        # Smaller margin should require larger sample size
        assert n_small_margin > n_large_margin

    def test_proportion_estimation_basic(self):
        """Test basic sample size calculation for proportion estimation."""
        confidence_level = 95
        margin_of_error = 0.05
        expected_proportion = 0.5

        alpha = 1 - (confidence_level / 100)
        z_score = stats.norm.ppf(1 - alpha / 2)

        p = expected_proportion
        n = (z_score ** 2 * p * (1 - p)) / (margin_of_error ** 2)
        n = int(np.ceil(n))

        # For 95% confidence, p=0.5, margin=0.05:
        # n ≈ (1.96^2 * 0.5 * 0.5) / 0.05^2 ≈ 384.16 -> 385
        assert n >= 384
        assert n <= 385

    def test_proportion_estimation_p_equals_half_is_max(self):
        """Test that p=0.5 gives maximum (most conservative) sample size."""
        confidence_level = 95
        margin_of_error = 0.05
        alpha = 1 - (confidence_level / 100)
        z_score = stats.norm.ppf(1 - alpha / 2)

        p_half = 0.5
        p_other = 0.3

        n_half = (z_score ** 2 * p_half * (1 - p_half)) / (margin_of_error ** 2)
        n_other = (z_score ** 2 * p_other * (1 - p_other)) / (margin_of_error ** 2)

        # p=0.5 should give the largest sample size
        assert n_half > n_other

    def test_independence_test_basic(self):
        """Test basic sample size calculation for independence test."""
        alpha = 0.05
        power = 0.80
        effect_size = 0.3
        df = 1

        z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)

        n = ((z_alpha + z_beta) ** 2) / (effect_size ** 2) + df + 1
        n = int(np.ceil(n))

        # With these parameters, n should be around 87-88
        assert n >= 85
        assert n <= 90

    def test_independence_test_smaller_effect_needs_more_samples(self):
        """Test that smaller effect size requires larger sample size."""
        alpha = 0.05
        power = 0.80
        df = 1

        z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)

        effect_small = 0.1
        effect_large = 0.5

        n_small = ((z_alpha + z_beta) ** 2) / (effect_small ** 2) + df + 1
        n_large = ((z_alpha + z_beta) ** 2) / (effect_large ** 2) + df + 1

        # Smaller effect should require larger sample size
        assert n_small > n_large

    def test_independence_test_higher_power_needs_more_samples(self):
        """Test that higher power requires larger sample size."""
        alpha = 0.05
        effect_size = 0.3
        df = 1

        z_alpha = stats.norm.ppf(1 - alpha)

        power_low = 0.70
        power_high = 0.90

        z_beta_low = stats.norm.ppf(power_low)
        z_beta_high = stats.norm.ppf(power_high)

        n_low = ((z_alpha + z_beta_low) ** 2) / (effect_size ** 2) + df + 1
        n_high = ((z_alpha + z_beta_high) ** 2) / (effect_size ** 2) + df + 1

        # Higher power should require larger sample size
        assert n_high > n_low

    def test_finite_population_correction_reduces_sample_size(self):
        """Test that finite population correction reduces required sample size."""
        n_uncorrected = 100
        population_size = 500

        n_corrected = n_uncorrected / (1 + ((n_uncorrected - 1) / population_size))
        n_corrected = int(np.ceil(n_corrected))

        # Corrected sample size should be smaller
        assert n_corrected < n_uncorrected
        assert n_corrected >= 83  # Expected: 83.47 -> 84
        assert n_corrected <= 84

    def test_z_score_values(self):
        """Test that z-scores are calculated correctly for common confidence levels."""
        # 90% confidence level
        z_90 = stats.norm.ppf(1 - 0.10 / 2)
        assert abs(z_90 - 1.645) < 0.001

        # 95% confidence level
        z_95 = stats.norm.ppf(1 - 0.05 / 2)
        assert abs(z_95 - 1.960) < 0.001

        # 99% confidence level
        z_99 = stats.norm.ppf(1 - 0.01 / 2)
        assert abs(z_99 - 2.576) < 0.001
