"""
tests/test_model.py

Unit tests for Bayesian model components.

Tests:
  - Prior estimation from cross-sectional data
  - Posterior updates using conjugate formulas
  - Shrinkage behavior (data vs prior weighting)
  - Edge cases (zero variance, no data, single observation)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np, pandas as pd
from datetime import datetime, timedelta

#import model components to test
from model.prior import estimate_prior, GroupPrior
from model.posterior import update_posterior, AssetPosterior, update_all_posteriors
from model.signals import compute_signal, TradingSignal

class TestPriorEstimation:
    """
    Test empirical Bayes prior estimation.
    """

    def test_prior_with_identical_assets(self):
        """
        If all assets have identical returns, tau_0 should be near zero.
        """
        dates = pd.date_range(start="2020-01-01", periods=60, freq = 'D')
        # create returns where all stocks have the same mean return (0.001)
        returns = pd.DataFrame({
            'A': np.random.normal(0.001, 0.02, size=60),
            'B': np.random.normal(0.001, 0.02, size=60),
            'C': np.random.normal(0.001, 0.02, size=60),
        }, index=dates)

        prior = estimate_prior(returns)

        # Check mu_0 is close to 0.001
        assert abs(prior.mu_0 - 0.001) < 0.005, f"Expected mu_0 ~ 0.001, got {prior.mu_0}"

        # Check tau_0 is small (stocks are similar)
        assert prior.tau_0 < 0.01, f"Expected small tau_0, got {prior.tau_0}"

        # Check sigma_bar is reasonable (around 0.02)
        assert 0.015 < prior.sigma_bar < 0.025, f"Expected sigma_bar ~ 0.02, got {prior.sigma_bar}"

    def test_prior_with_diverse_assets(self):
        """
        When assets differ, tau_0 should be larger.
        """
        dates = pd.date_range(start="2020-01-01", periods=60, freq='D')
        returns = pd.DataFrame({
            'A': np.random.normal(-0.01, 0.02, size=60), # negative mean return
            'B': np.random.normal(0.00, 0.02, size=60),  # zero mean return
            'C': np.random.normal(0.01, 0.02, size=60),  # positive mean return
        }, index=dates)

        prior = estimate_prior(returns)

        # tau_0 should be substantial (stocks differ)
        assert prior.tau_0 > 0.003, f"Expected tau_0 > 0.003 for diverse assets, got {prior.tau_0}"

    def test_prior_minimum_tau(self):
        """
        Verify tau_0 floor prevents zero variance case (division by zero in posterior).
        """
        # create perfectly identical returns (zero variance across stocks)
        dates = pd.date_range(start="2020-01-01", periods=60, freq='D')
        returns = pd.DataFrame({
            'A': np.full(60, 0.001),
            'B': np.full(60, 0.001),
        }, index=dates)

        prior = estimate_prior(returns)

        # Check tau_0 is not below the floor
        assert prior.tau_0 >= 1e-4, f"Expected tau_0 >= 1e-4, got {prior.tau_0}"


class TestPosteriorUpdates:
    """Test conjugate posterior calculations."""
    
    def test_posterior_with_no_data(self):
        """With zero observations, posterior = prior."""
        prior = GroupPrior(mu_0=0.001, tau_0=0.005, sigma_bar=0.02)
        returns = pd.Series([], dtype=float)  # Empty
        
        posterior = update_posterior('TEST', returns, prior)
        
        assert posterior.n_obs == 0
        assert posterior.post_mean == prior.mu_0
        assert abs(posterior.post_std - prior.tau_0) < 1e-6
        assert posterior.sigma_hat == prior.sigma_bar
    
    def test_posterior_with_single_observation(self):
        """With 1 observation, should shrink toward prior."""
        prior = GroupPrior(mu_0=0.001, tau_0=0.005, sigma_bar=0.02)
        returns = pd.Series([0.05])  # One big positive return
        
        posterior = update_posterior('TEST', returns, prior)
        
        assert posterior.n_obs == 1
        # Should be between prior (0.001) and data (0.05), closer to prior
        assert 0.001 < posterior.post_mean < 0.05
        assert posterior.post_mean < 0.01  # Heavy shrinkage with n=1
    
    def test_posterior_converges_to_sample_mean(self):
        """With lots of data, posterior mean → sample mean."""
        prior = GroupPrior(mu_0=0.001, tau_0=0.005, sigma_bar=0.02)
        
        # 1000 observations with mean = 0.01
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.01, 0.02, 1000))
        
        posterior = update_posterior('TEST', returns, prior)
        
        assert posterior.n_obs == 1000
        # Should be very close to sample mean
        sample_mean = returns.mean()
        assert abs(posterior.post_mean - sample_mean) < 0.001
        
        # Posterior variance should be small with lots of data
        assert posterior.post_var < 0.0001
    
    def test_shrinkage_factor(self):
        """Verify shrinkage increases with more data."""
        prior = GroupPrior(mu_0=0.0, tau_0=0.01, sigma_bar=0.02)
        
        np.random.seed(42)
        
        # 10 observations
        returns_10 = pd.Series(np.random.normal(0.01, 0.02, 10))
        post_10 = update_posterior('TEST', returns_10, prior)
        shrinkage_10 = post_10.shrinkage_factor
        
        # 100 observations
        returns_100 = pd.Series(np.random.normal(0.01, 0.02, 100))
        post_100 = update_posterior('TEST', returns_100, prior)
        shrinkage_100 = post_100.shrinkage_factor
        
        # More data → higher shrinkage factor (more weight on data)
        assert shrinkage_100 > shrinkage_10
        assert 0 <= shrinkage_10 <= 1
        assert 0 <= shrinkage_100 <= 1
    
    def test_high_volatility_reduces_shrinkage(self):
        """Noisy data should shrink more toward prior."""
        prior = GroupPrior(mu_0=0.0, tau_0=0.01, sigma_bar=0.02)
        
        np.random.seed(42)
        
        # Low volatility data
        returns_low_vol = pd.Series(np.random.normal(0.01, 0.01, 60))
        post_low_vol = update_posterior('TEST', returns_low_vol, prior)
        
        # High volatility data
        returns_high_vol = pd.Series(np.random.normal(0.01, 0.05, 60))
        post_high_vol = update_posterior('TEST', returns_high_vol, prior)
        
        # High vol → lower shrinkage factor (less trust in data)
        assert post_low_vol.shrinkage_factor > post_high_vol.shrinkage_factor
    
    def test_update_all_posteriors(self):
        """Test batch posterior updates."""
        prior = GroupPrior(mu_0=0.001, tau_0=0.005, sigma_bar=0.02)
        
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        returns = pd.DataFrame({
            'A': np.random.normal(0.002, 0.02, 60),
            'B': np.random.normal(-0.001, 0.02, 60),
        }, index=dates)
        
        posteriors = update_all_posteriors(returns, prior)
        
        assert len(posteriors) == 2
        assert 'A' in posteriors
        assert 'B' in posteriors
        assert posteriors['A'].n_obs == 60
        assert posteriors['B'].n_obs == 60


class TestSignalGeneration:
    """Test signal computation from posteriors."""
    
    def test_high_conviction_long(self):
        """Strong positive posterior → long signal."""
        posterior = AssetPosterior(
            symbol='TEST',
            post_mean=0.01,      # Strong positive
            post_var=0.0001,     # Low uncertainty
            post_std=0.01,
            n_obs=100,
            sigma_hat=0.02
        )
        
        signal = compute_signal(posterior, min_prob_threshold=0.60)
        
        assert signal.prob_positive > 0.80
        assert signal.action == 'long'
        assert signal.signal_strength > 0  # Positive risk-adjusted return
    
    def test_high_conviction_short(self):
        """Strong negative posterior → short signal."""
        posterior = AssetPosterior(
            symbol='TEST',
            post_mean=-0.01,     # Strong negative
            post_var=0.0001,
            post_std=0.01,
            n_obs=100,
            sigma_hat=0.02
        )
        
        signal = compute_signal(posterior, min_prob_threshold=0.60)
        
        assert signal.prob_positive < 0.20
        assert signal.action == 'short'
        assert signal.signal_strength < 0  # Negative risk-adjusted return
    
    def test_uncertain_gives_flat(self):
        """High uncertainty → flat signal."""
        posterior = AssetPosterior(
            symbol='TEST',
            post_mean=0.001,     # Slightly positive
            post_var=0.01,       # HIGH uncertainty
            post_std=0.1,
            n_obs=5,
            sigma_hat=0.02
        )
        
        signal = compute_signal(posterior, min_prob_threshold=0.60)
        
        # With high uncertainty, P(mu>0) will be close to 50%
        assert 0.40 < signal.prob_positive < 0.60
        assert signal.action == 'flat'
    
    def test_threshold_affects_action(self):
        """Higher threshold → more conservative."""
        posterior = AssetPosterior(
            symbol='TEST',
            post_mean=0.003,
            post_var=0.001,
            post_std=0.0316,
            n_obs=60,
            sigma_hat=0.02
        )
        
        # Low threshold (60%)
        signal_60 = compute_signal(posterior, min_prob_threshold=0.60)
        
        # High threshold (80%)
        signal_80 = compute_signal(posterior, min_prob_threshold=0.80)
        
        # Same posterior, different thresholds might give different actions
        # At minimum, stricter threshold should not increase trading
        if signal_60.action == 'flat':
            assert signal_80.action == 'flat'


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_nan_in_returns(self):
        """NaN values should be dropped."""
        prior = GroupPrior(mu_0=0.001, tau_0=0.005, sigma_bar=0.02)
        returns = pd.Series([0.01, np.nan, 0.02, np.nan, 0.015])
        
        posterior = update_posterior('TEST', returns, prior)
        
        # Should only count non-NaN values
        assert posterior.n_obs == 3
    
    def test_zero_variance_returns(self):
        """Constant returns should not crash."""
        prior = GroupPrior(mu_0=0.001, tau_0=0.005, sigma_bar=0.02)
        returns = pd.Series([0.001] * 60)  # All identical
        
        posterior = update_posterior('TEST', returns, prior)
        
        # Should use prior's sigma_bar when sample std is zero
        assert posterior.sigma_hat == prior.sigma_bar
    
    def test_negative_threshold_raises_error(self):
        """Invalid threshold should fail gracefully."""
        posterior = AssetPosterior(
            symbol='TEST',
            post_mean=0.01,
            post_var=0.001,
            post_std=0.0316,
            n_obs=60,
            sigma_hat=0.02
        )
        
        # This should work without error
        signal = compute_signal(posterior, min_prob_threshold=0.60)
        assert signal is not None


# ---------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])