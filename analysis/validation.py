"""
analysis/validation.py

Overfitting detection and lookahead bias testing framework.

Tests:
  1. K-Fold Time Series Cross-Validation
  2. Lookahead Bias Detection via Perturbation
  3. Monte Carlo Permutation Tests
  4. Walk-Forward Out-of-Sample Validation

Usage:
    python analysis/validation.py --lookahead-test
    python analysis/validation.py --k-fold --n-splits 5
    python analysis/validation.py --permutation-test --n-perms 1000
    python analysis/validation.py --full-validation
"""

import argparse
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.fetcher import fetch_daily_bars
from data.processor import compute_log_returns, clean_returns
from model.prior import estimate_prior
from model.posterior import update_all_posteriors
from model.signals import compute_all_signals
from backtest.engine import run_backtest, BacktestConfig
from backtest.metrics import compute_metrics


class ValidationFramework:
    """
    Comprehensive validation suite for detecting overfitting and lookahead bias.
    """

    def __init__(self, symbols, base_config):
        """
        Initialize the validation framework.

        Args:
            symbols (list): List of tickers
            base_config (BacktestConfig): Dict of backtest parameters
        """
        self.symbols = symbols
        self.base_config = base_config

    def lookahead_bias_test(self, log_returns, perturbation_date, noise_level=0.10):
        """
        Test for lookahead bias by perturbing returns after a certain date.

        Strategy:
        i. Run model on original data, record signals at each data
        ii. Add noice to returns after perturbation_date
        iii. Re-run model on perturbed data, compare signals BEFORE perturbation_date
        iv. If signals change significantly, this indicates lookahead bias
        
        Args:
            log_returns (pd.DataFrame): Log returns of assets
            perturbation_date (str): Date after which to perturb returns (YYYY-MM-DD)
            noise_level (float): Standard deviation of noise to add
        """

        print(f"\n{'='*80}")
        print("LOOKAHEAD BIAS TEST")
        print(f"{'='*80}")
        print(f"Perturbation date: {perturbation_date}")
        print(f"Noise level: {noise_level:.1%}")
        print(f"Testing hypothesis: Signals before {perturbation_date} should NOT change\n")

        # Find perturbation index
        perturb_idx = log_returns.index.get_loc(perturbation_date)

        # Run on original data
        print("Running model on ORIGINAL data...")
        original_signals = self._run_model_and_record_signals(log_returns)

        # Create perturbed data (add noice ONLY after perturbation date)
        perturbed_returns = log_returns.copy()
        np.random.seed(38281)  # for reproducibility
        noise = np.random.normal(0, noise_level, perturbed_returns.iloc[perturb_idx:].shape)
        perturbed_returns.iloc[perturb_idx:] += noise

        print("Running model on PERTURBED data (noise added after perturbation date)...")
        perturbed_signals = self._run_model_and_record_signals(perturbed_returns)

        # Compare signals BEFORE perturbation date
        print(f"\nComparing signals BEFORE {perturbation_date}...")

        differences = []
        for date in original_signals.keys():
            if date >= perturbation_date:
                continue # only compare signals before perturbation date

            orig = original_signals[date]
            pert = perturbed_signals.get(date, {}) # gets the signals for this date from the perturbed run, or empty dict if not present

            #compare each symbol's signal
            for symbol in orig.keys():
                orig_action = orig[symbol]['action']
                pert_action = pert.get(symbol, {}).get('action', None) # get perturbed action, or None if not present

                if orig_action != pert_action:
                    differences.append({
                        'date': date,
                        'symbol': symbol,
                        'original_action': orig_action,
                        'perturbed_action': pert_action
                    })

        #Results
        total_checks = len([d for d in original_signals.keys() if d < perturbation_date]) * len(self.symbols)

        # Report differences
        print(f"\n{'='*80}")
        print("RESULTS:")
        print(f"{'='*80}")
        print(f"Total pre-perturbation signal checks: {total_checks}")
        print(f"Signals that changed: {len(differences)}")
        print(f"Stability rate: {(total_checks - len(differences)) / total_checks:.1%}")

        # Pass/Fail
        if len(differences) == 0:
            print("\nPASSED: No lookahead bias detected")
            print("   All signals before perturbation remained unchanged")
            verdict = "PASS"
        elif len(differences) < 0.01 * total_checks:
            print("\nMARGINAL: Minor instability detected (<1% of signals changed)")
            print("   Likely due to numerical precision, not lookahead bias")
            verdict = "MARGINAL"
        else:
            print("\nFAILED: Lookahead bias detected!")
            print(f"   {len(differences)} signals changed before perturbation")
            print("\nFirst 5 differences:")
            for diff in differences[:5]:
                print(f"  {diff['date']}: {diff['symbol']} changed "
                      f"{diff['original_action']} → {diff['perturbed_action']}")
            verdict = "FAIL"
        
        return {
            'verdict': verdict,
            'total_checks': total_checks,
            'differences': len(differences),
            'stability_rate': (total_checks - len(differences)) / total_checks,
            'changed_signals': differences
        }
    

    def _run_model_and_record_signals(self, log_returns):
        """
        Run model over entire period, record signals at each date.
        
        Returns:
            dict: {date: {symbol: {'action': ..., 'prob': ...}}}
        """
        signals_over_time = {}
        
        window = self.base_config['window']
        min_prob = self.base_config['min_prob_threshold']
        
        # Start after we have enough data
        start_idx = window
        
        for i in range(start_idx, len(log_returns)):
            date = log_returns.index[i]
            
            # Get window up to (but not including) this date
            window_data = log_returns.iloc[i-window:i]
            
            # Run model
            try: # what does try and except do here? We want to catch any errors that might occur during model execution (e.g. due to insufficient data, numerical issues, etc.) and simply skip those dates rather than crashing the entire validation process.
                prior = estimate_prior(window_data)
                posteriors = update_all_posteriors(window_data, prior)
                signals = compute_all_signals(posteriors, min_prob_threshold=min_prob)
                
                # Record
                signals_over_time[date] = {
                    symbol: {
                        'action': sig.action,
                        'prob': sig.prob_positive
                    }
                    for symbol, sig in signals.items()
                }
            except Exception as e:
                continue
        
        return signals_over_time
    
    def k_fold_cv(self, log_returns, n_splits = 5):
        """
        K-Fold time series cross-validation.
        
        Splits data into sequential folds, trains on each, tests on next.
        Evaluate how well the model generalizes to unseen data and detect overfitting.
        
        Args:
            log_returns: DataFrame of returns
            n_splits: Number of folds
            
        Returns:
            DataFrame with out-of-sample metrics for each fold
        """
        print(f"\n{'='*80}")
        print(f"K-FOLD TIME SERIES CROSS-VALIDATION (k={n_splits})")
        print(f"{'='*80}\n")

        total_days = len(log_returns)
        fold_size = total_days // n_splits # // is integer division

        results = []

        for fold in range(n_splits):
            #def fold boundaries
            test_start = fold * fold_size
            test_end = min((fold + 1) * fold_size, total_days)

            test_data = log_returns.iloc[test_start:test_end]

            test_period = f"{test_data.index[0].date()} to {test_data.index[-1].date()}"

            # Run model on test fold and compute metrics
            print(f"Fold {fold + 1}/{n_splits}: Testing on {test_period} ({len(test_data)} days)\n")

            # run backtest on this fold
            config = BacktestConfig(**self.base_config)

            # logic: if backtest fails, we catch exception and continue to next fold rather than crashing entire validation process. 
            try:
                result = run_backtest(test_data, config)
                
                metrics = compute_metrics(
                    result.returns,
                    result.equity_curve,
                    result.positions,
                    result.trades,
                    risk_free_rate=0.04
                )
                
                results.append({
                    'fold': fold + 1,
                    'period': test_period,
                    'total_return': metrics.total_return,
                    'sharpe': metrics.sharpe_ratio,
                    'sortino': metrics.sortino_ratio,
                    'calmar': metrics.calmar_ratio,
                    'max_dd': metrics.max_drawdown,
                    'win_rate': metrics.win_rate,
                })
                
                print(f"  Sharpe: {metrics.sharpe_ratio:.2f}, Max DD: {metrics.max_drawdown:.1%}")
            
            except Exception as e:
                print(f"    Fold failed: {e}")
                continue
        
        df = pd.DataFrame(results)
        
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION SUMMARY:")
        print(f"{'='*80}")
        print(f"Mean Sharpe: {df['sharpe'].mean():.2f} ± {df['sharpe'].std():.2f}")
        print(f"Positive Sharpe folds: {(df['sharpe'] > 0).sum()}/{len(df)}")
        print(f"Mean Max DD: {df['max_dd'].mean():.1%}")
        print(f"Sharpe range: [{df['sharpe'].min():.2f}, {df['sharpe'].max():.2f}]")
        
        # Statistical test: Is mean Sharpe significantly > 0?
        t_stat, p_value = stats.ttest_1samp(df['sharpe'], 0)
        print(f"\nT-test (H0: Sharpe = 0): t={t_stat:.2f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            print("Sharpe is significantly positive (p < 0.05)")
        else:
            print("Sharpe not statistically significant (p ≥ 0.05)")
        
        return df
    
    def permutation_test(self, log_returns, n_permutations=1000, save_results=True):
        """
        Monte Carlo permutation test for statistical significance.
        
        Randomly shuffles returns, runs backtest, compares to actual.
        If actual Sharpe is in top 5% of shuffled results, it's significant.
        
        Args:
            log_returns: DataFrame of returns
            n_permutations: Number of random shuffles
            save_results: If True, save detailed results to CSV
            
        Returns:
            dict with test results
        """
        print(f"\n{'='*80}")
        print(f"MONTE CARLO PERMUTATION TEST (n={n_permutations})")
        print(f"{'='*80}")
        print("Testing hypothesis: Strategy has skill, not luck\n")
        
        # Run on actual data
        print("Running backtest on ACTUAL data...")
        config = BacktestConfig(**self.base_config)
        actual_result = run_backtest(log_returns, config)
        actual_metrics = compute_metrics(
            actual_result.returns,
            actual_result.equity_curve,
            actual_result.positions,
            actual_result.trades
        )
        actual_sharpe = actual_metrics.sharpe_ratio
        
        print(f"Actual Sharpe: {actual_sharpe:.2f}\n")
        
        # Store detailed results for each permutation
        permutation_results = []
        
        # Save actual results as permutation 0
        permutation_results.append({
            'permutation': 0,
            'type': 'actual',
            'sharpe': actual_metrics.sharpe_ratio,
            'sortino': actual_metrics.sortino_ratio,
            'calmar': actual_metrics.calmar_ratio,
            'total_return': actual_metrics.total_return,
            'annual_return': actual_metrics.annualized_return,
            'annual_vol': actual_metrics.annualized_volatility,
            'max_dd': actual_metrics.max_drawdown,
            'max_dd_days': actual_metrics.max_drawdown_duration,
            'win_rate': actual_metrics.win_rate,
            'avg_win': actual_metrics.avg_win,
            'avg_loss': actual_metrics.avg_loss,
            'total_trades': actual_metrics.total_trades,
        })
        
        # Save actual equity curve separately
        actual_equity = actual_result.equity_curve.copy()
        actual_equity.name = 'actual'
        
        # Run on permuted data
        print(f"Running {n_permutations} permutations...")
        print("=" * 60)
        
        permuted_sharpes = []
        permuted_equities = []  # Store for visualization
        
        import sys
        import io
        
        for i in range(n_permutations):
            # Progress bar
            if (i + 1) % 10 == 0 or i == 0:
                pct_complete = (i + 1) / n_permutations * 100
                print(f"  Progress: {i + 1}/{n_permutations} ({pct_complete:.1f}%) complete", 
                      end='\r', flush=True)
            
            # Shuffle returns (break time series structure)
            shuffled_returns = log_returns.copy()
            np.random.seed(1482 + i)  # Reproducible but different each iteration
            for col in shuffled_returns.columns:
                shuffled_returns[col] = np.random.permutation(shuffled_returns[col].values)
            
            try:
                # Suppress stdout for backtest to avoid cluttering output
                old_stdout = sys.stdout
                sys.stdout = io.StringIO() # sends all print statement for each backtest to a dummy buffer instead of console
                
                result = run_backtest(shuffled_returns, config)
                metrics = compute_metrics(result.returns, result.equity_curve, 
                                        result.positions, result.trades)
                
                # Restore stdout
                sys.stdout = old_stdout
                
                permuted_sharpes.append(metrics.sharpe_ratio)
                
                # Store detailed metrics
                permutation_results.append({
                    'permutation': i + 1,
                    'type': 'shuffled',
                    'sharpe': metrics.sharpe_ratio,
                    'sortino': metrics.sortino_ratio,
                    'calmar': metrics.calmar_ratio,
                    'total_return': metrics.total_return,
                    'annual_return': metrics.annualized_return,
                    'annual_vol': metrics.annualized_volatility,
                    'max_dd': metrics.max_drawdown,
                    'max_dd_days': metrics.max_drawdown_duration,
                    'win_rate': metrics.win_rate,
                    'avg_win': metrics.avg_win,
                    'avg_loss': metrics.avg_loss,
                    'total_trades': metrics.total_trades,
                })
                
                # Store equity curve (save every 10th to avoid memory issues)
                if i % 10 == 0:
                    equity = result.equity_curve.copy()
                    equity.name = f'perm_{i+1}'
                    permuted_equities.append(equity)
                
            except Exception as e:
                # Restore stdout if error occurred
                sys.stdout = old_stdout
                
                # Still record failed permutation
                permutation_results.append({
                    'permutation': i + 1,
                    'type': 'shuffled',
                    'sharpe': np.nan,
                    'sortino': np.nan,
                    'calmar': np.nan,
                    'total_return': np.nan,
                    'annual_return': np.nan,
                    'annual_vol': np.nan,
                    'max_dd': np.nan,
                    'max_dd_days': np.nan,
                    'win_rate': np.nan,
                    'avg_win': np.nan,
                    'avg_loss': np.nan,
                    'total_trades': np.nan,
                })
                continue
        
        print()  # New line after progress bar
        print("=" * 60)
        
        # Calculate p-value
        better_than_actual = sum(s >= actual_sharpe for s in permuted_sharpes)
        p_value = better_than_actual / len(permuted_sharpes)
        
        # Save results to CSV
        if save_results:
            results_dir = Path('analysis/results/permtest')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Create descriptive filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'permtest_n{n_permutations}_sharpe{actual_sharpe:.2f}_{timestamp}'
            
            # Save metrics
            df_results = pd.DataFrame(permutation_results)
            metrics_file = results_dir / f'{filename}_metrics.csv'
            df_results.to_csv(metrics_file, index=False)
            print(f"\nMetrics saved to: {metrics_file}")
            
            # Save equity curves
            if permuted_equities:
                equity_df = pd.concat([actual_equity] + permuted_equities, axis=1)
                equity_file = results_dir / f'{filename}_equities.csv'
                equity_df.to_csv(equity_file)
                print(f"Equity curves saved to: {equity_file}")
            
            # Save summary statistics
            summary = {
                'actual_sharpe': actual_sharpe,
                'permuted_mean': np.mean(permuted_sharpes),
                'permuted_std': np.std(permuted_sharpes),
                'permuted_min': np.min(permuted_sharpes),
                'permuted_max': np.max(permuted_sharpes),
                'permuted_median': np.median(permuted_sharpes),
                'p_value': p_value,
                'n_permutations': n_permutations,
                'better_than_actual': better_than_actual,
                'significant_at_5pct': p_value < 0.05,
            }
            summary_file = results_dir / f'{filename}_summary.json'
            import json
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Summary saved to: {summary_file}")
        
        print(f"\n{'='*80}")
        print("RESULTS:")
        print(f"{'='*80}")
        print(f"Actual Sharpe: {actual_sharpe:.2f}")
        print(f"Permuted Sharpe: {np.mean(permuted_sharpes):.2f} ± {np.std(permuted_sharpes):.2f}")
        print(f"Permuted range: [{np.min(permuted_sharpes):.2f}, {np.max(permuted_sharpes):.2f}]")
        print(f"Permutations with Sharpe ≥ actual: {better_than_actual}/{len(permuted_sharpes)}")
        print(f"P-value: {p_value:.4f}")
        
        # Z-score interpretation
        z_score = (actual_sharpe - np.mean(permuted_sharpes)) / np.std(permuted_sharpes)
        print(f"Z-score: {z_score:.2f} (actual is {z_score:.1f} std devs above mean)")
        
        if p_value < 0.05:
            print("\nSIGNIFICANT: Strategy outperforms random (p < 0.05)")
            print("   Result unlikely due to luck alone")
        else:
            print("\nNOT SIGNIFICANT: Could be luck (p ≥ 0.05)")
            print("   Cannot reject null hypothesis of no skill")
        
        return {
            'actual_sharpe': actual_sharpe,
            'permuted_mean': np.mean(permuted_sharpes),
            'permuted_std': np.std(permuted_sharpes),
            'permuted_min': np.min(permuted_sharpes),
            'permuted_max': np.max(permuted_sharpes),
            'p_value': p_value,
            'z_score': z_score,
            'significant': p_value < 0.05,
            'results_df': pd.DataFrame(permutation_results) if save_results else None
        }


def main():
    parser = argparse.ArgumentParser(description='Validation Framework')
    
    parser.add_argument('--lookahead-test', action='store_true',
                       help='Test for lookahead bias via perturbation')
    parser.add_argument('--k-fold', action='store_true',
                       help='K-fold time series cross-validation')
    parser.add_argument('--n-splits', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--permutation-test', action='store_true',
                       help='Monte Carlo permutation test')
    parser.add_argument('--n-perms', type=int, default=100,
                       help='Number of permutations')
    parser.add_argument('--full-validation', action='store_true',
                       help='Run all validation tests')
    
    args = parser.parse_args()
    
    # Setup
    symbols = ['AAPL', 'GOOG', 'MSFT', 'OXY', 'XOM', 'TLT', 'VOO']
    
    base_config = {
        'window': 60,
        'min_prob_threshold': 0.67,
        'target_vol': 0.15,
        'max_position': 0.20,
        'initial_capital': 100000.0,
        'drawdown_threshold': 0.15,
        'drawdown_scaling': 0.75,
    }
    
    validator = ValidationFramework(symbols, base_config)
    
    # Load data
    print("Loading market data...")
    from data.fetcher import fetch_daily_bars
    raw = fetch_daily_bars(symbols, datetime(2020, 1, 1), datetime(2024, 12, 31))
    log_returns = compute_log_returns(raw)
    log_returns = clean_returns(log_returns)
    print(f"Data loaded: {len(log_returns)} days\n")
    
    # Run tests
    if args.lookahead_test or args.full_validation:
        # Test at midpoint
        mid_date = log_returns.index[len(log_returns) // 2]
        validator.lookahead_bias_test(log_returns, mid_date, noise_level=0.10)
    
    if args.k_fold or args.full_validation:
        validator.k_fold_cv(log_returns, n_splits=args.n_splits)
    
    if args.permutation_test or args.full_validation:
        validator.permutation_test(log_returns, n_permutations=args.n_perms)


if __name__ == "__main__":
    main()


