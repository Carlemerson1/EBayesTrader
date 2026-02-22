"""
analysis/parameter_study.py

Robust parameter analysis framework.

Runs backtests across multiple parameter combinations and time periods
to identify stable, robust configurations.

Usage:
    python3 analysis/parameter_study.py --sweep <parameter>
    python3 analysis/parameter_study.py --full-grid 
    python3 analysis/parameter_study.py --walk-forward <parameter>
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


import argparse
import itertools
import pandas as pd, numpy as np
from datetime import datetime, timedelta
import json

from data.fetcher import fetch_daily_bars
from data.processor import clean_returns, compute_log_returns
from backtest.engine import run_backtest, BacktestConfig
from backtest.metrics import compute_metrics


class ParameterStudy:
    """
    Conducts systematic parameter analysis.
    """
    
    def __init__(self, symbols, output_dir='analysis/results'):
        """
        Initialize parameter study.
        
        Args:
            symbols: List of ticker symbols
            output_dir: Directory to save results
        """
        self.symbols = symbols
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default parameter ranges
        self.param_grid = {
            'window': [40, 50, 60, 70, 90],
            'min_prob_threshold': [0.60, 0.65, 0.67, 0.70, 0.75],
            'target_vol': [0.12, 0.15, 0.18, 0.20],
            'max_position': [0.15, 0.20, 0.25],
            'drawdown_threshold': [0.10, 0.15, 0.20],
        }

        # Test periods for walk-forward analysis
        self.test_periods = [
            ('2020-01-01', '2021-12-31'),   # COVID era
            ('2022-01-01', '2022-12-31'),   # Bull to bear
            ('2022-01-01', '2023-12-31'),   # Bear market
            ('2023-01-01', '2024-12-31'),   # Recovery phase
            ('2020-01-01', '2024-12-31'),   # Full period
        ]

    def load_data(self, start_date, end_date):
        """Load market data for given period."""
        print(f".  Loading data: {start_date} to {end_date}")

        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        raw = fetch_daily_bars(self.symbols, start, end)
        log_returns=compute_log_returns(raw)
        log_returns=clean_returns(log_returns)

        return log_returns
    
    def run_single_backtest(self, log_returns, config_dict, verbose=False):
        """
        Run backtest with given configuration.
        
        Returns dict of metrics or None if failed.
        """
        config = BacktestConfig(**config_dict)
        
        # Check if we have enough data
        if len(log_returns) < config.window + 20:
            if verbose:
                print(f"    Insufficient data: {len(log_returns)} days, need {config.window + 20}")
            return None
        
        try:
            result = run_backtest(log_returns, config)
            
            metrics = compute_metrics(
                result.returns,
                result.equity_curve,
                result.positions,
                result.trades,
                risk_free_rate=0.03678
            )
            
            return {
                'total_return': metrics.total_return,
                'annual_return': metrics.annualized_return,
                'annual_vol': metrics.annualized_volatility,
                'sharpe': metrics.sharpe_ratio,
                'sortino': metrics.sortino_ratio,
                'calmar': metrics.calmar_ratio,
                'max_dd': metrics.max_drawdown,
                'max_dd_days': metrics.max_drawdown_duration,
                'win_rate': metrics.win_rate,
                'avg_win': metrics.avg_win,
                'avg_loss': metrics.avg_loss,
                'total_trades': metrics.total_trades,
                'avg_turnover': metrics.avg_turnover,
            }
        
        except Exception as e:
            if verbose:
                print(f"    Backtest failed: {type(e).__name__}: {e}")
            return None
        

    def sweep_single_parameter(self, param_name, base_config, log_returns, period_name):
        """
        Sweep one parameter while holding others constant.
        
        Args:
            param_name: Name of parameter to sweep
            base_config: Dict of baseline config
            log_returns: Market data
            period_name: Name of test period
            
        Returns:
            DataFrame with results
        """
        print(f"\nSweeping parameter: {param_name} over period: {period_name}")

        results = []

        for value in self.param_grid[param_name]:
            #create config with this parm value
            config = base_config.copy()
            config[param_name] = value

            print(f".  Testing {param_name}={value}... ")
            metrics = self.run_single_backtest(log_returns, config)

            if metrics:
                results.append({
                    'period': period_name,
                    'parameter': param_name,
                    'value': value,
                    **metrics
                })

        return pd.DataFrame(results)
    
    def full_grid_sweep(self, log_returns, period_name, max_combinations=100):
        """
        Test all parameter combinations (or random sample if too many).
        
        WARNING: Can be slow! Use max_combinations to limit.
        """
        # Generate all combinations
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[p] for p in param_names]
        
        all_combinations = list(itertools.product(*param_values))
        
        print(f"\nFull grid sweep: {len(all_combinations)} total combinations")
        
        if len(all_combinations) > max_combinations:
            print(f"   Sampling {max_combinations} random combinations...")
            np.random.seed(42)
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            combinations = [all_combinations[i] for i in indices]
        else:
            combinations = all_combinations
        
        results = []
        failures = []
        
        for i, combo in enumerate(combinations, 1):
            # Build config dict
            config = {param_names[j]: combo[j] for j in range(len(param_names))}
            config['initial_capital'] = 100000.0
            
            if i % 10 == 0 or i == 1:
                print(f"  Progress: {i}/{len(combinations)} "
                      f"(Successful: {len(results)}, Failed: {len(failures)})")
            
            metrics = self.run_single_backtest(log_returns, config)
            
            if metrics:
                results.append({
                    'period': period_name,
                    **config,
                    **metrics
                })
            else:
                failures.append(config)
        
        print(f"\nCompleted: {len(results)} successful, {len(failures)} failed")
        
        if len(results) == 0:
            print("No successful backtests! Check data length and parameters.")
            return pd.DataFrame()
        
        return pd.DataFrame(results)

    def walk_forward_analysis(self, param_name, base_config):
        """
        Test parameter across multiple time periods (walk-forward).
        
        Checks if parm setting is robust across different market regimes.
        """

        all_results = []

        for start_date, end_date in self.test_periods:
            period_name = f"{start_date[:4]}-{end_date[:4]}"

            #load data for this period
            log_returns = self.load_data(start_date, end_date)

            #sweep this parameter
            df = self.sweep_single_parameter(
                param_name,
                base_config,
                log_returns,
                period_name
            )
    
            all_results.append(df)

        combined = pd.concat(all_results, ignore_index=True)

        return combined
        
    def analyze_stability(self, results_df):
        """
        Analyze parameter stability across periods.
        
        Returns:
            DataFrame with stability metrics for each parameter value
        """
        print("\nAnalyzing parameter stability across periods...")

        stability_report = []

        for param_name in results_df['parameter'].unique():
            param_data = results_df[results_df['parameter'] == param_name]

            for value in param_data['value'].unique():
                value_data = param_data[param_data['value'] == value]

                # compute stats across periods

                #returns
                return_mean = value_data['total_return'].mean()
                return_vol = value_data['total_return'].std()
                return_min = value_data['total_return'].min()
                return_max = value_data['total_return'].max()

                #sharpe
                sharpe_mean = value_data['sharpe'].mean()
                sharpe_std = value_data['sharpe'].std()
                sharpe_min = value_data['sharpe'].min()
                sharpe_max = value_data['sharpe'].max()

                #sortino
                sortino_mean = value_data['sortino'].mean()
                sortino_std = value_data['sortino'].std()
                sortino_min = value_data['sortino'].min()
                sortino_max = value_data['sortino'].max()

                #calmar
                calmar_mean = value_data['calmar'].mean()
                calmar_std = value_data['calmar'].std()
                calmar_min = value_data['calmar'].min()
                calmar_max = value_data['calmar'].max()

                #drawdown
                dd_mean = value_data['max_dd'].mean()
                dd_std = value_data['max_dd'].std()
                dd_min = value_data['max_dd'].min()
                dd_max = value_data['max_dd'].max()

                #count positive sharpe periods
                positive_periods = (value_data['sharpe'] > 0).sum()
                total_periods = len(value_data)

                stability_report.append({
                    'parameter': param_name,
                    'value': value,
                    'return_mean': return_mean,
                    'return_vol': return_vol,
                    'return_min': return_min,
                    'return_max': return_max,
                    'sharpe_mean': sharpe_mean,
                    'sharpe_std': sharpe_std,
                    'sharpe_min': sharpe_min,
                    'sharpe_max': sharpe_max,
                    'sortino_mean': sortino_mean,
                    'sortino_std': sortino_std,
                    'sortino_min': sortino_min,
                    'sortino_max': sortino_max,
                    'calmar_mean': calmar_mean,
                    'calmar_std': calmar_std,
                    'calmar_min': calmar_min,
                    'calmar_max': calmar_max,
                    'dd_mean': dd_mean,
                    'dd_std': dd_std,
                    'dd_min': dd_min,
                    'dd_max': dd_max,
                    'positive_periods': positive_periods,
                    'total_periods': total_periods,
                    'positive_pct': positive_periods / total_periods * 100
                })

        return pd.DataFrame(stability_report).sort_values('sharpe_mean', ascending=False)
    
    def save_results(self, df, filename):
        """Save results to CSV with timestamp."""
        # Add timestamp to filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = filename.rsplit('.', 1)[0]
        ext = filename.rsplit('.', 1)[1] if '.' in filename else 'csv'
        
        timestamped_filename = f"{base_name}_{timestamp}.{ext}"
        filepath = self.output_dir / timestamped_filename
        
        df.to_csv(filepath, index=False)
        print(f"\nResults saved to: {filepath}")
        return filepath
    
    def generate_report(self, stability_df):
        """Generate comprehensive human-readable report."""
        print("\n" + "="*80)
        print(" "*25 + "PARAMETER STABILITY REPORT")
        print("="*80)
        
        for param in stability_df['parameter'].unique():
            param_data = stability_df[stability_df['parameter'] == param].head(10)  # Top 10
            
            print(f"\n{'='*80}")
            print(f"{param.upper()}")
            print(f"{'='*80}")
            
            # Summary table
            print(f"\n{'Value':<8} {'Return':<15} {'Sharpe':<15} {'Sortino':<15} {'Calmar':<15}")
            print(f"{'':8} {'Mean±Std':<15} {'Mean±Std':<15} {'Mean±Std':<15} {'Mean±Std':<15}")
            print("-" * 80)
            
            for _, row in param_data.iterrows():
                print(f"{row['value']:<8.2f} "
                      f"{row['return_mean']:>6.1%}±{row['return_vol']:>5.1%}  "
                      f"{row['sharpe_mean']:>6.2f}±{row['sharpe_std']:>5.2f}  "
                      f"{row['sortino_mean']:>6.2f}±{row['sortino_std']:>5.2f}  "
                      f"{row['calmar_mean']:>6.2f}±{row['calmar_std']:>5.2f}")
            
            # Drawdown and consistency table
            print(f"\n{'Value':<8} {'Max DD':<20} {'Positive %':<12} {'Range':<30}")
            print(f"{'':8} {'Mean [Min, Max]':<20} {'':<12} {'Sharpe [Min, Max]':<30}")
            print("-" * 80)
            
            for _, row in param_data.iterrows():
                print(f"{row['value']:<8.2f} "
                      f"{row['dd_mean']:>6.1%} [{row['dd_min']:>6.1%}, {row['dd_max']:>6.1%}]  "
                      f"{row['positive_pct']:>10.0f}%  "
                      f"[{row['sharpe_min']:>5.2f}, {row['sharpe_max']:>5.2f}]")
            
            # Recommendation with reasoning
            best = param_data.iloc[0]
            print(f"\n{'RECOMMENDED:':<20} {best['value']}")
            print(f"{'':20} Sharpe: {best['sharpe_mean']:.2f} (±{best['sharpe_std']:.2f})")
            print(f"{'':20} Consistency: {best['positive_pct']:.0f}% positive periods")
            print(f"{'':20} Max DD: {best['dd_mean']:.1%} avg, worst {best['dd_min']:.1%}")
            
            # Statistical significance note
            if len(param_data) > 1:
                second_best = param_data.iloc[1]
                sharpe_diff = best['sharpe_mean'] - second_best['sharpe_mean']
                std_pooled = np.sqrt((best['sharpe_std']**2 + second_best['sharpe_std']**2) / 2)
                
                if sharpe_diff > std_pooled:
                    print(f"{'':20} Margin over 2nd place ({second_best['value']}) "
                          f"exceeds pooled std deviation")
                else:
                    print(f"{'':20} Difference from 2nd place ({second_best['value']}) "
                          f"not statistically significant")
        
        print("\n" + "="*80)
        print("INTERPRETATION GUIDE:")
        print("  • Mean±Std: Average ± standard deviation across test periods")
        print("  • Positive %: Percent of periods with Sharpe > 0")
        print("  • [Min, Max]: Range across all test periods")
        print("  • Choose params with: high mean, low std, high positive %")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Parameter Analysis Framework')
    
    parser.add_argument('--sweep', type=str, choices=['window', 'threshold', 'vol', 'position', 'drawdown'],
                       help='Sweep a single parameter')
    parser.add_argument('--full-grid', action='store_true',
                       help='Run full grid sweep (slow!)')
    parser.add_argument('--walk-forward', type=str,
                       help='Walk-forward analysis for parameter (e.g., "window")')
    
    args = parser.parse_args()
    
    # Default universe
    symbols = ['AAPL', 'GOOG', 'MSFT', 'OXY', 'XOM', 'TLT', 'VOO']
    
    study = ParameterStudy(symbols)
    
    # Baseline config
    baseline = {
        'window': 60,
        'min_prob_threshold': 0.67,
        'target_vol': 0.15,
        'max_position': 0.20,
        'initial_capital': 100000.0,
        'drawdown_threshold': 0.15,
        'drawdown_scaling': 0.75,
    }
    
    if args.sweep:
        # Single parameter sweep on full period
        param_map = {
            'window': 'window',
            'threshold': 'min_prob_threshold',
            'vol': 'target_vol',
            'position': 'max_position',
            'drawdown': 'drawdown_threshold',
        }
        param_name = param_map[args.sweep]
        
        log_returns = study.load_data('2020-01-01', '2024-12-31')
        results = study.sweep_single_parameter(param_name, baseline, log_returns, '2020-2024')
        
        study.save_results(results, f'sweep_{args.sweep}.csv')
        
        print("\n" + "="*70)
        print(results[['value', 'sharpe', 'calmar', 'max_dd']].to_string(index=False))
    
    elif args.full_grid:
        # Full grid sweep
        log_returns = study.load_data('2020-01-01', '2026-02-20')
        results = study.full_grid_sweep(log_returns, '2020-2024', max_combinations=200)
        
        study.save_results(results, 'full_grid_sweep.csv')
        
        # Show top 10
        print("\nTop 10 configurations by Sharpe:")
        top10 = results.nlargest(10, 'sharpe')[[
            'window', 
            'min_prob_threshold', 
            'target_vol', 
            'max_position',
            'drawdown_threshold',
            'total_return', 
            'sharpe', 
            'calmar'
        ]]
        top10['total_return'] = (top10['total_return'] * 100).round(1).astype(str) + '%' # Format as percentage
        print(top10.to_string(index=False))
    
    elif args.walk_forward:
        # Walk-forward analysis
        param_map = {
            'window': 'window',
            'threshold': 'min_prob_threshold',
            'vol': 'target_vol',
            'position': 'max_position',
            'drawdown': 'drawdown_threshold',
        }
        param_name = param_map[args.walk_forward]
        
        results = study.walk_forward_analysis(param_name, baseline)
        study.save_results(results, f'walk_forward_{args.walk_forward}.csv')
        
        # Stability analysis
        stability = study.analyze_stability(results)
        study.save_results(stability, f'stability_{args.walk_forward}.csv')
        
        study.generate_report(stability)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
