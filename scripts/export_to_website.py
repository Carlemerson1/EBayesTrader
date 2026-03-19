"""
scripts/export_to_website.py

Exports current portfolio state, equity curve, signals, and validation
stats to a single JSON file for the personal website portfolio page.

Run daily (or manually after each rebalance):
    python scripts/export_to_website.py

    # Specify website repo path if different from default:
    python scripts/export_to_website.py --website-path ~/Projects/PersonalWebsite

The script writes to:
    PersonalWebsite/static/data/trading_data.json
"""

import sys
import json
import argparse
import glob
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def load_equity_curve(log_dir: Path) -> list:
    history_file = log_dir / 'portfolio_history.csv'
    if not history_file.exists():
        return []
    df = pd.read_csv(history_file, index_col=0)
    df.index = pd.to_datetime(df.index, format='mixed')
    df = df.sort_index()
    # Daily resample — last value per day
    daily = df['portfolio_value'].resample('D').last().dropna()
    return [
        {'date': str(d.date()), 'value': round(float(v), 2)}
        for d, v in daily.items()
    ]


def load_positions(alpaca_config) -> dict:
    try:
        from execution.trader import AlpacaTrader
        trader  = AlpacaTrader(alpaca_config)
        account = trader.get_account()
        raw_pos = trader.get_positions()
        return {
            'value':     round(float(account.equity), 2),
            'cash':      round(float(account.cash), 2),
            'positions': {k: int(v) for k, v in raw_pos.items()},
        }
    except Exception as e:
        print(f"  Warning: could not fetch live positions ({e})")
        return {'value': None, 'cash': None, 'positions': {}}


def load_signals(strategy_config) -> list:
    try:
        from data.fetcher import fetch_daily_bars
        from data.processor import compute_log_returns, clean_returns
        from model.prior import estimate_prior
        from model.posterior import update_all_posteriors
        from model.signals import compute_all_signals
        from datetime import timedelta

        end   = datetime.now()
        start = end - timedelta(days=int(strategy_config.window * 1.5) + 10)
        raw, _ = fetch_daily_bars(strategy_config.symbols, start, end)

        log_returns = compute_log_returns(raw)
        log_returns = clean_returns(log_returns)
        window      = log_returns.tail(strategy_config.window)

        prior      = estimate_prior(window)
        posteriors = update_all_posteriors(window, prior)
        signals    = compute_all_signals(posteriors, strategy_config.min_prob_threshold)

        return sorted([
            {
                'symbol':          sym,
                'prob':            round(float(sig.prob_positive), 4),
                'action':          sig.action,
                'expected_return': round(float(sig.expected_return), 6),
            }
            for sym, sig in signals.items()
        ], key=lambda x: x['prob'], reverse=True)

    except Exception as e:
        print(f"  Warning: could not compute signals ({e})")
        return []


def load_validation_stats(results_dir: Path) -> dict:
    stats = {}

    # Latest strategy permutation test
    perm_files = sorted(glob.glob(str(results_dir / 'permtest' / '*_summary.json')))
    if perm_files:
        with open(perm_files[-1]) as f:
            d = json.load(f)
        stats['strategy'] = {
            'sharpe':        round(d.get('actual_sharpe', 0), 3),
            'permuted_mean': round(d.get('permuted_mean', 0), 3),
            'p_value':       round(d.get('p_value', 1), 4),
            'n_perms':       d.get('n_permutations', 0),
            'significant':   d.get('significant_at_5pct', False),
        }

    # Latest regime permutation test
    regime_files = sorted(glob.glob(str(results_dir / 'regime_perm' / '*_summary.json')))
    if regime_files:
        with open(regime_files[-1]) as f:
            d = json.load(f)
        stats['regime'] = {
            'observed_accuracy': round(d.get('observed_accuracy', 0), 4),
            'p_value':           round(d.get('p_value', 1), 4),
            'effect_size_z':     round(d.get('effect_size_z', 0), 2),
            'n_perms':           d.get('n_perms', 0),
            'verdict':           d.get('verdict', ''),
        }

    return stats


# ── main ──────────────────────────────────────────────────────────────────────

def export(website_path: Path):
    print("EBayesTrader → PersonalWebsite export")
    print("=" * 50)

    project_root = Path(__file__).parent.parent
    log_dir      = project_root / 'logs'
    results_dir  = project_root / 'analysis' / 'results'
    output_file  = website_path / 'static' / 'data' / 'trading_data.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    from config.settings import AlpacaConfig, AGGRESSIVE_GROWTH_CONFIG
    alpaca_config   = AlpacaConfig()
    strategy_config = AGGRESSIVE_GROWTH_CONFIG

    print("Loading equity curve...")
    equity_curve = load_equity_curve(log_dir)
    print(f"  {len(equity_curve)} daily data points")

    print("Loading live positions...")
    portfolio = load_positions(alpaca_config)
    print(f"  Portfolio value: ${portfolio['value']:,.2f}" if portfolio['value'] else "  (unavailable)")

    print("Computing current signals...")
    signals = load_signals(strategy_config)
    print(f"  {len(signals)} signals computed")

    print("Loading validation stats...")
    validation = load_validation_stats(results_dir)
    print(f"  Found: {list(validation.keys())}")

    payload = {
        'last_updated':  datetime.now().strftime('%Y-%m-%d %H:%M'),
        'environment':   'paper_trading',
        'portfolio':     portfolio,
        'equity_curve':  equity_curve,
        'signals':       signals,
        'validation':    validation,
        'config': {
            'universe':   strategy_config.symbols,
            'window':     strategy_config.window,
            'target_vol': strategy_config.target_vol,
        }
    }

    with open(output_file, 'w') as f:
        json.dump(payload, f, indent=2)

    size_kb = output_file.stat().st_size / 1024
    print(f"\nWritten to: {output_file}  ({size_kb:.1f} KB)")
    print("Done. Commit and push PersonalWebsite to deploy.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--website-path',
        type=str,
        default=str(Path(__file__).parent.parent.parent / 'PersonalWebsite'),
        help='Path to PersonalWebsite repo root'
    )
    args = parser.parse_args()
    export(Path(args.website_path))