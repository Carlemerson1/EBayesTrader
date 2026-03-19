"""
analysis/test_regime_permutation.py

Permutation test for the ML regime detector.

Tests whether the walk-forward accuracy is genuinely due to predictive signal
or could plausibly arise by chance.

Null hypothesis H0: The regime labels have no real predictive structure —
    a model trained on randomly shuffled labels would perform just as well.

Method:
    1. Run the real walk-forward test, record observed accuracy.
    2. Repeat N times:
         - Shuffle the regime labels (preserving class frequencies per year)
         - Retrain and re-evaluate walk-forward accuracy on shuffled labels
    3. p-value = fraction of permutations that match or exceed observed accuracy.
    4. If p < 0.05, the model has statistically significant predictive signal.

Usage:
    python analysis/test_regime_permutation.py
    python analysis/test_regime_permutation.py --n-perms 500
    python analysis/test_regime_permutation.py --n-perms 200 --train-years 3
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from data.fetcher import fetch_daily_bars
from risk.regime_detector_ml import ImprovedRegimeDetector


# ── helpers ───────────────────────────────────────────────────────────────────

def run_single_walkforward(prices, label_override: dict = None,
                           train_years: int = 3) -> float:
    """
    Run one pass of walk-forward validation and return mean accuracy.

    Args:
        prices:         Full price DataFrame (symbols x dates).
        label_override: If provided, a dict {date -> regime_label} that
                        replaces the model's own labeling. Used for
                        permutation runs where we inject shuffled labels.
        train_years:    Number of years to use for each training window.

    Returns:
        Mean monthly accuracy across all test years.
    """
    years = sorted(prices.index.year.unique())
    all_correct = 0
    all_total = 0

    for i in range(len(years) - train_years):
        train_end_year = years[i + train_years - 1]
        test_year_val  = years[i + train_years]

        train_data = prices[prices.index.year <= train_end_year]
        test_data  = prices[prices.index.year == test_year_val]

        detector = ImprovedRegimeDetector()
        detector.train(train_data, market_proxy='VOO', verbose=False)

        for month in range(1, 13):
            test_month = test_data[
                (test_data.index.year == test_year_val) &
                (test_data.index.month == month)
            ]
            if len(test_month) == 0:
                continue

            end_date = test_month.index[-1]
            data_up_to = prices[prices.index <= end_date]

            # Prediction always uses real features
            prediction = detector.predict(
                data_up_to, market_proxy='VOO', verbose=False
            ).regime

            # Actual label: use override (shuffled) if provided, else real
            if label_override is not None:
                actual = label_override.get(end_date)
                if actual is None:
                    continue
            else:
                real_labels = detector.label_regimes_improved(
                    data_up_to, market_proxy='VOO'
                )
                if end_date not in real_labels.index:
                    continue
                actual = real_labels.loc[end_date]

            all_correct += int(prediction == actual)
            all_total   += 1

    return all_correct / all_total if all_total > 0 else 0.0


def build_real_labels(prices, train_years: int = 3) -> dict:
    """
    Compute the real monthly end-of-month labels used in walk-forward,
    stored as {date: label}. Used as the base for shuffling.
    """
    years      = sorted(prices.index.year.unique())
    label_map  = {}
    detector   = ImprovedRegimeDetector()   # just for labeling method

    for i in range(len(years) - train_years):
        test_year_val = years[i + train_years]
        test_data     = prices[prices.index.year == test_year_val]

        for month in range(1, 13):
            test_month = test_data[
                (test_data.index.year == test_year_val) &
                (test_data.index.month == month)
            ]
            if len(test_month) == 0:
                continue

            end_date    = test_month.index[-1]
            data_up_to  = prices[prices.index <= end_date]
            real_labels = detector.label_regimes_improved(
                data_up_to, market_proxy='VOO'
            )
            if end_date in real_labels.index:
                label_map[end_date] = real_labels.loc[end_date]

    return label_map


def shuffle_labels(label_map: dict, seed: int = None) -> dict:
    """
    Shuffle labels while preserving class frequencies.
    Shuffling is done within each calendar year to preserve
    rough temporal structure (so we don't give the null model
    an artificially easy task by mixing bear-heavy years with bull-heavy years).
    """
    rng    = np.random.default_rng(seed)
    result = {}

    # Group by year
    by_year: dict[int, list] = {}
    for date, label in label_map.items():
        yr = date.year
        by_year.setdefault(yr, []).append((date, label))

    for yr, entries in by_year.items():
        dates  = [e[0] for e in entries]
        labels = [e[1] for e in entries]
        shuffled = rng.permutation(labels)
        for date, label in zip(dates, shuffled):
            result[date] = label

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def run_permutation_test(prices, n_perms: int = 200, train_years: int = 3,
                         random_seed: int = 42, output_dir: str = None) -> None:

    print("\n" + "=" * 80)
    print("REGIME DETECTOR — PERMUTATION TEST")
    print("=" * 80)
    print(f"Permutations : {n_perms}")
    print(f"Train window : {train_years} years")
    print(f"Null H0      : Shuffled labels perform as well as real labels")
    print("=" * 80)

    # ── Step 1: observed accuracy ─────────────────────────────────────────────
    print("\nRunning walk-forward on REAL labels...")
    observed_accuracy = run_single_walkforward(prices, label_override=None,
                                               train_years=train_years)
    print(f"Observed accuracy: {observed_accuracy:.1%}")

    # ── Step 2: build the label map once (expensive — reused across perms) ───
    print("\nBuilding label map for permutation runs...")
    real_label_map = build_real_labels(prices, train_years=train_years)
    n_samples = len(real_label_map)
    print(f"Label map size: {n_samples} monthly observations")

    label_counts = pd.Series(real_label_map.values()).value_counts()
    print("Label distribution:")
    for regime, count in label_counts.items():
        print(f"  {regime:<10} {count:>3} ({count/n_samples*100:.1f}%)")

    # ── Step 3: permutation runs ──────────────────────────────────────────────
    print(f"\nRunning {n_perms} permutations (this will take a few minutes)...")
    null_accuracies = []

    rng = np.random.default_rng(random_seed)

    for perm_idx in range(n_perms):
        shuffled = shuffle_labels(real_label_map, seed=int(rng.integers(0, 2**31)))
        acc = run_single_walkforward(prices, label_override=shuffled,
                                     train_years=train_years)
        null_accuracies.append(acc)

        # Progress
        if (perm_idx + 1) % 25 == 0:
            pct_done = (perm_idx + 1) / n_perms * 100
            null_so_far = np.mean(null_accuracies)
            print(f"  [{perm_idx+1:>4}/{n_perms}]  {pct_done:.0f}% done  |  "
                  f"null mean so far: {null_so_far:.1%}") # null mean is the accuracy of the model trained on shuffled labels

    null_accuracies = np.array(null_accuracies)

    # ── Step 4: results ───────────────────────────────────────────────────────
    p_value   = np.mean(null_accuracies >= observed_accuracy)
    null_mean = null_accuracies.mean()
    null_std  = null_accuracies.std()
    null_p95  = np.percentile(null_accuracies, 95)
    null_p99  = np.percentile(null_accuracies, 99)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nObserved accuracy  : {observed_accuracy:.1%}")
    print(f"\nNull distribution  :")
    print(f"  Mean             : {null_mean:.1%}")
    print(f"  Std dev          : {null_std:.1%}")
    print(f"  95th percentile  : {null_p95:.1%}")
    print(f"  99th percentile  : {null_p99:.1%}")
    print(f"  Min / Max        : {null_accuracies.min():.1%} / {null_accuracies.max():.1%}")
    print(f"\np-value            : {p_value:.4f}  "
          f"({int(p_value * n_perms)}/{n_perms} permutations >= observed)")

    print("\n" + "-" * 80)
    if p_value < 0.01:
        verdict = "HIGHLY SIGNIFICANT (p < 0.01) — strong evidence of genuine signal"
    elif p_value < 0.05:
        verdict = "SIGNIFICANT (p < 0.05) — model has real predictive power"
    elif p_value < 0.10:
        verdict = "MARGINAL (p < 0.10) — weak signal, interpret cautiously"
    else:
        verdict = f"NOT SIGNIFICANT (p = {p_value:.2f}) — cannot reject null hypothesis"

    print(f"Verdict: {verdict}")
    print("-" * 80)

    # ── Step 5: effect size ───────────────────────────────────────────────────
    effect_size = (observed_accuracy - null_mean) / null_std if null_std > 0 else 0
    print(f"\nEffect size (z-score): {effect_size:.2f}")
    print(f"  (observed is {effect_size:.1f} standard deviations above the null mean)")
    print(f"  > 2.0 = large effect,  1.0–2.0 = medium,  < 1.0 = small")

    # ── Step 6: ASCII histogram of null distribution ──────────────────────────
    print("\nNull distribution histogram:")
    bins  = np.linspace(null_accuracies.min(), null_accuracies.max(), 15)
    hist, edges = np.histogram(null_accuracies, bins=bins)
    max_bar = max(hist)

    for count, left, right in zip(hist, edges[:-1], edges[1:]):
        bar    = "█" * int(count / max_bar * 30)
        marker = " ← observed" if left <= observed_accuracy < right else ""
        print(f"  {left:.1%}–{right:.1%}  {bar}{marker}")

    if observed_accuracy >= edges[-1]:
        print(f"  > {edges[-1]:.1%}       ← observed (off chart — very strong signal)")

    print("=" * 80)

    # ── Step 7: CSV / JSON output ─────────────────────────────────────────────
    import json

    out_dir = Path(output_dir) if output_dir else Path("analysis/results/regime_perm")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp and base filename — matches scheme: regime_perm_n{N}_acc{acc}_{timestamp}
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    acc_str     = f"{observed_accuracy:.4f}".replace("0.", "").replace(".", "")[:4]
    stem        = f"regime_perm_n{n_perms}_acc{acc_str}_{timestamp}"

    # File 1: null distribution — one row per permutation, for histogram/density plot
    null_df = pd.DataFrame({
        'permutation':    range(1, n_perms + 1),
        'null_accuracy':  null_accuracies,
        'beats_observed': (null_accuracies >= observed_accuracy).astype(int),
    })
    null_path = out_dir / f"{stem}_null_dist.csv"
    null_df.to_csv(null_path, index=False)

    # File 2: per-percentile summary — useful for plotting confidence bands
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    summary_df = pd.DataFrame({
        'percentile': percentiles,
        'null_accuracy': [np.percentile(null_accuracies, p) for p in percentiles],
    })
    summary_path = out_dir / f"{stem}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # File 3: full results JSON — mirrors permtest summary.json structure
    results_json = {
        'test_type':          'regime_permutation',
        'timestamp':           timestamp,
        'n_perms':             n_perms,
        'train_years':         train_years,
        'observed_accuracy':   round(observed_accuracy, 6),
        'null_mean':           round(float(null_mean), 6),
        'null_std':            round(float(null_std), 6),
        'null_min':            round(float(null_accuracies.min()), 6),
        'null_max':            round(float(null_accuracies.max()), 6),
        'null_p05':            round(float(np.percentile(null_accuracies, 5)), 6),
        'null_p25':            round(float(np.percentile(null_accuracies, 25)), 6),
        'null_p50':            round(float(np.percentile(null_accuracies, 50)), 6),
        'null_p75':            round(float(np.percentile(null_accuracies, 75)), 6),
        'null_p95':            round(float(null_p95), 6),
        'null_p99':            round(float(null_p99), 6),
        'p_value':             round(float(p_value), 6),
        'effect_size_z':       round(float(effect_size), 4),
        'n_beats_observed':    int((null_accuracies >= observed_accuracy).sum()),
        'verdict':             verdict,
    }
    json_path = out_dir / f"{stem}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nOutput written to: {out_dir}/")
    print(f"  {stem}_null_dist.csv  — {n_perms} rows (permutation index, accuracy, beats_observed)")
    print(f"  {stem}_summary.csv   — percentile table for confidence band plots")
    print(f"  {stem}_summary.json  — full results dict")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Permutation test for regime detector")
    parser.add_argument("--n-perms",     type=int, default=200,
                        help="Number of permutations (default: 200)")
    parser.add_argument("--train-years", type=int, default=3,
                        help="Training window in years (default: 3)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir",  type=str, default=None,
                        help="Directory for CSV output (default: analysis/results/regime_perm)")
    args = parser.parse_args()

    print("\nFetching historical data...")
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'XOM', 'VOO']
    raw, _  = fetch_daily_bars(symbols, datetime(2017, 1, 1), datetime(2026, 3, 1))
    prices  = raw['close'].unstack(level='symbol')

    run_permutation_test(
        prices,
        n_perms=args.n_perms,
        train_years=args.train_years,
        random_seed=args.seed,
        output_dir=args.output_dir,
    )