"""
Test regime detector with walk-forward validation.

This shows if the model can predict future regimes without seeing them.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from data.fetcher import fetch_daily_bars
from risk.regime_detector_ml import ImprovedRegimeDetector


def walk_forward_test(prices, train_years=3, test_year=1):
    """
    Walk-forward validation of regime detector.
    
    Args:
        prices: Price data
        train_years: Years to use for training
        test_year: Year to test on
        
    Returns:
        DataFrame with results for each test period
    """
    results = []
    
    # Get yearly periods
    years = sorted(prices.index.year.unique())
    
    print("\n" + "="*80)
    print("WALK-FORWARD REGIME DETECTOR VALIDATION")
    print("="*80)
    print(f"Available years: {years[0]}-{years[-1]}")
    print(f"Strategy: Train on {train_years} years, test on next year\n")
    
    for i in range(len(years) - train_years - test_year + 1):
        train_end_year = years[i + train_years - 1]
        test_year_val = years[i + train_years]
        
        # Split data
        train_data = prices[prices.index.year <= train_end_year]
        test_data = prices[prices.index.year == test_year_val]
        
        print(f"\n{'='*80}")
        print(f"Test Period: {test_year_val}")
        print(f"Train: {years[i]}-{train_end_year} ({len(train_data)} days)")
        print(f"Test:  {test_year_val} ({len(test_data)} days)")
        print(f"{'='*80}")
        
        # Train model
        detector = ImprovedRegimeDetector()
        detector.train(train_data, market_proxy='VOO', verbose=False)
        
        # Test on each month of test year
        monthly_predictions = []
        monthly_actuals = []
        
        for month in range(1, 13):
            test_month = test_data[
                (test_data.index.year == test_year_val) & 
                (test_data.index.month == month)
            ]
            
            if len(test_month) == 0:
                continue
            
            # Get data up to this month
            data_up_to_month = prices[prices.index <= test_month.index[-1]]
            
            # Predict
            regime_signal = detector.predict(data_up_to_month, market_proxy='VOO', verbose=False)
            prediction = regime_signal.regime
            
            # Get actual label using FULL history up to this month
            # (MA200 needs ~200 days of prior data — can't compute on isolated test year)
            actual_labels = detector.label_regimes_improved(data_up_to_month, market_proxy='VOO')
            if test_month.index[-1] in actual_labels.index:
                actual = actual_labels.loc[test_month.index[-1]]
            else:
                continue
            
            monthly_predictions.append(prediction)
            monthly_actuals.append(actual)
            
            correct = "✓" if prediction == actual else "✗"
            print(f"  {test_year_val}-{month:02d}: Predicted {prediction:<7} | Actual {actual:<7} | {correct}")
        
        # Calculate accuracy for this year
        if len(monthly_predictions) > 0:
            accuracy = sum(p == a for p, a in zip(monthly_predictions, monthly_actuals)) / len(monthly_predictions)
            
            results.append({
                'test_year': test_year_val,
                'train_period': f"{years[i]}-{train_end_year}",
                'accuracy': accuracy,
                'correct': sum(p == a for p, a in zip(monthly_predictions, monthly_actuals)),
                'total': len(monthly_predictions),
            })
            
            print(f"\n  Year {test_year_val} Accuracy: {accuracy:.1%} ({sum(p == a for p, a in zip(monthly_predictions, monthly_actuals))}/{len(monthly_predictions)} correct)")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("\nFetching historical data...")
    
    # Fetch extended history (need more years for walk-forward)
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'XOM', 'VOO']
    raw, _ = fetch_daily_bars(symbols, datetime(2017, 1, 1), datetime(2026, 3, 1))
    
    prices = raw['close'].unstack(level='symbol')
    
    # Run walk-forward test
    results = walk_forward_test(prices, train_years=3, test_year=1)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(results.to_string(index=False))
    
    print(f"\nAverage Accuracy: {results['accuracy'].mean():.1%}")
    print(f"Std Dev:          {results['accuracy'].std():.1%}")
    print(f"Best Year:        {results.loc[results['accuracy'].idxmax(), 'test_year']} ({results['accuracy'].max():.1%})")
    print(f"Worst Year:       {results.loc[results['accuracy'].idxmin(), 'test_year']} ({results['accuracy'].min():.1%})")
    
    # Interpretation
    avg_acc = results['accuracy'].mean()
    if avg_acc > 0.55:
        print(f"\nSTRONG: Model generalizes well ({avg_acc:.1%} > 55%)")
    elif avg_acc > 0.45:
        print(f"\nMODERATE: Model has some predictive power ({avg_acc:.1%})")
    else:
        print(f"\nWEAK: Model doesn't generalize ({avg_acc:.1%} < 45%)")
    
    print("="*80)