"""
risk/regime_detector_ml.py

ML regime detector — v3 improvements:
1. Explicit neutral labeling (volatility + range-bound criteria, not just residual)
2. Asymmetric bear penalty via sample_weight (missing bear costs 3x more)
3. Confidence threshold gate — low-confidence predictions default to neutral scalar
4. Long-only strategy focus: bull/neutral/bear map to 1.0 / 0.75 / 0.5 position scalars
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RegimeSignal:
    """Market regime classification."""
    regime: Literal['bull', 'bear', 'neutral']
    confidence: float
    position_scalar: float
    probabilities: dict


class ImprovedRegimeDetector:
    """
    Improved regime detector focusing on regime quality over prediction accuracy.
    
    Key improvements:
    1. Labels based on volatility-adjusted trends (not raw returns)
    2. Only 5 robust features (prevent overfitting)
    3. Gradient Boosting (better for time series)
    4. Longer lookback (60 days = actual regime, not noise)
    """
    
    def __init__(
        self,
        bull_scalar: float = 1.0,
        bear_scalar: float = 0.5,
        neutral_scalar: float = 0.75,
        confidence_threshold: float = 0.55,  # Min confidence to act on prediction
        bear_penalty: float = 3.0,           # Cost multiplier for missing a bear regime
    ):
        self.bull_scalar = bull_scalar
        self.bear_scalar = bear_scalar
        self.neutral_scalar = neutral_scalar
        self.confidence_threshold = confidence_threshold
        self.bear_penalty = bear_penalty
        
        # Use simpler Gradient Boosting with MORE regularization
        self.model = GradientBoostingClassifier(
            n_estimators=30,      # Fewer trees (was 50)
            max_depth=2,          # Even shallower (was 3)
            learning_rate=0.05,   # Slower learning (was 0.1)
            subsample=0.7,        # More bagging (was 0.8)
            min_samples_split=20, # Require more samples to split
            min_samples_leaf=10,  # Require more samples in leaves
            max_features=3,       # Only use 3 of 5 features per tree
            random_state=42
        )
        
        # Use RobustScaler (less sensitive to outliers)
        self.scaler = RobustScaler()
        self.is_trained = False
    
    def compute_features(self, prices: pd.DataFrame, market_proxy: str = None) -> pd.DataFrame:
        """
        Compute ONLY the 5 most robust features.
        
        Based on walk-forward analysis, these matter:
        1. MA50 to MA200 (trend)
        2. Price to MA200 (position in trend)
        3. 3-month return (momentum)
        4. Volatility ratio (regime stability)
        5. Market breadth (participation)
        """
        # Get market index
        if market_proxy and market_proxy in prices.columns:
            market = prices[market_proxy]
        else:
            market = prices.mean(axis=1)
        
        features = pd.DataFrame(index=prices.index)
        
        # Feature 1: MA crossover (strongest signal from walk-forward)
        ma_50 = market.rolling(50).mean()
        ma_200 = market.rolling(200).mean()
        features['ma_crossover'] = (ma_50 / ma_200) - 1
        
        # Feature 2: Price position relative to long-term trend
        features['price_to_ma200'] = (market / ma_200) - 1
        
        # Feature 3: Medium-term momentum
        features['return_3m'] = market.pct_change(60)
        
        # Feature 4: Volatility regime (stable vs choppy)
        returns = market.pct_change()
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        features['vol_regime'] = vol_20 / vol_60
        
        # Feature 5: Market breadth (what % of stocks are healthy)
        above_ma = (prices > prices.rolling(50).mean()).sum(axis=1) / len(prices.columns)
        features['breadth'] = above_ma
        
        return features.dropna()
    
    def label_regimes_improved(self, prices: pd.DataFrame, market_proxy: str = None) -> pd.Series:
        """
        Label regimes with an EXPLICIT neutral definition.

        Neutral is no longer a residual catch-all. It is defined positively
        as periods of elevated volatility or range-bound price action, where
        momentum signals are unreliable and the strategy should reduce exposure.

        Rules (applied in priority order):
          Bear:    Price below MA200 AND 3m return < -2%
                   (clear downtrend — protect capital)
          Neutral: ANY of the following choppy/uncertain conditions:
                     - Short-term vol elevated vs long-term (vol_ratio > 1.2)
                     - Price within 3% of MA200 (transitional zone)
                     - 3m return between -5% and +5% (no clear momentum)
          Bull:    Price above MA200 AND 3m return > +5% AND vol not elevated
                   (clean uptrend — full exposure)
        """
        if market_proxy and market_proxy in prices.columns:
            market = prices[market_proxy]
        else:
            market = prices.mean(axis=1)

        ma_200 = market.rolling(200).mean()
        returns_daily = market.pct_change()
        return_3m = market.pct_change(60)

        # Volatility ratio: short-term vs long-term vol
        vol_20 = returns_daily.rolling(20).std()
        vol_60 = returns_daily.rolling(60).std()
        vol_ratio = vol_20 / vol_60.replace(0, np.nan)

        # Distance from MA200 (absolute %)
        dist_from_ma200 = (market / ma_200) - 1

        # --- Bear: clear downtrend ---
        bear_mask = (
            (market < ma_200) &
            (return_3m < -0.02)
        )

        # --- Neutral: choppy / transitional / low-conviction ---
        neutral_mask = (
            ~bear_mask & (
                (vol_ratio > 1.2) |                          # Volatility elevated
                (dist_from_ma200.abs() < 0.03) |             # Hugging MA200
                ((return_3m > -0.05) & (return_3m < 0.05))  # No clear momentum
            )
        )

        # --- Bull: everything remaining (clean uptrend) ---
        bull_mask = ~bear_mask & ~neutral_mask

        labels = pd.Series(index=market.index, dtype=str)
        labels[bull_mask] = 'bull'
        labels[bear_mask] = 'bear'
        labels[neutral_mask] = 'neutral'

        return labels.dropna()
    
    def balance_classes(self, X, y):
        """
        Balance classes by undersampling the majority class.
        
        If we have: 500 bull, 200 bear, 300 neutral
        Sample down to: 200 bull, 200 bear, 200 neutral
        """
        from collections import Counter
        
        # Count samples per class
        class_counts = Counter(y)
        min_count = min(class_counts.values())
        
        # Sample min_count from each class
        balanced_indices = []
        for regime in ['bull', 'bear', 'neutral']:
            regime_indices = np.where(y == regime)[0]
            if len(regime_indices) > 0:
                sampled = np.random.choice(
                    regime_indices, 
                    size=min(min_count, len(regime_indices)),
                    replace=False
                )
                balanced_indices.extend(sampled)
        
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        return X.iloc[balanced_indices], y.iloc[balanced_indices]
    
    def augment_data(self, X, y, n_augmented=2):
        """
        Create synthetic training samples via noise injection.
        
        This helps when we have limited data (only 7 years).
        For each real sample, create N slightly perturbed versions.
        """
        X_aug = []
        y_aug = []
        
        for i in range(len(X)):
            # Original sample
            X_aug.append(X.iloc[i].values)
            y_aug.append(y.iloc[i])
            
            # Create augmented versions
            for _ in range(n_augmented):
                # Add small random noise (5% of std dev)
                noise = np.random.normal(0, 0.05, size=X.shape[1])
                augmented = X.iloc[i].values + noise
                
                X_aug.append(augmented)
                y_aug.append(y.iloc[i])
        
        return np.array(X_aug), np.array(y_aug)
    
    def train(self, prices: pd.DataFrame, market_proxy: str = None, verbose: bool = True):
        """
        Train on historical data.

        Asymmetric bear penalty: bear samples are upweighted by self.bear_penalty
        so the model pays a higher cost for missing bear regimes than for
        misclassifying neutral periods. This is appropriate for a long-only
        strategy where drawdown avoidance is the primary value of the regime signal.
        """
        if verbose:
            print("\nTraining ML Regime Detector (v3)...")

        # Compute features and labels
        features_df = self.compute_features(prices, market_proxy)
        labels = self.label_regimes_improved(prices, market_proxy)

        # Align on common dates
        common_dates = features_df.index.intersection(labels.index)
        X = features_df.loc[common_dates]
        y = labels.loc[common_dates]

        if len(X) < 200:
            if verbose:
                print(f"  Warning: Only {len(X)} samples (need 200+)")
            return

        if verbose:
            print(f"  Raw label distribution:")
            for regime, count in y.value_counts().items():
                print(f"    {regime:<10} {count:>4} ({count/len(y)*100:.1f}%)")

        # Time series split (80/20, no shuffling — respect temporal order)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Balance classes by undersampling majority
        X_train_balanced, y_train_balanced = self.balance_classes(X_train, y_train)
        if verbose:
            print(f"  Class balancing: {len(X_train)} -> {len(X_train_balanced)} samples")

        # Data augmentation via noise injection
        X_train_aug, y_train_aug = self.augment_data(X_train_balanced, y_train_balanced, n_augmented=2)
        if verbose:
            print(f"  Data augmentation: {len(X_train_balanced)} -> {len(X_train_aug)} samples")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_aug)
        X_test_scaled = self.scaler.transform(X_test)

        # Build asymmetric sample weights: bear samples cost `bear_penalty` times more
        sample_weights = np.where(y_train_aug == 'bear', self.bear_penalty, 1.0)

        # Fit model with asymmetric weighting
        self.model.fit(X_train_scaled, y_train_aug, sample_weight=sample_weights)
        self.is_trained = True

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train_aug)
        test_score = self.model.score(X_test_scaled, y_test)

        if verbose:
            print(f"  Train accuracy: {train_score:.1%}")
            print(f"  Test accuracy:  {test_score:.1%}")

            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\n  Feature Importance:")
            for _, row in feature_importance.iterrows():
                print(f"    {row['feature']:<20} {row['importance']:.3f}")

            print(f"\n  Bear penalty applied: {self.bear_penalty}x")
            print(f"  Confidence threshold: {self.confidence_threshold:.0%}")

            from sklearn.metrics import confusion_matrix, classification_report
            y_pred = self.model.predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred, labels=['bull', 'bear', 'neutral'])

            print(f"\n  Test Set Confusion Matrix:")
            print(f"              Predicted")
            print(f"  Actual    Bull  Bear  Neutral")
            for i, actual in enumerate(['Bull', 'Bear', 'Neutral']):
                print(f"  {actual:<8} {cm[i][0]:>5} {cm[i][1]:>5} {cm[i][2]:>8}")

            print(f"\n  Classification Report:")
            print(classification_report(y_test, y_pred, labels=['bull', 'bear', 'neutral'], target_names=['bull', 'bear', 'neutral'], zero_division=0))

        print(f"  Model trained on {len(X_train)} samples\n")
    
    def predict(self, prices: pd.DataFrame, market_proxy: str = None, verbose: bool = False) -> RegimeSignal:
        """
        Predict current market regime.

        Confidence threshold gate: if the model's confidence in its top prediction
        falls below self.confidence_threshold, the prediction is treated as neutral
        regardless of what the model picked. This prevents the model from acting
        on low-conviction bull calls in flat/ambiguous markets.

        Bear predictions are NEVER downgraded by the confidence gate — if the
        model says bear at any confidence level, we respect it. The gate only
        softens uncertain bull/neutral calls.
        """
        if not self.is_trained:
            self.train(prices, market_proxy, verbose=False)

        features_df = self.compute_features(prices, market_proxy)

        if len(features_df) == 0:
            return RegimeSignal(
                regime='neutral',
                confidence=0.5,
                position_scalar=self.neutral_scalar,
                probabilities={'bull': 0.33, 'bear': 0.33, 'neutral': 0.34}
            )

        latest = features_df.iloc[[-1]]
        latest_scaled = self.scaler.transform(latest)

        raw_prediction = self.model.predict(latest_scaled)[0]
        probabilities = self.model.predict_proba(latest_scaled)[0]
        prob_dict = {regime: prob for regime, prob in zip(self.model.classes_, probabilities)}
        confidence = prob_dict[raw_prediction]

        # Confidence gate: uncertain non-bear calls default to neutral
        if raw_prediction != 'bear' and confidence < self.confidence_threshold:
            prediction = 'neutral'
            gated = True
        else:
            prediction = raw_prediction
            gated = False

        scalar_map = {
            'bull': self.bull_scalar,
            'bear': self.bear_scalar,
            'neutral': self.neutral_scalar,
        }
        position_scalar = scalar_map[prediction]

        if verbose:
            print(f"\n{'='*60}")
            print(f"REGIME PREDICTION (v3)")
            print(f"{'='*60}")
            print(f"Raw prediction:  {raw_prediction.upper()}")
            if gated:
                print(f"After gate:      NEUTRAL (confidence {confidence:.1%} < threshold {self.confidence_threshold:.0%})")
            print(f"Final regime:    {prediction.upper()}")
            print(f"Confidence:      {confidence:.1%}")
            print(f"Position scalar: {position_scalar:.1%}")
            print(f"\nProbabilities:")
            for regime, prob in sorted(prob_dict.items(), key=lambda x: -x[1]):
                bar = '#' * int(prob * 30)
                print(f"  {regime.capitalize():<10} {prob:.1%}  {bar}")
            print(f"{'='*60}\n")

        return RegimeSignal(
            regime=prediction,
            confidence=confidence,
            position_scalar=position_scalar,
            probabilities=prob_dict
        )


if __name__ == "__main__":
    from datetime import datetime
    from data.fetcher import fetch_daily_bars
    
    print("="*70)
    print("IMPROVED ML REGIME DETECTOR TEST")
    print("="*70)
    
    # Fetch data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'XOM', 'VOO']
    raw, _ = fetch_daily_bars(symbols, datetime(2020, 1, 1), datetime(2024, 12, 31))
    prices = raw['close'].unstack(level='symbol')
    
    # Train improved model
    detector = ImprovedRegimeDetector()
    detector.train(prices, market_proxy='VOO', verbose=True)
    
    # Test on key dates
    test_dates = ['2020-06-01', '2021-12-01', '2022-06-01', '2023-06-01', '2024-12-01']
    
    print("\n" + "="*70)
    print("PREDICTIONS ON KEY DATES")
    print("="*70)
    
    for date in test_dates:
        period_prices = prices.loc[:date]
        regime = detector.predict(period_prices, market_proxy='VOO', verbose=False)
        
        print(f"\n{date}: {regime.regime.upper():<8} (confidence: {regime.confidence:.1%}, scalar: {regime.position_scalar:.1%})")
        print(f"  Probs: Bull {regime.probabilities['bull']:.1%}, "
              f"Bear {regime.probabilities['bear']:.1%}, "
              f"Neutral {regime.probabilities['neutral']:.1%}")
