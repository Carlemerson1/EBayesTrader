"""
config/settings.py

Central configuration for the trading strategy.

All hyperparms, universe defs, and system settings live here.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class StrategyConfig:
    """
    Core strategy paramaters.
    """
    # Universe
    symbols: List[str] = None                         # List of stock symbols to include in the universe (e.g. ['AAPL', 'MSFT', 'GOOG'])

    # Model Parameters
    window: int = 60                                 # Rolling window size for estimating priors and posteriors (in trading days)
    min_prob_threshold: float = 0.67                 # Minimum posterior probability P(mu > 0 | data) threshold to take a position (e.g. 0.67 = 67% confidence)

    # Risk Parameters
    target_vol: float = 0.20                         # Target annualized portfolio volatility
    max_position: float = 0.20                       # Maximum position size as a fraction of portfolio (e.g. 0.20 = 20% max allocation to any single stock)

    # Drawdown Protection
    drawdown_threshold: float = 0.15                 # Trigger at 15% drawdown
    drawdown_recovery: float = 0.13                  # Recover at 13% drawdown (i.e. if we cut risk at -15%, we can restore full risk once drawdown improves to -13%)
    drawdown_scaling: float = 0.75                   # Scale exposure to 75% when active

    # Execution
    rebalance_frequency: int = 5                     # Rebalance every N trading days
    stop_loss_threshold: float = -0.01               # Close position if loses >1% in a day

    # Capital
    initial_capital: float = 100000.0               # Starting capital for backtesting

    def __post_init__(self):
        """Set default universe if not provided"""
        if self.symbols is None:
            self.symbols = [
                "AAPL",     # Tech
                "MSFT",     # Tech
                "GOOG",     # Tech
                "OXY",      # Energy
                "XOM",      # Energy
                "VOO",      # ETF
                "TLT",      # Bond
            ]

@dataclass
class AlpacaConfig:
    """
    Alpaca API configuration.
    
    These should be set via environment variables in .env file for security.
        ALPACA_API_KEY: Your Alpaca API key
        ALPACA_SECRET_KEY: Your Alpaca secret key
        ALPACA_BASE_URL: //paper-api.alpaca.markets (for paper trading) or https://api.alpaca.markets (for live trading)
    """

    api_key: str = None
    secret_key: str = None
    base_url: str = "https://paper-api.alpaca.markets"  # Default to paper trading endpoint

    def __post_init__(self):
        """Load from environment if not provided."""
        import os
        from dotenv import load_dotenv

        load_dotenv()

        if self.api_key is None:
            self.api_key = os.getenv('ALPACA_API_KEY')
        if self.secret_key is None:
            self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        if os.getenv('ALPACA_BASE_URL'):
            self.base_url = os.getenv('ALPACA_BASE_URL')

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca credentials not found. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in your .env file."
            )
        
# ---------------------------------------------------------
# Production configurations
# ---------------------------------------------------------

# Default configuration for the strategy. Can be overridden by passing a custom config to the backtest or live trading functions.
DEFAULT_CONFIG = StrategyConfig()

# Conservative configuration (lower vol, tighter stops)
CONSERVATIVE_CONFIG = StrategyConfig(
    target_vol=0.12,
    min_prob_threshold=0.72,
    stop_loss_threshold=-0.008
)

# Aggressive configuration (higher vol, looser stops)
AGGRESSIVE_GROWTH_CONFIG = StrategyConfig(
    symbols=[
        # Tech Growth
        'NVDA', 'AMD', 'AVGO', 'PLTR', 'CRWD', 'SNOW',
        # Mega-cap Tech
        'AAPL', 'GOOGL', 'MSFT', 'META',
        # Energy
        'XOM', 'CVX', 'COP', 'HL',
        # Consumer
        'AMZN',
        # ETFs
        'VOO', 'QQQ',
        # Bonds
        'TLT', 'IEF',
    ],
    window=60,                    # These are defaults - will be optimized
    min_prob_threshold=0.67,
    target_vol=0.18,              # Bumped up for aggressive
    max_position=0.20,
    initial_capital=100000.0,
    drawdown_threshold=0.15,
    drawdown_scaling=0.75,
    drawdown_recovery=0.13,
    rebalance_frequency=5,
)

# ---------------------------------------------------------
# This block only runs when you execute file directly:
# python3 config/settings.py
# Does NOT run when another file imports from this module
# ---------------------------------------------------------

if __name__ == "__main__":
    print("=== Strategy Configuration ===\n")

    config = DEFAULT_CONFIG

    print(f"Universe: {', '.join(config.symbols)}")
    print(f"Rolling Window: {config.window} days")
    print(f"Min Prob Threshold: {config.min_prob_threshold:.0%}")
    print(f"Target Volatility: {config.target_vol:.0%}")
    print(f"Max Position: {config.max_position:.0%}")
    print(f"Rebalance Frequency: Every {config.rebalance_frequency} days")
    print(f"Stop-Loss Threshold: {config.stop_loss_threshold:.1%}")
    print(f"Drawdown Protection: Trigger at {config.drawdown_threshold:.0%}, "
          f"recover at {config.drawdown_recovery:.0%}, "
          f"scaling factor {config.drawdown_scaling:.0%}")
    
    print("\n=== Alpaca Configuration ===\n")

    try:
        alpaca_config = AlpacaConfig()
        print(f"Base URL: {alpaca_config.base_url}")
        print(f"API Key: {alpaca_config.api_key[:8]}..." if alpaca_config.api_key else "API Key: Not Set")
        print("Credentials loaded successfully.")
    except ValueError as e:
        print(str(e))

