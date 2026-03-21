"""
execution/trader.py

Handles order submission and position management via Alpaca API
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from model.config.settings import AlpacaConfig


@dataclass
class OrderResult:
    """Result of a single order submission"""
    symbol: str
    side: str # 'buy' or 'sell'
    qty: float # quantity ordered
    success: bool
    order_id: Optional[str] = None
    error: Optional[str] = None


class AlpacaTrader:
    """
    Manages trading execution via Alpaca.
    
    Handles:
     - Position reconciliation (current vs target)
     - Order submission (market orders)
     - Order status tracking
     - Account info queries
     """

    def __init__(self, config: AlpacaConfig):
        """Initialize trading client.
        
        Args:
            config: AlpacaConfig dataclass with API keys
        """
        self.config = config
        self.client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=True if 'paper' in config.base_url else False
        )

        is_paper = 'paper' in config.base_url.lower()
        print(f" Connected to Alpaca ({'paper' if is_paper else 'live'} trading)")

    def get_account(self):
        """Fetch account information."""
        return self.client.get_account()
    
    def get_positions(self) -> Dict[str, float]:
        """
        Gets current positions as a dict of {symbol: qty}
        
        Returns:
            Dict mapping symbol to signed quantity (positive=long, negative=short)
        """
        positions = {}
        try:
            alpaca_positions = self.client.get_all_positions()
            for pos in alpaca_positions:
                qty = float(pos.qty)
                # Alpaca uses seperate 'side' field for long/short
                if pos.side == 'short':
                    qty = -qty
                positions[pos.symbol] = qty
        except Exception as e:
            print(f"Error fetching positions: {e}")
            
        return positions
    

    def get_portfolio_value(self) -> float:
        """
        Returns current portfolio equity.
        """
        account = self.get_account()
        return float(account.equity)
    
    def compute_target_quantities(
        self,
        target_weights: Dict[str, float],
        portfolio_value: float,
        current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Converts target weights to share quantities.
        
        Args:
            target_weights: Dict {symbol: target_weight} in [-1,1]
            portfolio_value: Total portfolio value in dollars
            current_prices: Dict {symbol: price_per_share}

        Returns:
            Dict {symbol: target_shares} (rounded to integers)
        """
        target_quantities = {}
        
        for symbol, weight in target_weights.items():
            if symbol not in current_prices:
                print(f"WARNING: No price data for {symbol}, skipping")
                continue

            target_dollar = weight * portfolio_value
            price = current_prices[symbol]

            if price == 0:
                print(f"WARNING: Price for {symbol} is zero, skipping")
                continue

            target_qty = int(target_dollar / price)  # round to whole share
            target_quantities[symbol] = target_qty

        return target_quantities
    
    def submit_orders(
        self,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        dry_run: bool = False
    ) -> List[OrderResult]:
        """
        Submits market orders to move from current positions to target weights.

        Args:
            target_weights: Dict {symbol: target_weight} in [-1,1]
            current_prices: Dict {symbol: price}
            dry_run: If True, prints orders but doesnt submit to Alpaca

        Returns:
            List of OrderResult objects with order outcomes
        """

        portfolio_value = self.get_portfolio_value()
        current_positions = self.get_positions() # Dict {symbol: qty}

        target_quantities = self.compute_target_quantities(
            target_weights, 
            portfolio_value,
            current_prices
        ) # Dict {symbol: target_qty}

        #compute deltas between current and target
        orders = []
        
        all_symbols = set(current_positions.keys()) | set(target_quantities.keys())

        for symbol in all_symbols:
            current_qty = current_positions.get(symbol, 0)
            target_qty = target_quantities.get(symbol, 0)
            delta = target_qty - current_qty

            if abs(delta) < 1: # skip if less than 1 share difference
                continue

            # determine order side
            # Determine order side and handle transition from long to short
            if delta > 0:
                side = OrderSide.BUY
                qty = abs(delta)
            else:
                # Selling or shorting
                side = OrderSide.SELL
                qty = abs(delta)
                
                # SPECIAL CASE: Going from long to short requires TWO orders
                if current_qty > 0 and target_qty < 0:
                    # Split into two orders:
                    # 1. Close long position
                    # 2. Open short position
                    
                    print(f"[SPLIT ORDER] {symbol}: Closing {current_qty} long, then shorting {abs(target_qty)}")
                    
                    # Order 1: Close the long
                    try:
                        close_request = MarketOrderRequest(
                            symbol=symbol,
                            qty=current_qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY
                        )
                        close_order = self.client.submit_order(close_request)
                        print(f"  → Closed long: SELL {current_qty} (Order ID: {close_order.id})")

                        # WAIT for close order to fill before submitting short
                        import time
                        max_wait = 10  # seconds
                        waited = 0
                        while waited < max_wait:
                            order_status = self.client.get_order_by_id(close_order.id)
                            if order_status.status in ['filled', 'partially_filled']:
                                print(f"  → Close order filled")
                                break
                            time.sleep(0.5)
                            waited += 0.5
                        
                        if order_status.status not in ['filled', 'partially_filled']:
                            print(f"  → Warning: Close order still pending after {max_wait}s")

                    except Exception as e:
                        print(f"  → Error closing long: {e}")
                        orders.append(OrderResult(symbol=symbol, side="SELL", qty=current_qty, success=False, error=str(e)))
                        continue
                    
                    # Order 2: Open the short
                    try:
                        short_request = MarketOrderRequest(
                            symbol=symbol,
                            qty=abs(target_qty),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY
                        )
                        short_order = self.client.submit_order(short_request)
                        print(f"  → Opened short: SELL {abs(target_qty)} (Order ID: {short_order.id})")
                        
                        orders.append(OrderResult(symbol=symbol, side="SELL+SHORT", qty=abs(delta), success=True, order_id=short_order.id))
                    except Exception as e:
                        print(f"  → Error opening short: {e}")
                        orders.append(OrderResult(symbol=symbol, side="SHORT", qty=abs(target_qty), success=False, error=str(e)))
                    
                    continue  # Skip normal order submission below

                # SPECIAL CASE 2: Going from short to long (cover + buy)
                if current_qty < 0 and target_qty > 0:
                    # This works in one order, but let's log it clearly
                    print(f"[COVER + BUY] {symbol}: Covering {abs(current_qty)} short, then buying {target_qty} long")
                    
                    try:
                        order_request = MarketOrderRequest(
                            symbol=symbol,
                            qty=qty,  # Total delta
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY
                        )
                        
                        order = self.client.submit_order(order_request)
                        print(f"  → Submitted: BUY {qty} shares (Order ID: {order.id})")
                        
                        orders.append(OrderResult(
                            symbol=symbol,
                            side="COVER+BUY",
                            qty=qty,
                            success=True,
                            order_id=order.id
                        ))
                        continue  # Skip normal submission
                    except Exception as e:
                        print(f"  → Error opening short: {e}")
                        orders.append(OrderResult(symbol=symbol, side="BUY", qty=qty, success=False, error=str(e)))
                    
                    continue  # Skip the normal order submission below

                if dry_run:
                    print(f"   [DRY RUN] {side.value} {qty} shares of {symbol} "
                        f"(current: {current_qty}, target: {target_qty})")
                    orders.append(OrderResult(
                        symbol=symbol,
                        side=side.value,
                        qty=qty,
                        success=True,
                        order_id="DRY_RUN"
                    ))
                    continue

            #submit market order
            try:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
                
                order = self.client.submit_order(order_request)

                print(f"Submitted {side.value} order for {qty} shares of {symbol} "
                      f"(Order ID: {order.id})")
                
                orders.append(OrderResult(
                    symbol=symbol,
                    side=side.value,
                    qty=qty,
                    success=True,
                    order_id=order.id
                ))

            except Exception as e:
                print(f"Error submitting order for {symbol}: {e}")
                orders.append(OrderResult(
                    symbol=symbol,
                    side=side.value,
                    qty=qty,
                    success=False,
                    error=str(e)
                ))

        return orders

    def close_all_positions(self):
        """Closes all open positions (liquidate portfolio)."""
        try:
            self.client.close_all_positions(cancel_orders=True)
            print("All positions closed successfully.")
        except Exception as e:
            print(f"Error closing positions: {e}")


# -------------------------------------------------------
# This block only runs when you execute file directly:
#   python3 execution/trader.py
# Does NOT run when another file imports from this module
# -------------------------------------------------------

if __name__ == "__main__":
    from model.config.settings import AlpacaConfig

    print("=== Alpaca Trader Test ===\n")

    #Initialize
    config = AlpacaConfig()
    trader = AlpacaTrader(config)

    #get account info
    account = trader.get_account()
    print(f"\nAccount Status: {account.status}")
    print(f"Portfolio Value: ${float(account.equity):,.2f}")
    print(f"Cash: ${float(account.cash):,.2f}")
    print(f"Buying Power: ${float(account.buying_power):,.2f}")

    #get current positions
    positions = trader.get_positions()
    print(f"\nCurrent Positions: {len(positions)}")
    for symbol, qty in positions.items():
        print(f"  {symbol}: {qty} shares")

    # Test order submission (DRY RUN)
    print("\n=== Testing Order Submission (DRY RUN) ===\n")

    test_weights = {
        "AAPL": 0.10,
        "MSFT": 0.15,  
        "GOOG": -0.05  #short position
    }

    test_prices = {
        "AAPL": 150.0,
        "MSFT": 380.0,
        "GOOG": 140.0
    }

    orders = trader.submit_orders(
        target_weights=test_weights,
        current_prices=test_prices,
        dry_run=True
    )

    print(f"\n{len(orders)} orders generated (not submitted)")


