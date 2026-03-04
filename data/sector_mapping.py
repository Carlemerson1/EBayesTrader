"""
data/sector_mapping.py

Static sector classification for stocks.
Add new stocks here as needed.
"""

SECTOR_MAP = {
    # Technology
    'AAPL': 'tech', 'GOOGL': 'tech', 'GOOG': 'tech', 'MSFT': 'tech', 
    'META': 'tech', 'AMZN': 'tech', 'NVDA': 'tech', 'AMD': 'tech', 
    'TSLA': 'tech', 'AVGO': 'tech', 'ORCL': 'tech', 'CRM': 'tech', 
    'ADBE': 'tech', 'NFLX': 'tech', 'INTC': 'tech', 'QCOM': 'tech',
    'CRWD': 'tech', 'PLTR': 'tech', 'SNOW': 'tech',
    
    # Finance
    'JPM': 'finance', 'BAC': 'finance', 'GS': 'finance', 'MS': 'finance',
    'V': 'finance', 'C': 'finance', 'WFC': 'finance', 'BLK': 'finance',
    'SCHW': 'finance', 'AXP': 'finance',
    
    # Energy
    'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy', 'SLB': 'energy',
    'EOG': 'energy', 'PXD': 'energy', 'MPC': 'energy', 'VLO': 'energy',
    'OXY': 'energy',
    
    # Healthcare
    'JNJ': 'healthcare', 'UNH': 'healthcare', 'PFE': 'healthcare',
    'ABBV': 'healthcare', 'TMO': 'healthcare', 'ABT': 'healthcare',
    'BMY': 'healthcare', 'LLY': 'healthcare', 'MRK': 'healthcare',
    'AMGN': 'healthcare', 'ISRG': 'healthcare', 'VRTX': 'healthcare',
    
    # Consumer
    'WMT': 'consumer', 'HD': 'consumer', 'MCD': 'consumer', 'NKE': 'consumer',
    'SBUX': 'consumer', 'TGT': 'consumer', 'LOW': 'consumer', 'COST': 'consumer',
    'DIS': 'consumer', 'CMCSA': 'consumer', 'SHOP': 'consumer',
    
    # Industrial
    'CAT': 'industrial', 'BA': 'industrial', 'GE': 'industrial',
    'HON': 'industrial', 'UPS': 'industrial', 'LMT': 'industrial',
    'MMM': 'industrial', 'AMAT': 'industrial',
    
    # Semiconductors
    'ASML': 'semi', 'TSM': 'semi',
    
    # Bonds
    'TLT': 'bonds', 'IEF': 'bonds', 'LQD': 'bonds', 'HYG': 'bonds',
    'AGG': 'bonds', 'BND': 'bonds',
    
    # Market ETFs
    'VOO': 'market', 'SPY': 'market', 'QQQ': 'market', 'DIA': 'market',
    'VTI': 'market',
}


def get_sector(symbol: str) -> str:
    """
    Get sector for a symbol.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Sector name (e.g., 'tech', 'finance', 'energy')
        Returns 'other' if symbol not in mapping
    """
    return SECTOR_MAP.get(symbol, 'other')


def get_symbols_by_sector(symbols: list[str]) -> dict[str, list[str]]:
    """
    Group symbols by sector.
    
    Args:
        symbols: List of stock symbols
        
    Returns:
        Dict mapping {sector: [symbols]}
    """
    sector_groups = {}
    
    for symbol in symbols:
        sector = get_sector(symbol)
        if sector not in sector_groups:
            sector_groups[sector] = []
        sector_groups[sector].append(symbol)
    
    return sector_groups


def print_sector_summary(symbols: list[str]):
    """Print a summary of sectors in the universe."""
    sector_groups = get_symbols_by_sector(symbols)
    
    print("\n" + "="*60)
    print("SECTOR BREAKDOWN")
    print("="*60)
    
    for sector, sector_symbols in sorted(sector_groups.items()):
        pct = len(sector_symbols) / len(symbols) * 100
        print(f"{sector.upper():<15} {len(sector_symbols):>3} stocks ({pct:>5.1f}%)")
        print(f"  {', '.join(sorted(sector_symbols))}")
    
    print("="*60 + "\n")