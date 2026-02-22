"""
dashboard/visualizer.py

Professional trading dashboard for Bayesian hierarchical strategy.

Clean, minimal design inspired by academic research papers.
Color scheme: Blue (#00539B) and white.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from execution.trader import AlpacaTrader
from config.settings import AlpacaConfig
from backtest.metrics import compute_drawdowns

# Color palette
PRIMARY_BLUE = '#00539B'
LIGHT_BLUE = '#E8F4F8'
RED = '#DC3545'
GRAY = '#6C757D'
DARK_GRAY = '#343A40'
WHITE = '#FFFFFF'

# Page config
st.set_page_config(
    page_title="Bayesian Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional look
st.markdown(f"""
<style>
    /* Global styles */
    .main {{
        background-color: {WHITE};
        font-family: 'Computer Modern', 'Times New Roman', serif;
    }}
    
    /* Headers */
    h1 {{
        color: {PRIMARY_BLUE};
        font-family: 'Computer Modern', 'Times New Roman', serif;
        font-weight: 600;
        font-size: 2.5rem;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }}
    
    h2 {{
        color: {PRIMARY_BLUE};
        font-family: 'Computer Modern', 'Times New Roman', serif;
        font-weight: 500;
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid {LIGHT_BLUE};
        padding-bottom: 0.5rem;
    }}
    
    h3 {{
        color: {DARK_GRAY};
        font-family: 'Computer Modern', 'Times New Roman', serif;
        font-weight: 500;
        font-size: 1.1rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }}
    
    /* Metrics */
    .stMetric {{
        background-color: {WHITE};
        padding: 1.5rem;
        border: 1px solid {LIGHT_BLUE};
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    
    .stMetric label {{
        color: {DARK_GRAY};
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .stMetric [data-testid="stMetricValue"] {{
        color: {PRIMARY_BLUE};
        font-size: 1.75rem;
        font-weight: 600;
    }}
    
    /* Tables */
    .dataframe {{
        border: 1px solid {LIGHT_BLUE};
        border-radius: 4px;
        overflow: hidden;
    }}
    
    .dataframe thead tr th {{
        background-color: {LIGHT_BLUE};
        color: {PRIMARY_BLUE};
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.5px;
        padding: 1rem;
        border: none;
    }}
    
    .dataframe tbody tr td {{
        padding: 0.75rem 1rem;
        border-bottom: 1px solid {LIGHT_BLUE};
    }}
    
    /* Remove default Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Dividers */
    hr {{
        margin: 2rem 0;
        border: none;
        border-top: 1px solid {LIGHT_BLUE};
    }}
</style>
""", unsafe_allow_html=True)


def load_live_data():
    """Load current portfolio data from Alpaca."""
    try:
        config = AlpacaConfig()
        trader = AlpacaTrader(config)
        
        account = trader.get_account()
        positions = trader.get_positions()
        portfolio_value = float(account.equity)
        
        return {
            'portfolio_value': portfolio_value,
            'cash': float(account.cash),
            'positions': positions,
            'buying_power': float(account.buying_power),
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def create_equity_curve(equity_data, benchmark_data=None):
    """Create clean equity curve visualization."""
    fig = go.Figure()
    
    # Strategy line
    fig.add_trace(go.Scatter(
        x=equity_data.index,
        y=equity_data.values,
        mode='lines',
        name='Strategy',
        line=dict(color=PRIMARY_BLUE, width=2.5),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>$%{y:,.2f}<extra></extra>'
    ))
    
    # Benchmark line
    if benchmark_data is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_data.index,
            y=benchmark_data.values,
            mode='lines',
            name='Benchmark',
            line=dict(color=GRAY, width=1.5, dash='dot'),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>$%{y:,.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Portfolio Value (USD)',
        hovermode='x unified',
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(family='Computer Modern, Times New Roman', size=11, color=DARK_GRAY),
        height=400,
        margin=dict(l=60, r=20, t=20, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=10)
        ),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor=LIGHT_BLUE,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor=LIGHT_BLUE,
            showline=True,
            linewidth=1,
            linecolor=LIGHT_BLUE,
            tickfont=dict(size=10),
            tickformat='$,.0f'
        )
    )
    
    return fig


def create_drawdown_chart(drawdown_series):
    """Create underwater equity curve."""
    fig = go.Figure()
    
    # Filled area
    fig.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=drawdown_series.values * 100,
        fill='tozeroy',
        mode='lines',
        name='Drawdown',
        line=dict(color=RED, width=2),
        fillcolor='rgba(220, 53, 69, 0.15)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.2f}%<extra></extra>'
    ))
    
    # Trigger threshold line
    fig.add_hline(
        y=-15,
        line_dash="dash",
        line_width=1,
        line_color=RED,
        opacity=0.5,
        annotation_text="Circuit Breaker (-15%)",
        annotation_position="right",
        annotation_font=dict(size=9, color=RED)
    )
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(family='Computer Modern, Times New Roman', size=11, color=DARK_GRAY),
        height=400,
        margin=dict(l=60, r=20, t=20, b=40),
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor=LIGHT_BLUE,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor=LIGHT_BLUE,
            showline=True,
            linewidth=1,
            linecolor=LIGHT_BLUE,
            tickfont=dict(size=10)
        )
    )
    
    return fig


def create_positions_table(positions):
    """Format positions as DataFrame."""
    if not positions:
        return pd.DataFrame()
    
    data = []
    for symbol, qty in positions.items():
        side = 'Long' if qty > 0 else 'Short'
        data.append({
            'Symbol': symbol,
            'Position': side,
            'Shares': abs(int(qty)),
        })
    
    return pd.DataFrame(data)


def create_signal_bars(signals_data):
    """Create clean horizontal bar chart for signals."""
    if not signals_data:
        return go.Figure()
    
    symbols = list(signals_data.keys())
    probs = [signals_data[s]['prob'] for s in symbols]
    
    # Color based on direction
    colors = [PRIMARY_BLUE if p >= 0.5 else RED for p in probs]
    
    fig = go.Figure(go.Bar(
        y=symbols,
        x=probs,
        orientation='h',
        marker=dict(color=colors, opacity=0.8),
        text=[f"{p:.1%}" for p in probs],
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate='<b>%{y}</b><br>P(μ > 0) = %{x:.1%}<extra></extra>'
    ))
    
    # 50% neutral line
    fig.add_vline(x=0.5, line_dash="dash", line_width=1, line_color=GRAY, opacity=0.5)
    
    fig.update_layout(
        xaxis_title='Probability of Positive Return',
        yaxis_title='',
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(family='Computer Modern, Times New Roman', size=11, color=DARK_GRAY),
        height=300,
        margin=dict(l=60, r=60, t=20, b=40),
        xaxis=dict(
            range=[0, 1],
            showgrid=True,
            gridwidth=0.5,
            gridcolor=LIGHT_BLUE,
            showline=True,
            linewidth=1,
            linecolor=LIGHT_BLUE,
            tickfont=dict(size=10),
            tickformat='.0%'
        ),
        yaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor=LIGHT_BLUE,
            tickfont=dict(size=10)
        )
    )
    
    return fig


def format_currency(value):
    """Format value as currency."""
    return f"${value:,.2f}"


def format_percentage(value):
    """Format value as percentage."""
    return f"{value:.2f}%"


def main():
    """Main dashboard application."""
    
    # Title
    st.markdown(f"""
    <h1 style='text-align: center; margin-bottom: 0.25rem;'>
        Bayesian Hierarchical Trading System
    </h1>
    <p style='text-align: center; color: {GRAY}; font-size: 0.875rem; margin-top: 0;'>
        Real-time Portfolio Monitor • Paper Trading Environment
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Load data
    live_data = load_live_data()
    
    if live_data:
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Portfolio Value",
                format_currency(live_data['portfolio_value'])
            )
        
        with col2:
            st.metric(
                "Available Cash",
                format_currency(live_data['cash'])
            )
        
        with col3:
            st.metric(
                "Buying Power",
                format_currency(live_data['buying_power'])
            )
        
        with col4:
            st.metric(
                "Open Positions",
                len(live_data['positions'])
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Primary charts (two columns)
        st.markdown("## Performance")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### Equity Curve")
            # Demo data (replace with actual backtest results)
            dates = pd.date_range('2024-01-01', periods=100, freq='D')
            equity = pd.Series(
                100000 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100))),
                index=dates
            )
            fig_equity = create_equity_curve(equity)
            st.plotly_chart(fig_equity, use_container_width=True)
        
        with col_right:
            st.markdown("### Drawdown")
            drawdown = compute_drawdowns(equity)
            fig_dd = create_drawdown_chart(drawdown)
            st.plotly_chart(fig_dd, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Secondary section
        st.markdown("## Holdings & Metrics")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### Current Positions")
            positions_df = create_positions_table(live_data['positions'])
            if not positions_df.empty:
                st.dataframe(
                    positions_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                        "Position": st.column_config.TextColumn("Position", width="small"),
                        "Shares": st.column_config.NumberColumn("Shares", width="small"),
                    }
                )
            else:
                st.info("No open positions", icon="ℹ️")
        
        with col_right:
            st.markdown("### Performance Metrics")
            # Placeholder metrics
            metrics_data = pd.DataFrame({
                'Metric': [
                    'Sharpe Ratio',
                    'Sortino Ratio',
                    'Calmar Ratio',
                    'Maximum Drawdown',
                    'Win Rate'
                ],
                'Value': ['1.64', '2.34', '2.09', '-13.0%', '54.3%']
            })
            st.dataframe(
                metrics_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Metric": st.column_config.TextColumn("Metric", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="small"),
                }
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tertiary section
        st.markdown("## Model State")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### Signal Strength")
            # Demo signal data
            sample_signals = {
                'XOM': {'prob': 0.955},
                'OXY': {'prob': 0.893},
                'TLT': {'prob': 0.815},
                'AAPL': {'prob': 0.450},
                'MSFT': {'prob': 0.261},
            }
            fig_signals = create_signal_bars(sample_signals)
            st.plotly_chart(fig_signals, use_container_width=True)
        
        with col_right:
            st.markdown("### Recent Activity")
            activity_data = pd.DataFrame({
                'Timestamp': [
                    '2026-02-21 09:35',
                    '2026-02-21 09:36',
                    '2026-02-14 14:23'
                ],
                'Event': [
                    'Portfolio rebalanced',
                    'Executed 4 orders',
                    'Stop-loss: AAPL (-1.2%)'
                ]
            })
            st.dataframe(
                activity_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Timestamp": st.column_config.TextColumn("Time", width="small"),
                    "Event": st.column_config.TextColumn("Event", width="large"),
                }
            )
    
    else:
        st.error("Unable to connect to Alpaca API. Verify credentials in .env file.")
    
    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <p style='text-align: center; color: {GRAY}; font-size: 0.75rem;'>
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EST
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()