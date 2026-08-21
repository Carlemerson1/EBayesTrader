# EBayesTrader

A Bayesian hierarchical trading system with an ML-based regime detector, live paper trading via Alpaca, and a Streamlit dashboard.

More on the project and live results: [carlemerson.com](https://carlemerson.com)

## Overview

EBayesTrader models asset returns using an empirical Bayes hierarchical framework, with a gradient boosting classifier gating position sizes based on detected market regime (bull / bear / neutral). The strategy trades a universe of sector ETFs (XLK, XLE, XLF) and runs daily rebalancing. Core model code (priors, posteriors, signal generation, regime detector, risk manager) lives in a private submodule to keep the proprietary logic out of the public repo while the surrounding infrastructure stays open.

Validation includes walk-forward testing and a block-bootstrap permutation test on the strategy returns, which showed significance at the 5% level. The regime detector has been validated separately with high walk-forward accuracy.

## File Structure

```
EBayesTrader/
├── model/                    # private submodule (ebayestrader-core) — not visible publicly
│   ├── prior.py
│   ├── posterior.py
│   ├── signals.py
│   ├── regime_detector_ml.py
│   ├── risk/
│   │   └── manager.py
│   └── config/
│       ├── settings.py
│       └── .env             # gitignored, not committed
├── data/
│   └── fetcher.py
├── backtest/
│   └── engine.py
├── analysis/
│   ├── validation.py
│   └── results/
│       ├── permtest/
│       └── regime_perm/
├── dashboard/
│   └── visualizer.py         # Streamlit app, deployed on Streamlit Community Cloud
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone --recurse-submodules https://github.com/Carlemerson1/EBayesTrader.git
cd EBayesTrader
pip install -r requirements.txt
```

You'll need Alpaca API keys in `model/config/.env`:

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

Note: the `model/` submodule (`ebayestrader-core`) is private. Without access to it, the repo will clone but the core model and regime detector won't be available.

## Running

```bash
python backtest/engine.py        # run backtest + validation
streamlit run dashboard/visualizer.py   # local dashboard
```

## Status

Actively paper trading. Current metrics are still short-horizon and shouldn't be treated as stable; the regime detector's behavior during a bull-market transition is the key thing being watched going forward.

## Links

- Live dashboard and write-up: [carlemerson.com](https://carlemerson.com)
