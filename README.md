# Strategy Regime Stress Lab

Strategy Regime Stress Lab is a Python research toolkit for small trading teams that want to stress test strategy return series across market regimes. It is a risk analysis and research tool, not an investment adviser, trading signal product, or stock recommendation app.

## What It Does

- Loads local `market_prices.csv` and `strategy_returns.csv` files.
- Builds no-lookahead market regime labels from ETF OHLCV data.
- Runs regime-conditioned block bootstrap and historical replay stress tests.
- Reports CAGR, volatility, Sharpe, Sortino, max drawdown, VaR, CVaR, skew, kurtosis, and regime-wise performance.
- Generates deterministic HTML reports that cite computed metrics only.
- Provides an optional deep-generator interface with a dependency-free fallback when PyTorch is unavailable.
- Keeps heavy model training on Google Colab or another GPU runtime. Local machines should run package tests, reports, and lightweight demos only.

## Quick Start

```bash
python -m regime_lab.cli stress \
  --market examples/sample_market_prices.csv \
  --returns examples/sample_strategy_returns.csv \
  --out reports/stress_report.html
```

If you generate or download your own ETF data, use this schema:

```csv
date,symbol,open,high,low,close,volume
2024-01-02,SPY,100,101,99,100.5,1000000
```

Strategy returns use this schema:

```csv
date,strategy_id,return,benchmark,exposure
2024-01-02,alpha,0.001,0.0005,0.8
```

Only `date`, `strategy_id`, and `return` are required for strategy returns.

## Python API

```python
from regime_lab import (
    build_report,
    fit_deep_generator,
    fit_regime_model,
    load_market_data,
    load_strategy_returns,
    stress_test,
)

market = load_market_data("market_prices.csv")
returns = load_strategy_returns("strategy_returns.csv")
regimes = fit_regime_model(market, n_regimes=4)
results = stress_test(returns, regimes, n_sims=1000, seed=42)
build_report(results, output="report.html")
```

## Guardrails

- The project does not produce buy/sell/hold recommendations, target prices, or personalized investment advice.
- The generated report is deterministic and based on computed metrics.
- LLM-generated narrative is intentionally excluded from v1. If added later, it should only summarize already-computed metrics.
- This repository is not legal advice. Paid securities analysis or investment advice may trigger registration or licensing duties. Review relevant Taiwan and U.S. rules before commercial deployment:
  - Taiwan Securities Investment Trust and Consulting Act: https://law.fsc.gov.tw/LawContent.aspx?id=FL030633
  - U.S. investment adviser definition: https://www.law.cornell.edu/uscode/text/15/80b-2
  - SEC internet adviser reforms: https://www.sec.gov/newsroom/press-releases/2024-42
  - SEC AI-washing enforcement: https://www.sec.gov/newsroom/press-releases/2024-36
  - FINRA Notice 24-09: https://www.finra.org/rules-guidance/notices/24-09

## Data Notes

The package core only reads local CSV files. If you use third-party market data such as Yahoo Finance, Polygon, Nasdaq Data Link, Alpha Vantage, broker exports, or internal data, check the provider's license before redistributing data or reports.

The included sample files are synthetic and exist only to demonstrate the workflow.

## Development

Run the test suite with the bundled Python runtime or any environment that has `numpy` and `pandas` installed:

```bash
python -m unittest discover -s tests
```

## Colab Training Policy

This project assumes local hardware may be CPU-only or old. Do not run heavy deep-generator training locally by default.

Recommended workflow:

1. Use the baseline regime and stress-test engine locally.
2. Run deep-generator experiments in Google Colab with GPU enabled.
3. Save trained artifacts under `artifacts/models/` in Colab.
4. Download only the trained model artifacts and evaluation reports back to this repository.
5. If Colab free-tier GPU memory or runtime is insufficient, stop the run and consider Colab Pro before increasing model size.

See [notebooks/04_colab_deep_generator_training.ipynb](notebooks/04_colab_deep_generator_training.ipynb) for the Colab-oriented workflow.
