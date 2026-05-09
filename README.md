# Strategy Regime Stress Lab

## 繁體中文

Strategy Regime Stress Lab 是一個 Python 研究工具，用於協助小型交易團隊針對策略報酬序列進行 market regime 壓力測試。它是研究與風險分析工具，不是投資顧問、交易訊號產品，也不是股票推薦 App。

### 功能

- 讀取本地 `market_prices.csv` 與 `strategy_returns.csv`。
- 從 ETF OHLCV 資料建立避免 look-ahead bias 的 market regime labels。
- 執行 regime-conditioned block bootstrap 與 historical replay stress tests。
- 輸出 CAGR、volatility、Sharpe、Sortino、max drawdown、VaR、CVaR、skew、kurtosis 與 regime-wise performance。
- 產生 deterministic HTML report，只引用已計算的 metrics。
- 提供 optional deep-generator 介面；預設為 bootstrap，不會在本機自動訓練 PyTorch。
- 重型模型訓練預設放在 Google Colab 或其他 GPU runtime，本機只跑測試、報告與輕量 demo。

### 套件管理

本專案使用 Poetry 管理 Python 套件與虛擬環境。請先在系統環境安裝 Poetry，並使用系統可見的 `poetry` 指令，不要使用 Codex runtime 內建的 Poetry。

本 repo 透過 `poetry.toml` 設定讓 Poetry 在專案資料夾內建立 `.venv`：

```text
/Users/hawkiyc/Documents/strategy-regime-stress-lab/.venv
```

安裝依賴：

```bash
poetry install
```

若需要 Colab / notebook 相關套件：

```bash
poetry install --extras notebook
```

若需要在 GPU runtime 訓練 deep generator：

```bash
poetry install --extras deep
```

### 快速開始

```bash
poetry run regime-lab stress \
  --market examples/sample_market_prices.csv \
  --returns examples/sample_strategy_returns.csv \
  --out reports/stress_report.html
```

如果自行產生或下載 ETF 資料，請使用以下 schema：

```csv
date,symbol,open,high,low,close,volume
2024-01-02,SPY,100,101,99,100.5,1000000
```

策略報酬使用以下 schema：

```csv
date,strategy_id,return,benchmark,exposure
2024-01-02,alpha,0.001,0.0005,0.8
```

其中 `date`、`strategy_id`、`return` 為必要欄位。

### Python API

```python
from regime_lab import (
    build_report,
    fit_deep_generator,
    fit_regime_model,
    load_generator,
    load_market_data,
    load_strategy_returns,
    make_return_windows,
    stress_test,
)

market = load_market_data("market_prices.csv")
returns = load_strategy_returns("strategy_returns.csv")
regimes = fit_regime_model(market, n_regimes=4)
results = stress_test(returns, regimes, n_sims=1000, seed=42)
build_report(results, output="report.html")
```

### Deep Generator 資料流程

conditional WGAN-style generator 不直接吃 raw OHLCV rows，而是吃固定長度的 return windows 與每段 window 對應的 regime label：

- `windows`: shape `(n_windows, window_length)`，通常是策略報酬的 rolling slices。
- `regime_labels`: shape `(n_windows,)`，通常是每段 window 內的 majority market regime。

從真實策略報酬與 regime labels 建立訓練資料：

```python
window_data = make_return_windows(returns, regimes.regimes, window_length=20, stride=1)
```

本機預設為 bootstrap，不會訓練 PyTorch：

```python
generator = fit_deep_generator(window_data.windows, window_data.labels)
augmented = generator.augment(window_data.windows, window_data.labels, n_per_regime=100)
```

在 Colab GPU 訓練時，才明確指定 PyTorch backend：

```python
generator = fit_deep_generator(window_data.windows, window_data.labels, backend="torch", epochs=25)
```

Colab 訓練完成後，下載模型 artifact 並在本機載入：

```python
generator = load_generator("artifacts/models/deep_generator.pkl")
augmented = generator.augment(window_data.windows, window_data.labels, n_per_regime=100)
```

`augmented` dataset 同時包含真實與生成 windows，並以 `source == "real"` 或 `source == "synthetic"` 標記來源。

### 法規與產品邊界

- 本專案不產生 buy/sell/hold recommendations、target prices 或個人化投資建議。
- 報告是 deterministic，且只基於已計算 metrics。
- v1 不包含 LLM-generated narrative。若未來加入，也只能摘要已計算 metrics。
- 本 repo 不是法律意見。若未來收費並輸出證券分析或投資建議，可能涉及投顧或投資顧問登記/牌照義務，需先諮詢台灣與美國證券法專業人士。

參考：

- 台灣證券投資信託及顧問法: https://law.fsc.gov.tw/LawContent.aspx?id=FL030633
- U.S. investment adviser definition: https://www.law.cornell.edu/uscode/text/15/80b-2
- SEC internet adviser reforms: https://www.sec.gov/newsroom/press-releases/2024-42
- SEC AI-washing enforcement: https://www.sec.gov/newsroom/press-releases/2024-36
- FINRA Notice 24-09: https://www.finra.org/rules-guidance/notices/24-09

### 資料說明

核心 package 只讀取本地 CSV。若使用 Yahoo Finance、Polygon、Nasdaq Data Link、Alpha Vantage、券商匯出資料或內部資料，請先確認資料供應商授權條款，再散布資料或報告。

`examples/` 內的 sample files 是 synthetic data，只用來示範工作流程。

### 開發與測試

```bash
poetry install
poetry run python -m unittest discover -s tests
```

如果你使用 VS Code，請先查詢 Poetry 環境路徑：

```bash
poetry env info --path
```

它應該回傳專案內 `.venv`。然後在 VS Code 的 `Python: Select Interpreter` 手動選擇該路徑底下的 `bin/python`。本 repo 不會提交 `.vscode/settings.json`；IDE 設定留在你的本機。

### Colab 訓練政策

本專案假設本機可能是 CPU-only 或舊硬體。請不要預設在本機訓練重型 deep-generator。

建議流程：

1. 本機只跑 baseline regime 與 stress-test engine。
2. deep-generator experiments 放在 Google Colab 並啟用 GPU。
3. 在 Colab 將 trained artifacts 存到 `artifacts/models/`。
4. 訓練完成後，只下載 trained model artifacts 與 evaluation reports 回本 repo。
5. 如果 Colab free-tier GPU memory 或 runtime 不夠，停止訓練並評估是否升級 Colab Pro。

詳見 [notebooks/04_colab_deep_generator_training.ipynb](notebooks/04_colab_deep_generator_training.ipynb)。

---

## English

Strategy Regime Stress Lab is a Python research toolkit for small trading teams that want to stress test strategy return series across market regimes. It is a research and risk analysis tool, not an investment adviser, trading signal product, or stock recommendation app.

### Features

- Loads local `market_prices.csv` and `strategy_returns.csv` files.
- Builds no-lookahead market regime labels from ETF OHLCV data.
- Runs regime-conditioned block bootstrap and historical replay stress tests.
- Reports CAGR, volatility, Sharpe, Sortino, max drawdown, VaR, CVaR, skew, kurtosis, and regime-wise performance.
- Generates deterministic HTML reports that cite computed metrics only.
- Provides an optional deep-generator interface; the default backend is bootstrap and will not train PyTorch locally.
- Keeps heavy model training on Google Colab or another GPU runtime. Local machines should run tests, reports, and lightweight demos only.

### Package Management

This project uses Poetry for Python dependency and environment management. Install Poetry in your system environment first and use the system-visible `poetry` command, not Poetry from the Codex runtime.

This repository uses `poetry.toml` so Poetry creates `.venv` inside the project folder:

```text
/Users/hawkiyc/Documents/strategy-regime-stress-lab/.venv
```

Install dependencies:

```bash
poetry install
```

For notebook-related dependencies:

```bash
poetry install --extras notebook
```

For deep-generator training in a GPU runtime:

```bash
poetry install --extras deep
```

### Quick Start

```bash
poetry run regime-lab stress \
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

### Python API

```python
from regime_lab import (
    build_report,
    fit_deep_generator,
    fit_regime_model,
    load_generator,
    load_market_data,
    load_strategy_returns,
    make_return_windows,
    stress_test,
)

market = load_market_data("market_prices.csv")
returns = load_strategy_returns("strategy_returns.csv")
regimes = fit_regime_model(market, n_regimes=4)
results = stress_test(returns, regimes, n_sims=1000, seed=42)
build_report(results, output="report.html")
```

### Deep Generator Data Flow

The conditional WGAN-style generator does not consume raw OHLCV rows directly. It consumes fixed-length return windows and one regime label per window:

- `windows`: shape `(n_windows, window_length)`, usually rolling slices of strategy returns.
- `regime_labels`: shape `(n_windows,)`, usually the majority market regime inside each return window.

Build those inputs from real strategy returns and regime labels:

```python
window_data = make_return_windows(returns, regimes.regimes, window_length=20, stride=1)
```

Default local behavior is bootstrap-only and does not train PyTorch:

```python
generator = fit_deep_generator(window_data.windows, window_data.labels)
augmented = generator.augment(window_data.windows, window_data.labels, n_per_regime=100)
```

For Colab GPU training, explicitly request the PyTorch backend:

```python
generator = fit_deep_generator(window_data.windows, window_data.labels, backend="torch", epochs=25)
```

After Colab training, download the saved artifact and load it locally:

```python
generator = load_generator("artifacts/models/deep_generator.pkl")
augmented = generator.augment(window_data.windows, window_data.labels, n_per_regime=100)
```

The `augmented` dataset contains both real and synthetic windows and marks each row with `source == "real"` or `source == "synthetic"`.

### Guardrails

- The project does not produce buy/sell/hold recommendations, target prices, or personalized investment advice.
- The generated report is deterministic and based on computed metrics.
- LLM-generated narrative is intentionally excluded from v1. If added later, it should only summarize already-computed metrics.
- This repository is not legal advice. Paid securities analysis or investment advice may trigger registration or licensing duties. Review relevant Taiwan and U.S. rules before commercial deployment.

References:

- Taiwan Securities Investment Trust and Consulting Act: https://law.fsc.gov.tw/LawContent.aspx?id=FL030633
- U.S. investment adviser definition: https://www.law.cornell.edu/uscode/text/15/80b-2
- SEC internet adviser reforms: https://www.sec.gov/newsroom/press-releases/2024-42
- SEC AI-washing enforcement: https://www.sec.gov/newsroom/press-releases/2024-36
- FINRA Notice 24-09: https://www.finra.org/rules-guidance/notices/24-09

### Data Notes

The package core only reads local CSV files. If you use third-party market data such as Yahoo Finance, Polygon, Nasdaq Data Link, Alpha Vantage, broker exports, or internal data, check the provider's license before redistributing data or reports.

The sample files in `examples/` are synthetic and exist only to demonstrate the workflow.

### Development And Testing

```bash
poetry install
poetry run python -m unittest discover -s tests
```

If you use VS Code, first inspect the Poetry environment path:

```bash
poetry env info --path
```

It should point to the in-project `.venv`. Then use `Python: Select Interpreter` and choose `bin/python` under that path. This repository does not commit `.vscode/settings.json`; IDE settings stay local to your machine.

### Colab Training Policy

This project assumes local hardware may be CPU-only or old. Do not run heavy deep-generator training locally by default.

Recommended workflow:

1. Use the baseline regime and stress-test engine locally.
2. Run deep-generator experiments in Google Colab with GPU enabled.
3. Save trained artifacts under `artifacts/models/` in Colab.
4. Download only the trained model artifacts and evaluation reports back to this repository.
5. If Colab free-tier GPU memory or runtime is insufficient, stop the run and consider Colab Pro before increasing model size.

See [notebooks/04_colab_deep_generator_training.ipynb](notebooks/04_colab_deep_generator_training.ipynb) for the Colab-oriented workflow.
