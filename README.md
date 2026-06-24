# QuantIQ

A stock market analysis tool I built to learn quantitative finance and machine learning. It pulls real price data, runs technical analysis, and uses a machine learning model to predict whether a stock will go up or down the next day.

---

## What it does

- Fetches real daily stock prices from Alpha Vantage API
- Calculates technical indicators (RSI, Moving Averages, Bollinger Bands, Momentum) using pure Python math — no shortcuts
- Trains a Random Forest model on the indicators to predict next-day price direction
- Serves predictions through a REST API built with FastAPI
- Displays everything on a live dashboard
- Includes a backtesting engine that simulates trading on historical data to measure real performance

---

## Backtest results (Jan–Jun 2026, starting $10,000 each)

| Stock | My strategy | Just holding | Beat it? |
|-------|-------------|--------------|----------|
| AAPL  | +5.0%       | +13.4%       | No       |
| MSFT  | +9.1%       | -6.5%        | Yes      |
| GOOGL | +21.9%      | +12.2%       | Yes      |
| TSLA  | +38.6%      | +0.6%        | Yes      |

The model beat buy-and-hold on 3 out of 4 stocks. TSLA was the strongest result — the strategy captured two big swings that a passive holder would have missed completely.

ML accuracy on unseen test data: **63.1%** (random guessing = 50%)

---

## Why these results should be taken carefully

- Only 5 months of data — too short to draw strong conclusions
- Trade fees and slippage not included
- Only tested on 4 well-known stocks
- A longer test period across more stocks would be needed to validate properly

I included these limitations because I think being honest about what a model can and can't do is more important than making it look better than it is.

---

## Files
data_fetcher.py  — downloads stock data and saves to a local database

indicators.py    — calculates all technical indicators from scratch

model.py         — trains and evaluates the Random Forest model

server.py        — FastAPI backend that serves predictions

backtest.py      — simulates trading on historical data

dashboard.html   — frontend that connects to the API
---

## How to run it

**Install dependencies**
```bash
pip install requests pandas numpy scikit-learn fastapi uvicorn python-dotenv
```

**Add your API key**

Create a `.env` file with: ALPHAVANTAGE_API_KEY=your_key_here
Free key at: https://www.alphavantage.co/support/#api-key

**Fetch data and train the model**
```bash
python data_fetcher.py
python model.py
```

**Start the server**
```bash
uvicorn server:app --reload
```

**Open dashboard.html in your browser**

---

## Things I learned building this

- How technical indicators actually work mathematically, not just what they mean
- Why you can't shuffle time series data before splitting into train/test sets
- The difference between a model that memorizes and one that actually learns
- Why backtesting is necessary — accuracy alone doesn't tell you if a strategy makes money

---

## Built with

Python, FastAPI, scikit-learn, SQLite, Chart.js, Alpha Vantage API