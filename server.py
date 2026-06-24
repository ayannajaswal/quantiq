from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
from datetime import datetime

from data_fetcher import load_from_database, load_all_from_database
from indicators import add_all_indicators
from model import load_model, FEATURES

# ------------------------------------------------
# CREATE THE APP
# ------------------------------------------------
app = FastAPI(
    title="QuantIQ API",
    description="AI-powered stock market analysis",
    version="1.0.0"
)

# Allow frontend to talk to backend
# Without this, browsers block cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model once when server starts
# Not on every request — that would be slow
model = load_model()
SUPPORTED_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA"]


# ------------------------------------------------
# ROUTE 1 — Health check
# Always build this first — confirms server is alive
# ------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "running",
        "project": "QuantIQ",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


# ------------------------------------------------
# ROUTE 2 — Get stock data + indicators
# ------------------------------------------------
@app.get("/stock/{ticker}")
def get_stock(ticker: str):
    ticker = ticker.upper()

    if ticker not in SUPPORTED_TICKERS:
        return {
            "error": f"{ticker} not supported",
            "supported": SUPPORTED_TICKERS
        }

    df = load_from_database(ticker)
    df = add_all_indicators(df)

    # Get last 30 rows for the chart
    recent = df.tail(30).copy()

    # Convert to list of dictionaries for JSON
    data = recent[[
        "date", "open", "high", "low",
        "close", "volume", "ma_5", "ma_20",
        "rsi", "bollinger_upper", "bollinger_lower",
        "momentum", "volume_spike"
    ]].to_dict(orient="records")

    # Latest indicators
    latest = df.iloc[-1]

    return {
        "ticker": ticker,
        "days": len(df),
        "latest_price": round(latest["close"], 2),
        "latest_rsi": round(latest["rsi"], 2),
        "latest_ma5": round(latest["ma_5"], 2),
        "latest_ma20": round(latest["ma_20"], 2),
        "latest_momentum": round(latest["momentum"], 2),
        "history": data
    }


# ------------------------------------------------
# ROUTE 3 — Get ML prediction
# ------------------------------------------------
@app.get("/predict/{ticker}")
def get_prediction(ticker: str):
    ticker = ticker.upper()

    if ticker not in SUPPORTED_TICKERS:
        return {
            "error": f"{ticker} not supported",
            "supported": SUPPORTED_TICKERS
        }

    df = load_from_database(ticker)
    df = add_all_indicators(df)

    latest = df[FEATURES].iloc[-1:]
    prediction  = model.predict(latest)[0]
    probability = model.predict_proba(latest)[0]

    direction  = "UP" if prediction == 1 else "DOWN"
    confidence = round(probability[prediction] * 100, 1)

    # Generate signal based on indicators
    latest_row = df.iloc[-1]
    signals = []

    if latest_row["rsi"] < 40:
        signals.append("RSI oversold — bullish signal")
    elif latest_row["rsi"] > 65:
        signals.append("RSI overbought — bearish signal")

    if latest_row["close"] > latest_row["ma_20"]:
        signals.append("Price above MA20 — uptrend")
    else:
        signals.append("Price below MA20 — downtrend")

    if latest_row["momentum"] > 0:
        signals.append("Positive momentum")
    else:
        signals.append("Negative momentum")

    if latest_row["volume_spike"] > 1.5:
        signals.append("High volume — move is significant")

    return {
        "ticker":        ticker,
        "direction":     direction,
        "confidence":    confidence,
        "up_probability":   round(probability[1] * 100, 1),
        "down_probability": round(probability[0] * 100, 1),
        "signals":       signals,
        "timestamp":     datetime.now().isoformat()
    }


# ------------------------------------------------
# ROUTE 4 — Compare all stocks
# ------------------------------------------------
@app.get("/compare")
def compare_all():
    results = []

    for ticker in SUPPORTED_TICKERS:
        df = load_from_database(ticker)
        df = add_all_indicators(df)

        latest      = df.iloc[-1]
        features    = df[FEATURES].iloc[-1:]
        prediction  = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        direction  = "UP" if prediction == 1 else "DOWN"
        confidence = round(probability[prediction] * 100, 1)

        results.append({
            "ticker":     ticker,
            "price":      round(latest["close"], 2),
            "rsi":        round(latest["rsi"], 2),
            "momentum":   round(latest["momentum"], 2),
            "direction":  direction,
            "confidence": confidence
        })

    return {"stocks": results}