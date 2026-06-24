import requests
import pandas as pd
import sqlite3
import time
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Read the key from environment — never hardcoded
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

def fetch_stock_data(ticker):
    print(f"Fetching data for {ticker}...")

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": "compact",
        "apikey": API_KEY
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if "Time Series (Daily)" not in data:
        print("Error — API response was:")
        print(data)
        return None

    time_series = data["Time Series (Daily)"]

    rows = []
    for date, values in time_series.items():
        row = {
            "date":   date,
            "open":   float(values["1. open"]),
            "high":   float(values["2. high"]),
            "low":    float(values["3. low"]),
            "close":  float(values["4. close"]),
            "volume": int(values["5. volume"])
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Got {len(df)} days of data for {ticker}")
    return df


def save_to_database(df, ticker):
    print(f"Saving {ticker} to database...")

    conn = sqlite3.connect("quantiq.db")
    df["ticker"] = ticker

    # append = add to existing table, don't overwrite other tickers
    df.to_sql("stock_prices", conn, if_exists="append", index=False)

    conn.close()
    print("Saved successfully.")


def load_from_database(ticker):
    conn = sqlite3.connect("quantiq.db")
    df = pd.read_sql(
        f"SELECT * FROM stock_prices WHERE ticker = '{ticker}'",
        conn
    )
    conn.close()
    return df


def load_all_from_database():
    conn = sqlite3.connect("quantiq.db")
    df = pd.read_sql("SELECT * FROM stock_prices", conn)
    conn.close()
    return df


def fetch_multiple_stocks(tickers):
    all_data = []

    for ticker in tickers:
        print(f"\nFetching {ticker}...")
        df = fetch_stock_data(ticker)

        if df is not None:
            df["ticker"] = ticker
            all_data.append(df)
            save_to_database(df, ticker)

            print("Waiting 15 seconds (API rate limit)...")
            time.sleep(15)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal combined rows: {len(combined)}")
        return combined

    return None


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]

    combined_df = fetch_multiple_stocks(tickers)

    if combined_df is not None:
        print("\nRows per stock:")
        print(combined_df.groupby("ticker").size())