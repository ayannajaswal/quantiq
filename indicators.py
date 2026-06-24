import pandas as pd
import numpy as np
from data_fetcher import load_from_database

# ------------------------------------------------
# MOVING AVERAGE
# ------------------------------------------------
def calculate_ma(df, window):
    """
    Average price over last N days.
    window = how many days to average
    """
    df[f"ma_{window}"] = df["close"].rolling(window=window).mean()
    return df


# ------------------------------------------------
# RSI — Relative Strength Index
# ------------------------------------------------
def calculate_rsi(df, window=14):
    """
    Measures if stock is overbought or oversold.
    Scale: 0 to 100
    Below 40 = oversold = possible buy
    Above 65 = overbought = possible sell
    """
    delta = df["close"].diff()

    gain = delta.copy()
    loss = delta.copy()

    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    return df


# ------------------------------------------------
# BOLLINGER BANDS
# ------------------------------------------------
def calculate_bollinger(df, window=20):
    """
    A price envelope based on volatility.
    Upper band = price stretched too high
    Lower band = price stretched too low
    """
    rolling_mean = df["close"].rolling(window=window).mean()
    rolling_std  = df["close"].rolling(window=window).std()

    df["bollinger_upper"] = rolling_mean + (2 * rolling_std)
    df["bollinger_lower"] = rolling_mean - (2 * rolling_std)
    df["bollinger_mid"]   = rolling_mean

    return df


# ------------------------------------------------
# MOMENTUM
# ------------------------------------------------
def calculate_momentum(df, window=10):
    """
    How fast is the price moving?
    Positive = accelerating upward
    Negative = slowing down or falling
    """
    df["momentum"] = df["close"] - df["close"].shift(window)
    return df


# ------------------------------------------------
# VOLUME SPIKE
# ------------------------------------------------
def calculate_volume_spike(df, window=20):
    """
    Is today's volume unusually high?
    High volume confirms a real price move.
    Low volume move = might be fake/temporary
    """
    df["volume_avg"]   = df["volume"].rolling(window=window).mean()
    df["volume_spike"] = df["volume"] / df["volume_avg"]
    return df


# ------------------------------------------------
# RUN ALL INDICATORS TOGETHER
# ------------------------------------------------
def add_all_indicators(df):
    df = calculate_ma(df, 5)
    df = calculate_ma(df, 20)
    df = calculate_rsi(df)
    df = calculate_bollinger(df)
    df = calculate_momentum(df)
    df = calculate_volume_spike(df)

    # Drop rows where indicators couldn't be calculated
    # (first 20 rows won't have enough history)
    df = df.dropna().reset_index(drop=True)

    return df


# ------------------------------------------------
# TEST IT
# ------------------------------------------------
if __name__ == "__main__":
    df = load_from_database("AAPL")
    df = add_all_indicators(df)

    print(f"Shape: {df.shape}")
    print("\nColumns we now have:")
    print(df.columns.tolist())
    print("\nLast 3 rows:")
    print(df.tail(3).to_string())