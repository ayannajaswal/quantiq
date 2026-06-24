import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

from data_fetcher import load_all_from_database
from indicators import add_all_indicators


def create_labels(df):
    """
    For each stock separately, calculate if tomorrow
    was higher or lower than today.
    We do it per ticker so we don't compare
    Apple's price to Tesla's price accidentally.
    """
    result = []

    for ticker in df["ticker"].unique():
        stock = df[df["ticker"] == ticker].copy()
        stock = stock.sort_values("date").reset_index(drop=True)
        stock["tomorrow_close"] = stock["close"].shift(-1)
        stock["label"] = (stock["tomorrow_close"] > stock["close"]).astype(int)
        stock = stock.dropna().reset_index(drop=True)
        result.append(stock)

    combined = pd.concat(result, ignore_index=True)
    return combined


FEATURES = [
    "ma_5",
    "ma_20",
    "rsi",
    "bollinger_upper",
    "bollinger_lower",
    "momentum",
    "volume_spike"
]


def train_model(df):
    print("Preparing data...")

    X = df[FEATURES]
    y = df["label"]

    print(f"Total samples : {len(df)}")
    print(f"UP days       : {y.sum()}")
    print(f"DOWN days     : {(y==0).sum()}")

    # For multi-stock data we can shuffle
    # because we're not predicting one stock's sequence
    # we're learning general patterns across all stocks
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    print(f"\nTraining on {len(X_train)} samples")
    print(f"Testing on  {len(X_test)} samples")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nModel Accuracy: {accuracy * 100:.1f}%")
    print("\nDetailed Report:")
    print(classification_report(
        y_test, predictions,
        target_names=["DOWN", "UP"]
    ))

    return model, X_test, y_test


def show_feature_importance(model):
    importance = pd.DataFrame({
        "feature":    FEATURES,
        "importance": model.feature_importances_
    })
    importance = importance.sort_values("importance", ascending=False)
    importance["importance"] = (importance["importance"] * 100).round(1)

    print("\nFeature Importance:")
    print(importance.to_string(index=False))
    return importance


def predict_stock(model, ticker):
    """
    Predict tomorrow's direction for a specific stock
    using its most recent indicator values
    """
    from data_fetcher import load_from_database

    df = load_from_database(ticker)
    df = add_all_indicators(df)

    latest = df[FEATURES].iloc[-1:]
    prediction  = model.predict(latest)[0]
    probability = model.predict_proba(latest)[0]

    direction  = "UP" if prediction == 1 else "DOWN"
    confidence = probability[prediction] * 100

    print(f"\nPrediction for {ticker} tomorrow:")
    print(f"Direction  : {direction}")
    print(f"Confidence : {confidence:.1f}%")
    print(f"DOWN prob  : {probability[0]*100:.1f}%")
    print(f"UP prob    : {probability[1]*100:.1f}%")

    return direction, confidence


def save_model(model):
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("\nModel saved to model.pkl")


def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


if __name__ == "__main__":
    # Load all 4 stocks
    df = load_all_from_database()
    print(f"Loaded {len(df)} total rows from database")
    print(f"Stocks: {df['ticker'].unique()}")

    # Add indicators per stock
    print("\nCalculating indicators...")
    result = []
    for ticker in df["ticker"].unique():
        stock = df[df["ticker"] == ticker].copy()
        stock = add_all_indicators(stock)
        result.append(stock)
    df = pd.concat(result, ignore_index=True)

    # Create labels
    df = create_labels(df)
    print(f"After processing: {len(df)} samples ready for training")

    # Train
    model, X_test, y_test = train_model(df)

    # Feature importance
    show_feature_importance(model)

    # Predict all 4 stocks
    print("\n--- Predictions for tomorrow ---")
    for ticker in ["AAPL", "MSFT", "GOOGL", "TSLA"]:
        predict_stock(model, ticker)

    # Save
    save_model(model)