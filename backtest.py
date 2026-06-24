import pandas as pd
import numpy as np
from data_fetcher import load_from_database
from indicators import add_all_indicators
from model import load_model, FEATURES, create_labels


def run_backtest(ticker, starting_capital=10000.0):
    """
    Simulate trading based on ML predictions
    on historical data we already have.

    Rules:
    - Start with $10,000
    - If model predicts UP  → buy stock at open next day
    - If model predicts DOWN → sell / stay in cash
    - Track portfolio value every day
    - Compare against just holding the stock (buy and hold)
    """
    print(f"\nRunning backtest for {ticker}...")
    print(f"Starting capital: ${starting_capital:,.2f}")

    # Load and prepare data
    df = load_from_database(ticker)
    df = add_all_indicators(df)
    df = create_labels(df)
    df = df.reset_index(drop=True)

    model = load_model()

    # ------------------------------------------------
    # SIMULATE DAY BY DAY
    # ------------------------------------------------
    capital      = starting_capital
    shares_held  = 0
    in_market    = False
    trades       = []
    portfolio_values = []

    # We start from row 1 so we always have
    # yesterday's data to make today's decision
    for i in range(1, len(df)):

        today     = df.iloc[i]
        yesterday = df.iloc[i - 1]

        # Make prediction using yesterday's indicators
        # (we can't use today's — that would be cheating)
        features  = pd.DataFrame([yesterday[FEATURES]])
        prediction = model.predict(features)[0]

        price = today["close"]
        date  = today["date"]

        # ---- BUY SIGNAL ----
        if prediction == 1 and not in_market:
            shares_held = capital / price
            capital     = 0
            in_market   = True
            trades.append({
                "date":   date,
                "action": "BUY",
                "price":  round(price, 2),
                "shares": round(shares_held, 4)
            })

        # ---- SELL SIGNAL ----
        elif prediction == 0 and in_market:
            capital     = shares_held * price
            shares_held = 0
            in_market   = False
            trades.append({
                "date":   date,
                "action": "SELL",
                "price":  round(price, 2),
                "value":  round(capital, 2)
            })

        # ---- TRACK PORTFOLIO VALUE ----
        if in_market:
            portfolio_value = shares_held * price
        else:
            portfolio_value = capital

        portfolio_values.append({
            "date":  date,
            "value": round(portfolio_value, 2)
        })

    # Close any open position at end
    if in_market:
        final_price = df.iloc[-1]["close"]
        capital     = shares_held * final_price
        in_market   = False

    # ------------------------------------------------
    # BUY AND HOLD COMPARISON
    # How much would you have made just holding?
    # ------------------------------------------------
    start_price    = df.iloc[1]["close"]
    end_price      = df.iloc[-1]["close"]
    buy_hold_value = starting_capital * (end_price / start_price)

    # ------------------------------------------------
    # CALCULATE RESULTS
    # ------------------------------------------------
    final_value    = capital
    total_return   = ((final_value - starting_capital)
                      / starting_capital * 100)
    buy_hold_return = ((buy_hold_value - starting_capital)
                       / starting_capital * 100)

    num_trades = len(trades)
    buy_trades  = [t for t in trades if t["action"] == "BUY"]
    sell_trades = [t for t in trades if t["action"] == "SELL"]

    # Calculate win rate — how many trades were profitable
    wins = 0
    for j in range(min(len(buy_trades), len(sell_trades))):
        if sell_trades[j]["price"] > buy_trades[j]["price"]:
            wins += 1

    completed_trades = min(len(buy_trades), len(sell_trades))
    win_rate = (wins / completed_trades * 100) if completed_trades > 0 else 0

    # ------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------
    print(f"\n{'='*45}")
    print(f"  BACKTEST RESULTS — {ticker}")
    print(f"{'='*45}")
    print(f"  Period          : {df.iloc[1]['date']} → {df.iloc[-1]['date']}")
    print(f"  Starting capital: ${starting_capital:>10,.2f}")
    print(f"  Final value     : ${final_value:>10,.2f}")
    print(f"  Strategy return : {total_return:>+.1f}%")
    print(f"  Buy & hold      : {buy_hold_return:>+.1f}%")
    print(f"  Outperformed    : {'YES ✓' if total_return > buy_hold_return else 'NO ✗'}")
    print(f"{'='*45}")
    print(f"  Total trades    : {num_trades}")
    print(f"  Completed trades: {completed_trades}")
    print(f"  Win rate        : {win_rate:.1f}%")
    print(f"{'='*45}")

    print(f"\nTrade log:")
    for trade in trades[:10]:
        print(f"  {trade['date']}  {trade['action']}  "
              f"@ ${trade['price']}")
    if len(trades) > 10:
        print(f"  ... and {len(trades)-10} more trades")

    return {
        "ticker":           ticker,
        "starting_capital": starting_capital,
        "final_value":      round(final_value, 2),
        "strategy_return":  round(total_return, 2),
        "buy_hold_return":  round(buy_hold_return, 2),
        "outperformed":     total_return > buy_hold_return,
        "total_trades":     num_trades,
        "win_rate":         round(win_rate, 1),
        "portfolio_values": portfolio_values,
        "trades":           trades
    }


def run_all_backtests():
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    results = []

    for ticker in tickers:
        result = run_backtest(ticker)
        results.append(result)

    # Summary table
    print(f"\n{'='*55}")
    print(f"  SUMMARY — ALL STOCKS")
    print(f"{'='*55}")
    print(f"  {'Ticker':<8} {'Return':>8} {'B&H':>8} "
          f"{'Beat?':>6} {'Win Rate':>10}")
    print(f"  {'-'*45}")

    for r in results:
        beat = "YES ✓" if r["outperformed"] else "NO  ✗"
        print(f"  {r['ticker']:<8} "
              f"{r['strategy_return']:>+7.1f}% "
              f"{r['buy_hold_return']:>+7.1f}% "
              f"{beat:>6}  "
              f"{r['win_rate']:>8.1f}%")

    print(f"{'='*55}")
    return results


if __name__ == "__main__":
    run_all_backtests()