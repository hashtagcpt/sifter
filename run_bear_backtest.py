import matplotlib.pyplot as plt
from backend.backtest_engine import run_custom_backtest
import pandas as pd
import json

print("Starting Weekly Backtest over the last 5 years on the full universe...")
print("This includes the 2022 Bear Market.")
res = run_custom_backtest(max_size=250, period_years=5)

if "error" in res:
    print(res["error"])
else:
    trades = res["trades"]
    for t in trades:
        trade_logs = []
        for p in t["portfolio"]:
            trade_logs.append(f"{p['ticker']} (Price: ${p['entry']:.2f})")
        port_ret = t["weekly_return"]
        capital = t["capital"]
        date_str = t["date"]
        print(f"{date_str} | Portfolio: {', '.join(trade_logs)} | 1-Wk Port Ret: {port_ret*100:5.2f}% | Capital: ${capital:.2f}")

    print("\n" + "="*40)
    print("BACKTEST RESULTS (5 Years)")
    print("="*40)
    print(f"Starting Capital : ${res['starting_capital']:.2f}")
    print(f"Ending Capital   : ${res['ending_capital']:.2f}")
    print(f"Total Return     : {res['total_return'] * 100:.2f}%")
    print(f"Ann. Return      : {res['annualised_return'] * 100:.2f}%")
    print(f"Max Drawdown     : {res['max_drawdown'] * 100:.2f}%")
    if res.get('spy_total_return'):
        print(f"SPY Benchmark Ret: {res['spy_total_return'] * 100:.2f}%")
    print("="*40)
    
    eq = res["equity_curve"]
    dates = pd.to_datetime([x["date"] for x in eq])
    capital = [x["capital"] for x in eq]
    spy = [x["spy"] for x in eq]
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, capital, label='Sifter Strategy', color='#4facfe', linewidth=2)
    plt.plot(dates, spy, label='SPY Benchmark', color='#a1a1aa', linestyle='--')
    plt.title('5-Year Backtest (Including 2022 Bear Market)')
    plt.ylabel('Capital ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('static/bear_market_backtest.png', bbox_inches='tight')
    print("Saved chart to static/bear_market_backtest.png")
