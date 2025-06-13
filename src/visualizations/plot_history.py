import matplotlib.pyplot as plt
import yfinance as yf
from io import BytesIO
import matplotlib.dates as mdates

def plot_stock_with_prediction(ticker: str, predicted_price: float) -> bytes:
    if not ticker.endswith(".NS"):
        ticker += ".NS"

    df = yf.download(ticker, period="1y", interval="1d")
    if df.empty:
        raise ValueError("No data available for chart.")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], label="Historical Close Price", color='#1f77b4', linewidth=2)

    # Add a horizontal line for the prediction
    ax.axhline(y=predicted_price, color='crimson', linestyle='--', linewidth=2, label=f"Predicted Price: ₹{predicted_price:.2f}")

    # Mark predicted point
    ax.scatter(df.index[-1], predicted_price, color='crimson', edgecolors='black', s=100, zorder=5)
    ax.annotate(f"₹{predicted_price:.2f}", (df.index[-1], predicted_price),
                textcoords="offset points", xytext=(-60,10), ha='center', fontsize=10, color='crimson', weight='bold')

    # Format
    ax.set_title(f"{ticker.upper()} - 1Y Close Price vs ML Prediction", fontsize=16, fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
