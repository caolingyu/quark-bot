import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from mplfinance.original_flavor import candlestick_ohlc

# Initialize the connection to Binance with a proxy
exchange = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True,
    'proxies': {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890',
    },
})

# Function to fetch historical data for a symbol
def fetch_data(symbol, timeframe='1h', since=None, limit=500):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Function to detect wedge patterns and calculate win rate
def detect_wedge_and_calculate_win_rate(df, window=3, future_candles=10, risk_reward_ratio=2):
    # Define the rolling window
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['trend_high'] = df['High'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    df['trend_low'] = df['Low'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    mask_wedge_up = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['trend_high'] == 1) & (df['trend_low'] == 1)
    mask_wedge_down = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['trend_high'] == -1) & (df['trend_low'] == -1)
    df['wedge_pattern'] = np.nan
    df.loc[mask_wedge_up, 'wedge_pattern'] = 'Wedge Up'
    df.loc[mask_wedge_down, 'wedge_pattern'] = 'Wedge Down'
    df['result'] = None
    
    for i in range(len(df) - (future_candles + 1)):
        if df['wedge_pattern'].iloc[i] == 'Wedge Up':
            entry_price = df['Open'].iloc[i + 1]
            max_high_price = df['High'].iloc[i + 1 : i + future_candles + 1].max()
            min_low_price = df['Low'].iloc[i + 1 : i + future_candles + 1].min()
            
            if (max_high_price - entry_price) >= 2 * (entry_price - min_low_price):
                df['result'].iloc[i] = 'Win'
            else:
                df['result'].iloc[i] = 'Loss'

        elif df['wedge_pattern'].iloc[i] == 'Wedge Down':
            entry_price = df['Open'].iloc[i + 1]
            max_high_price = df['High'].iloc[i + 1 : i + future_candles + 1].max()
            min_low_price = df['Low'].iloc[i + 1 : i + future_candles + 1].min()
            
            if (entry_price - min_low_price) >= 2 * (max_high_price - entry_price):
                df['result'].iloc[i] = 'Win'
            else:
                df['result'].iloc[i] = 'Loss'

    win_rate = df['result'].value_counts(normalize=True).get('Win', 0)
    return df, win_rate

# Function to plot wedge patterns on a chart
def plot_wedge(df, symbol):
    plot_df = df[-100:].copy()
    wedge_df = plot_df.dropna(subset=['wedge_pattern'])
    fig, ax = plt.subplots(figsize=(10, 5))
    dates = mdates.date2num(plot_df.index.to_pydatetime())
    ohlc_data = [(dates[i], row[1], row[2], row[3], row[4]) for i, row in enumerate(plot_df.itertuples(index=False))]
    candlestick_ohlc(ax, ohlc_data, width=0.02, colorup='green', colordown='red', alpha=0.8)

    for index, row in wedge_df.iterrows():
        date = mdates.date2num(index.to_pydatetime())
        if row['wedge_pattern'] == 'Wedge Up':
            ax.plot(date, row['High'], '^', color='blue', markersize=10)
        elif row['wedge_pattern'] == 'Wedge Down':
            ax.plot(date, row['Low'], 'v', color='orange', markersize=10)

    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{symbol.replace("/", "_")}_wedge_pattern.png')
    plt.close(fig)

def test():
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']  # Add more symbols as required
    message = "以下是符合 wedge 形态的交易对：\n\n"

    for symbol in symbols:
        try:
            data = fetch_data(symbol)
            wedge_data, wedge_win_rate = detect_wedge_and_calculate_win_rate(data)
            
            if 'Wedge Up' in wedge_data['wedge_pattern'].values or 'Wedge Down' in wedge_data['wedge_pattern'].values:
                plot_wedge(wedge_data, symbol)
                message += f"{symbol} - 胜率: {wedge_win_rate:.2%}\n"
                print(f"图表已保存: {symbol.replace('/', '_')}_wedge_pattern.png")
        except Exception as e:
            message += f"无法处理 {symbol} - 错误: {str(e)}\n"
            print(f"无法处理 {symbol} - 错误: {str(e)}")

    print(message)

# Run the test function
test()