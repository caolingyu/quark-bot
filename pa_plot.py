import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pa.pa_index import pa_oscillator, check_zones
from pa.vector_candles import vector_candles
from pa.sup_res_ulti import pivot_low, pivot_high, calculate_importance

def is_bullish_pattern(df, index, pattern_type):
    if index < 1:
        return False
    prev_candle = df.iloc[index - 1]
    current_candle = df.iloc[index]
    
    if prev_candle['Close'] >= prev_candle['Open'] or current_candle['Close'] <= current_candle['Open']:
        return False
    
    if pattern_type == 'engulfing':
        return current_candle['Open'] <= prev_candle['Low'] and current_candle['Close'] >= prev_candle['High']
    elif pattern_type == 'harami':
        return (current_candle['Open'] <= prev_candle['Close'] and
                current_candle['Close'] >= prev_candle['Open'] and
                current_candle['High'] <= prev_candle['High'] and
                current_candle['Low'] >= prev_candle['Low'])
    return False

def find_pivot_points(df, window=10):
    highs = df['High'].rolling(window=window, center=True).apply(lambda x: x.argmax() == window // 2)
    lows = df['Low'].rolling(window=window, center=True).apply(lambda x: x.argmin() == window // 2)
    return highs.fillna(False).astype(bool), lows.fillna(False).astype(bool)

def add_trendlines(df, fig, window=10):
    highs, lows = find_pivot_points(df, window)
    
    for points, y_col, color, name in [
        (df.loc[highs], 'High', 'blue', 'Upper Trendline'),
        (df.loc[lows], 'Low', 'red', 'Lower Trendline')
    ]:
        if not points.empty:
            fig.add_trace(go.Scatter(
                x=points['timestamp'],
                y=points[y_col],
                mode='lines+markers',
                line=dict(color=color, width=1),
                marker=dict(size=4, color=color),
                name=name
            ), row=1, col=1)

def execute_trade(df, i, capital, risk_per_trade, tick_size, dca_levels):
    signal_candle = df.iloc[i]
    next_candle = df.iloc[i + 1]
    entry_price = signal_candle['High'] + tick_size
    
    if next_candle['High'] >= entry_price:
        stop_loss = signal_candle['Low'] - tick_size
        target_price = entry_price + 2 * (entry_price - stop_loss)
        
        risk_amount = capital * risk_per_trade
        position_size = risk_amount / (entry_price - stop_loss)
        avg_entry_price = entry_price
        
        buy_signals = [(next_candle['timestamp'], entry_price)]
        sell_signals = []
        dca_count = 0
        
        for j in range(i + 2, len(df)):
            current_price = df.iloc[j]['Close']
            
            for level in dca_levels:
                if current_price <= avg_entry_price * (1 - level) and dca_count < len(dca_levels):
                    dca_amount = capital * risk_per_trade * (1 + dca_count * 0.5)
                    dca_size = dca_amount / current_price
                    position_size += dca_size
                    avg_entry_price = (avg_entry_price * position_size + current_price * dca_size) / (position_size + dca_size)
                    dca_count += 1
                    buy_signals.append((df.iloc[j]['timestamp'], current_price))
            
            if df.iloc[j]['Low'] <= stop_loss:
                sell_signals.append((df.iloc[j]['timestamp'], stop_loss))
                return (stop_loss - avg_entry_price) * position_size, buy_signals, sell_signals, False
            if df.iloc[j]['High'] >= target_price:
                sell_signals.append((df.iloc[j]['timestamp'], target_price))
                return (target_price - avg_entry_price) * position_size, buy_signals, sell_signals, True
        
        exit_price = df.iloc[-1]['Close']
        sell_signals.append((df.iloc[-1]['timestamp'], exit_price))
        return (exit_price - avg_entry_price) * position_size, buy_signals, sell_signals, False
    
    return 0, [], [], False

def backtest_and_plot(df, initial_capital=10000, risk_per_trade=0.02, dca_levels=[0.05, 0.1]):
    tick_size = 0.01
    successful_trades = total_trades = total_profit_loss = 0
    buy_signals = sell_signals = []
    capital = initial_capital

    df = vector_candles(df)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df = pa_oscillator(df)
    df = check_zones(df)

    for i in range(1, len(df) - 1):
        if (is_bullish_pattern(df, i, 'engulfing') or is_bullish_pattern(df, i, 'harami')) and df.iloc[i]['PA'] < -10:
            total_trades += 1
            trade_profit, new_buy_signals, new_sell_signals, is_successful = execute_trade(df, i, capital, risk_per_trade, tick_size, dca_levels)
            
            total_profit_loss += trade_profit
            capital += trade_profit
            buy_signals.extend(new_buy_signals)
            sell_signals.extend(new_sell_signals)
            if is_successful:
                successful_trades += 1

    # Calculate support and resistance levels
    for source in ['Close', 'Low', 'High']:
        df[f'pivot_support_{source.lower()}'] = pivot_low(df[source], 20, 10)
        df[f'pivot_resistance_{source.lower()}'] = pivot_high(df[source], 20, 10)
    df['quick_pivot_support'] = pivot_low(df['Low'], 20, 5)
    df['quick_pivot_resistance'] = pivot_high(df['High'], 20, 5)

    levels = []
    for support, resistance in [('quick_pivot_support', 'quick_pivot_resistance'), ('pivot_support_low', 'pivot_resistance_high')] * 2:
        for level_type, column in [('support', support), ('resistance', resistance)]:
            levels.extend([(level, calculate_importance(3, 10), level_type) for level in df[df[column].notna()][column].unique() if not np.isnan(level)])

    # Prepare data for plotting
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    buy_df = pd.DataFrame(buy_signals, columns=['timestamp', 'price']).set_index('timestamp')
    sell_df = pd.DataFrame(sell_signals, columns=['timestamp', 'price']).set_index('timestamp')

    # Create the plot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

    color_map = {
        ('up', 1): 'lime', ('down', 1): 'red',
        ('up', 2): 'blue', ('down', 2): 'fuchsia',
        ('up', 0): 'gray', ('down', 0): 'darkgray'
    }
    colors_kline = [color_map[(row['candle_direction'], row['va'])] for _, row in df.iterrows()]

    # Plot candlesticks
    for i in range(len(df)):
        fig.add_trace(go.Scatter(x=[df.index[i], df.index[i]], y=[df['Low'].iloc[i], df['High'].iloc[i]], mode='lines', line=dict(color=colors_kline[i], width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=[df.index[i]], y=[abs(df['Close'].iloc[i] - df['Open'].iloc[i])], base=min(df['Open'].iloc[i], df['Close'].iloc[i]), marker_color=colors_kline[i], marker_line_width=0, width=0.6, showlegend=False), row=1, col=1)

    # Add buy and sell signals
    fig.add_trace(go.Scatter(x=buy_df.index, y=buy_df['price'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_df.index, y=sell_df['price'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell'), row=1, col=1)

    # Add PA oscillator
    fig.add_trace(go.Scatter(x=df.index, y=df['PA'], mode='lines', name='PA Oscillator'), row=2, col=1)
    for y in [80, -80, 5, -5]:
        fig.add_hline(y=y, line_dash="dash", line_color="gray", row=2, col=1)

    # Update layout
    fig.update_layout(title='BTC/USDT Candlestick Chart with Signals and PA Oscillator', xaxis_title='Time', yaxis_title='Price', xaxis_rangeslider_visible=False)
    fig.update_xaxes(showticklabels=False, type='category', categoryorder='category ascending')

    df = df.reset_index()
    add_trendlines(df, fig)
    fig.update_layout(xaxis=dict(type='category', categoryorder='category ascending', tickmode='auto', nticks=20))

    fig.show()

    # Print results
    success_rate = successful_trades / total_trades if total_trades > 0 else 0
    print(f"Success Rate: {success_rate:.2%} ({successful_trades}/{total_trades})")
    print(f"Total Profit/Loss: {total_profit_loss:.2f} USDT")
    print(f"Final Capital: {capital:.2f} USDT")
    print(f"Total Return: {(capital - initial_capital) / initial_capital:.2%}")

# Main execution
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '5m'
limit = 1000
ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
backtest_and_plot(df)