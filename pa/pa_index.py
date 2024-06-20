import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf


# 初始化 binance 交易所
binance = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True,
    'proxies': {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890',
    },
})

# 获取交易对的K线数据
def fetch_ohlcv(symbol, timeframe='1d', limit=100):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.rename(columns=lambda x: x.capitalize(), inplace=True)
    df.set_index('Timestamp', inplace=True)  # 确保时间戳为索引，并且是 DatetimeIndex
    df['Symbol'] = symbol  # 添加symbol列
    return df


# 计算K线数据的随机震荡指标
def stoch(close, high, low, length):
    L = low.rolling(window=length).min()
    H = high.rolling(window=length).max()
    K = 100 * (close - L) / (H - L)
    return K

# 使用移动平均线平滑数据
def sma(series, length):
    return series.rolling(window=length).mean()

# 计算标准差
def stdev(series, length):
    return series.rolling(window=length).std()

# 离散方法实现
def dispersion_method(series, method, length):
    if method == "标准差":
        return stdev(series, length)
    elif method == "方差":
        return series.rolling(window=length).var()
    elif method == "变异系数":
        return stdev(series, length) / sma(series, length)
    elif method == "信噪比":
        return sma(series, length) / stdev(series, length)
    elif method == "信噪比²":
        return (sma(series, length) ** 2) / (stdev(series, length) ** 2)
    elif method == "离散系数":
        return (series.rolling(window=length).var() ** 2) / sma(series, length)
    elif method == "效率":
        return (stdev(series, length) ** 2) / (sma(series, length) ** 2)
    # elif method == "高低范围":
    #     return high.rolling(window=length).max() - low.rolling(window=length).min()
    else:
        raise ValueError("Invalid dispersion method")

# 计算价格行动指数（PA振荡器）
def pa_oscillator(df, stoch_len=20, smooth_len=3, dispersion_len=20, dispersion_method_name="标准差"):
    df['K'] = stoch(df['Close'], df['High'], df['Low'], stoch_len)
    df['sK'] = sma(df['K'], smooth_len)
    df['Dispersion'] = dispersion_method(df['Close'], dispersion_method_name, dispersion_len)
    
    P = (df['sK'] - 50) / 50.0
    V = stoch(df['Dispersion'], df['Dispersion'], df['Dispersion'], dispersion_len)
    df['PA'] = P * V
    
    return df

# 判断价格是否处于支撑区、超买区或超卖区
def check_zones(df, straddle_area=5.0):
    df['Zone'] = 'Neutral'
    df.loc[df['PA'] < -80, 'Zone'] = 'Oversold'
    df.loc[df['PA'] > 80, 'Zone'] = 'Overbought'
    df.loc[(df['PA'] > -straddle_area) & (df['PA'] < straddle_area), 'Zone'] = 'Straddle'
    df.loc[(df['PA'] >= straddle_area) & (df['PA'] <= 80), 'Zone'] = 'Neutral'
    df.loc[(df['PA'] <= -straddle_area) & (df['PA'] >= -80), 'Zone'] = 'Neutral'
    return df

def detect_pa_index_patterns(df):
    df = pa_oscillator(df)
    df = check_zones(df)
    return df

# 获取 EMA 数据
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

# 绘制K线和EMA线
def plot_with_oscillator(df, symbol, ema_length=20, straddle_area=5.0, filename=None):
    df = df.iloc[30:]
    df['EMA'] = ema(df['Close'], ema_length)

    # 设置市场颜色和风格
    mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)

    # 添加EMA线
    ema_plot = mpf.make_addplot(df['EMA'], color='orange')

    # 添加PA振荡器线，设置为第一个副图
    pa_plot = mpf.make_addplot(df['PA'], panel=1, color='black', ylabel='PA Oscillator')

    # 添加Overbought和Oversold阈值线，设置为第一个副图
    overbought_line = mpf.make_addplot([80]*len(df), panel=1, color='gray', linestyle='--', secondary_y=False)
    oversold_line = mpf.make_addplot([-80]*len(df), panel=1, color='gray', linestyle='--', secondary_y=False)

    # 添加Straddle上下界线，设置为第一个副图
    straddle_upper = mpf.make_addplot([straddle_area]*len(df), panel=1, color='gray', linestyle='--', secondary_y=False)
    straddle_lower = mpf.make_addplot([-straddle_area]*len(df), panel=1, color='gray', linestyle='--', secondary_y=False)

    # 绘制图表，包括K线图和所有添加的线图
    apds = [ema_plot, pa_plot, overbought_line, oversold_line, straddle_upper, straddle_lower]
    fig, axlist = mpf.plot(df, type='candle', style=s, addplot=apds, volume=False, panel_ratios=(6, 2), figsize=(10, 8), title=f'{symbol} Candlestick Chart with EMA and PA Oscillator', returnfig=True)
    
    if filename:
        fig.savefig(filename)
    plt.close(fig)

# 主函数
def main():
    symbol = 'BTC/USDT'
    straddle_area = 5.0
    
    df = fetch_ohlcv(symbol, timeframe='1d', limit=100)
    df = pa_oscillator(df)
    df = check_zones(df, straddle_area)

    plot_with_oscillator(df, symbol, ema_length=20, straddle_area=straddle_area)



# 执行主函数
if __name__ == "__main__":
    main()