import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import pywt
import talib

def is_morning_star(df):
    if len(df) < 3:
        return False

    first_candle = df.iloc[-3]
    second_candle = df.iloc[-2]
    third_candle = df.iloc[-1]

    if (first_candle['Close'] < first_candle['Open'] and
        second_candle['Close'] < second_candle['Open'] and
        third_candle['Close'] > third_candle['Open'] and
        second_candle['Low'] < first_candle['Low'] and
        second_candle['Low'] < third_candle['Low'] and
        third_candle['Close'] > first_candle['Close']):
        return True
    
    return False

def is_evening_star(df):
    if len(df) < 3:
        return False

    first_candle = df.iloc[-3]
    second_candle = df.iloc[-2]
    third_candle = df.iloc[-1]

    if (first_candle['Close'] > first_candle['Open'] and
        second_candle['Close'] > second_candle['Open'] and
        third_candle['Close'] < third_candle['Open'] and
        second_candle['High'] > first_candle['High'] and
        second_candle['High'] > third_candle['High'] and
        third_candle['Close'] < first_candle['Close']):
        return True

    return False

def is_doji_star(df):
    if len(df) < 1:
        return False
    
    # 检查形态的基本条件（开盘价和收盘价的关系）
    latest_candle = df.iloc[-1]

    return abs(latest_candle['Close'] - latest_candle['Open']) < (0.1 * (latest_candle['High'] - latest_candle['Low']))


def is_hammer(df):
    if len(df) < 1:
        return False
    
    latest_candle = df.iloc[-1]
    
    # 锤头线特征：小实体位于上端，下影线长度至少为实体长度的两倍，无或很小的上影线
    body_length = abs(latest_candle['Close'] - latest_candle['Open'])
    lower_shadow = latest_candle['Low'] - min(latest_candle['Close'], latest_candle['Open'])
    upper_shadow = max(latest_candle['Close'], latest_candle['Open']) - latest_candle['High']
    
    return (lower_shadow >= 2 * body_length) and (upper_shadow < 0.1 * (latest_candle['High'] - latest_candle['Low']))


def is_shooting_star(df):
    if len(df) < 1:
        return False
    
    latest_candle = df.iloc[-1]
    
    # 流星线特征：小实体位于下端，上影线长度至少为实体长度的两倍，无或很小的下影线
    body_length = abs(latest_candle['Close'] - latest_candle['Open'])
    upper_shadow = latest_candle['High'] - max(latest_candle['Close'], latest_candle['Open'])
    lower_shadow = min(latest_candle['Close'], latest_candle['Open']) - latest_candle['Low']
    
    return (upper_shadow >= 2 * body_length) and (lower_shadow < 0.1 * (latest_candle['High'] - latest_candle['Low']))


def is_bullish_engulfing(df):
    if len(df) < 2:
        return False
    
    first_candle = df.iloc[-2]
    second_candle = df.iloc[-1]
    
    # 看涨吞没特征：第二根阳线的实体完全覆盖第一根阴线的实体
    return (first_candle['Close'] < first_candle['Open']) and (second_candle['Close'] > second_candle['Open']) and \
           (second_candle['Open'] < first_candle['Close']) and (second_candle['Close'] > first_candle['Open'])


def is_bearish_engulfing(df):
    if len(df) < 2:
        return False
    
    first_candle = df.iloc[-2]
    second_candle = df.iloc[-1]
    
    # 看跌吞没特征：第二根阴线的实体完全覆盖第一根阳线的实体
    return (first_candle['Close'] > first_candle['Open']) and (second_candle['Close'] < second_candle['Open']) and \
           (second_candle['Open'] > first_candle['Close']) and (second_candle['Close'] < first_candle['Open'])

def detect_head_shoulder(df, window=3):
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    mask_head_shoulder = ((df['high_roll_max'] > df['High'].shift(1)) & (df['high_roll_max'] > df['High'].shift(-1)) & (df['High'] < df['High'].shift(1)) & (df['High'] < df['High'].shift(-1)))
    mask_inv_head_shoulder = ((df['low_roll_min'] < df['Low'].shift(1)) & (df['low_roll_min'] < df['Low'].shift(-1)) & (df['Low'] > df['Low'].shift(1)) & (df['Low'] > df['Low'].shift(-1)))
    df['head_shoulder_pattern'] = np.nan
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = '头肩顶'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = '头肩底'
    # 检查当前最新一根K线的pattern
    latest_pattern = df['head_shoulder_pattern'].iloc[-1]
    
    return latest_pattern

def detect_multiple_tops_bottoms(df, window=3):
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['close_roll_max'] = df['Close'].rolling(window=roll_window).max()
    df['close_roll_min'] = df['Close'].rolling(window=roll_window).min()
    mask_top = (df['high_roll_max'] >= df['High'].shift(1)) & (df['close_roll_max'] < df['Close'].shift(1))
    mask_bottom = (df['low_roll_min'] <= df['Low'].shift(1)) & (df['close_roll_min'] > df['Close'].shift(1))
    df['multiple_top_bottom_pattern'] = np.nan
    df.loc[mask_top, 'multiple_top_bottom_pattern'] = '多重顶'
    df.loc[mask_bottom, 'multiple_top_bottom_pattern'] = '多重底'
    # 检查当前最新一根K线的pattern
    latest_pattern = df['multiple_top_bottom_pattern'].iloc[-1]
    
    return latest_pattern

def detect_triangle_pattern(df, window=3):
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    mask_asc = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['Close'] > df['Close'].shift(1))
    mask_desc = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['Close'] < df['Close'].shift(1))
    df['triangle_pattern'] = np.nan
    df.loc[mask_asc, 'triangle_pattern'] = '上升三角'
    df.loc[mask_desc, 'triangle_pattern'] = '下降三角'
    # 检查当前最新一根K线的pattern
    latest_pattern = df['triangle_pattern'].iloc[-1]
    
    return latest_pattern

def detect_wedge(df, window=3):
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['trend_high'] = df['High'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    df['trend_low'] = df['Low'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    mask_wedge_up = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['trend_high'] == 1) & (df['trend_low'] == 1)
    mask_wedge_down = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['trend_high'] == -1) & (df['trend_low'] == -1)
    df['wedge_pattern'] = np.nan
    df.loc[mask_wedge_up, 'wedge_pattern'] = '上升楔形'
    df.loc[mask_wedge_down, 'wedge_pattern'] = '下降楔形'
    # 检查当前最新一根K线的pattern
    latest_pattern = df['wedge_pattern'].iloc[-1]
    
    return latest_pattern

def detect_channel(df, window=3):
    roll_window = window
    channel_range = 0.1
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['trend_high'] = df['High'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    df['trend_low'] = df['Low'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    mask_channel_up = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['high_roll_max'] - df['low_roll_min'] <= channel_range * (df['high_roll_max'] + df['low_roll_min'])/2) & (df['trend_high'] == 1) & (df['trend_low'] == 1)
    mask_channel_down = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['high_roll_max'] - df['low_roll_min'] <= channel_range * (df['high_roll_max'] + df['low_roll_min'])/2) & (df['trend_high'] == -1) & (df['trend_low'] == -1)
    df['channel_pattern'] = np.nan
    df.loc[mask_channel_up, 'channel_pattern'] = '上升通道'
    df.loc[mask_channel_down, 'channel_pattern'] = '下降通道'
    # 检查当前最新一根K线的pattern
    latest_pattern = df['channel_pattern'].iloc[-1]
    
    return latest_pattern

def detect_double_top_bottom(df, window=3, threshold=0.05):
    roll_window = window
    range_threshold = threshold
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    mask_double_top = (df['high_roll_max'] >= df['High'].shift(1)) & (df['high_roll_max'] >= df['High'].shift(-1)) & (df['High'] < df['High'].shift(1)) & (df['High'] < df['High'].shift(-1)) & ((df['High'].shift(1) - df['Low'].shift(1)) <= range_threshold * (df['High'].shift(1) + df['Low'].shift(1))/2) & ((df['High'].shift(-1) - df['Low'].shift(-1)) <= range_threshold * (df['High'].shift(-1) + df['Low'].shift(-1))/2)
    mask_double_bottom = (df['low_roll_min'] <= df['Low'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(-1)) & (df['Low'] > df['Low'].shift(1)) & (df['Low'] > df['Low'].shift(-1)) & ((df['High'].shift(1) - df['Low'].shift(1)) <= range_threshold * (df['High'].shift(1) + df['Low'].shift(1))/2) & ((df['High'].shift(-1) - df['Low'].shift(-1)) <= range_threshold * (df['High'].shift(-1) + df['Low'].shift(-1))/2)
    df['double_pattern'] = np.nan
    df.loc[mask_double_top, 'double_pattern'] = '双顶'
    df.loc[mask_double_bottom, 'double_pattern'] = '双底'
    # 检查当前最新一根K线的pattern
    latest_pattern = df['double_pattern'].iloc[-1]
    
    return latest_pattern

def detect_trendline(df, window=2):
    roll_window = window
    df['slope'] = np.nan
    df['intercept'] = np.nan
    for i in range(window, len(df)):
        x = np.array(range(i-window, i))
        y = df['Close'][i-window:i]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        df.at[df.index[i], 'slope'] = m
        df.at[df.index[i], 'intercept'] = c
    mask_support = df['slope'] > 0
    mask_resistance = df['slope'] < 0
    df['support'] = np.nan
    df['resistance'] = np.nan
    df.loc[mask_support, 'support'] = df['Close'] * df['slope'] + df['intercept']
    df.loc[mask_resistance, 'resistance'] = df['Close'] * df['slope'] + df['intercept']
    return df[['support', 'resistance']]

def find_pivots(df):
    high_diffs = df['High'].diff()
    low_diffs = df['Low'].diff()
    higher_high_mask = (high_diffs > 0) & (high_diffs.shift(-1) < 0)
    lower_low_mask = (low_diffs < 0) & (low_diffs.shift(-1) > 0)
    lower_high_mask = (high_diffs < 0) & (high_diffs.shift(-1) > 0)
    higher_low_mask = (low_diffs > 0) & (low_diffs.shift(-1) < 0)
    df['signal'] = ''
    df.loc[higher_high_mask, 'signal'] = 'HH'
    df.loc[lower_low_mask, 'signal'] = 'LL'
    df.loc[lower_high_mask, 'signal'] = 'LH'
    df.loc[higher_low_mask, 'signal'] = 'HL'
    return df['signal'].dropna().tolist()

def detect_head_shoulder_filter(df, window=3, threshold=0.01, time_delay=1):
    roll_window = window
    df['High_smooth'] = savgol_filter(df['High'], roll_window, 2)
    df['Low_smooth'] = savgol_filter(df['Low'], roll_window, 2)
    df['high_roll_max'] = df['High_smooth'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low_smooth'].rolling(window=roll_window).min()
    df['head_height'] = df['high_roll_max'] - df['Low'].rolling(window=roll_window).min()
    df['inv_head_height'] = df['High'].rolling(window=roll_window).max() - df['low_roll_min']
    mask_head_shoulder = ((df['head_height'] > threshold) & (df['high_roll_max'] > df['High_smooth'].shift(time_delay)) & (df['high_roll_max'] > df['High_smooth'].shift(-time_delay)) & (df['High_smooth'] < df['High_smooth'].shift(time_delay)) & (df['High_smooth'] < df['High_smooth'].shift(-time_delay)))
    mask_inv_head_shoulder = ((df['inv_head_height'] > threshold) & (df['low_roll_min'] < df['Low_smooth'].shift(time_delay)) & (df['low_roll_min'] < df['Low_smooth'].shift(-time_delay)) & (df['Low_smooth'] > df['Low_smooth'].shift(time_delay)) & (df['Low_smooth'] > df['Low_smooth'].shift(-time_delay)))
    df['head_shoulder_pattern'] = np.nan
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 'Head and Shoulder'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = 'Inverse Head and Shoulder'
    return df['head_shoulder_pattern'].dropna().tolist()

def kalman_smooth(series, n_iter=10):
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    kf = kf.em(series, n_iter=n_iter)
    state_means, _ = kf.filter(series.values)
    return state_means.flatten()

def detect_head_shoulder_kf(df, window=3):
    roll_window = window
    df['High_smooth'] = kalman_smooth(df['High'])
    df['Low_smooth'] = kalman_smooth(df['Low'])
    df['high_roll_max'] = df['High_smooth'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low_smooth'].rolling(window=roll_window).min()
    mask_head_shoulder = ((df['high_roll_max'] > df['High_smooth'].shift(1)) & (df['high_roll_max'] > df['High_smooth'].shift(-1)) & (df['High_smooth'] < df['High_smooth'].shift(1)) & (df['High_smooth'] < df['High_smooth'].shift(-1)))
    mask_inv_head_shoulder = ((df['low_roll_min'] < df['Low_smooth'].shift(1)) & (df['low_roll_min'] < df['Low_smooth'].shift(-1)) & (df['Low_smooth'] > df['Low_smooth'].shift(1)) & (df['Low_smooth'] > df['Low_smooth'].shift(-1)))
    df['head_shoulder_pattern'] = np.nan
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = '头肩顶'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = '头肩底'
    latest_pattern = df['head_shoulder_pattern'].iloc[-1]
    return latest_pattern

def wavelet_denoise(series, wavelet='db1', level=1):
    coeff = pywt.wavedec(series, wavelet, mode="per")
    for i in range(1, len(coeff)):
        coeff[i] = pywt.threshold(coeff[i], value=np.std(coeff[i])/2, mode="soft")
    return pywt.waverec(coeff, wavelet, mode="per")

def detect_head_shoulder_wavelet(df, window=3):
    print(df)
    roll_window = window
    df['High_smooth'] = wavelet_denoise(df['High'], 'db1', level=1)
    df['Low_smooth'] = wavelet_denoise(df['Low'], 'db1', level=1)
    df['high_roll_max'] = df['High_smooth'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low_smooth'].rolling(window=roll_window).min()
    mask_head_shoulder = ((df['high_roll_max'] > df['High_smooth'].shift(1)) & (df['high_roll_max'] > df['High_smooth'].shift(-1)) & (df['High_smooth'] < df['High_smooth'].shift(1)) & (df['High_smooth'] < df['High_smooth'].shift(-1)))
    mask_inv_head_shoulder = ((df['low_roll_min'] < df['Low_smooth'].shift(1)) & (df['low_roll_min'] < df['Low_smooth'].shift(-1)) & (df['Low_smooth'] > df['Low_smooth'].shift(1)) & (df['Low_smooth'] > df['Low_smooth'].shift(-1)))
    df['head_shoulder_pattern'] = np.nan
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = '头肩顶'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = '头肩底'
    # 检查当前最新一根K线的pattern
    latest_pattern = df['head_shoulder_pattern'].iloc[-1]
    return latest_pattern

def fibonacci_retracement_levels(df):
    max_price = df['High'].max()
    min_price = df['Low'].min()
    diff = max_price - min_price
    levels = {
        '23.6%': max_price - 0.236 * diff,
        '38.2%': max_price - 0.382 * diff,
        '50%': max_price - 0.5 * diff,
        '61.8%': max_price - 0.618 * diff,
        '100%': min_price
    }
    return levels

def detect_fibonacci_level(df):
    levels = fibonacci_retracement_levels(df)
    current_price = df['Close'].iloc[-1]
    for level, value in levels.items():
        if current_price >= value:
            return f"当前价格位于斐波那契回调{level}以上."
    return "当前价格不处于斐波那契回调"

def analyze_latest_moving_averages(df):
    """
    分析股票收盘价的指数移动平均线(EMA)状态，特别关注最新一根K线。

    参数:
    df (DataFrame): 包含至少'Close'列的DataFrame，表示股票收盘价。

    返回:
    dict: 包含最新K线的EMA分析结果，包括趋势状态、交叉情况、均线密集度和价格与均线组的平均距离。
    """
    
    # 确保DataFrame非空并获取最后一行数据
    if df.empty:
        return {}
    
    # 计算EMA
    df['EMA20'] = talib.EMA(df['Close'], timeperiod=20)
    df['EMA60'] = talib.EMA(df['Close'], timeperiod=60)
    df['EMA120'] = talib.EMA(df['Close'], timeperiod=120)
    latest_row = df.iloc[-1]

    ema20 = talib.EMA(df['Close'], timeperiod=20).iloc[-1]
    ema60 = talib.EMA(df['Close'], timeperiod=60).iloc[-1]
    ema120 = talib.EMA(df['Close'], timeperiod=120).iloc[-1]
    
    # 分析结果初始化
    analysis_results = {}
    
    # 多头排列或空头排列判断
    if ema20 > ema60 > ema120:
        analysis_results['Trend'] = '多头排列'
    elif ema20 < ema60 < ema120:
        analysis_results['Trend'] = '空头排列'
    else:
        analysis_results['Trend'] = '中性'
    
    # 金叉或死叉检测
    if latest_row['EMA20'] > latest_row['EMA60'] and df.iloc[-2]['EMA20'] < df.iloc[-2]['EMA60']:
        analysis_results['Crossover'] = '金叉'
    elif latest_row['EMA20'] < latest_row['EMA60'] and df.iloc[-2]['EMA20'] > df.iloc[-2]['EMA60']:
        analysis_results['Crossover'] = '死叉'
    else:
        analysis_results['Crossover'] = '无交叉'
    
    # 均线密集程度评估
    MA_DENSITY_THRESHOLD_NEAR = 0.02  # 近密阈值
    MA_DENSITY_THRESHOLD_MEDIUM = 0.05  # 中密阈值
    if abs(ema20 - ema120) / latest_row['Close'] <= MA_DENSITY_THRESHOLD_NEAR:
        analysis_results['Density'] = '非常密集'
    elif abs(ema20 - ema120) / latest_row['Close'] <= MA_DENSITY_THRESHOLD_MEDIUM:
        analysis_results['Density'] = '较密集'
    else:
        analysis_results['Density'] = '分散'
    
    # 当前价格与均线组的平均距离
    PRICE_DISTANCE_THRESHOLD_NEAR = 0.01  # 接近阈值
    PRICE_DISTANCE_THRESHOLD_MEDIUM = 0.03  # 中等阈值
    price_distance_avg = (abs(latest_row['Close'] - ema20) + 
                        abs(latest_row['Close'] - ema60) + 
                        abs(latest_row['Close'] - ema120)) / 3 / latest_row['Close']
    if price_distance_avg <= PRICE_DISTANCE_THRESHOLD_NEAR:
        analysis_results['Distance_to_MA_Avg'] = '非常接近'
    elif price_distance_avg <= PRICE_DISTANCE_THRESHOLD_MEDIUM:
        analysis_results['Distance_to_MA_Avg'] = '接近'
    else:
        analysis_results['Distance_to_MA_Avg'] = '远离'
        
    return analysis_results

def detect_candlestick_patterns(df):
    patterns = []

    if is_morning_star(df):
        patterns.append('启明星')
    if is_evening_star(df):
        patterns.append('黄昏星')
    if is_doji_star(df):
        patterns.append('十字星')
    if is_hammer(df):
        patterns.append('看涨pinbar')
    if is_shooting_star(df):
        patterns.append('看跌pinbar')
    if is_bullish_engulfing(df):
        patterns.append('看涨吞没')
    if is_bearish_engulfing(df):
        patterns.append('看跌吞没')

    multiple_tops_bottoms = detect_multiple_tops_bottoms(df)
    if not pd.isna(multiple_tops_bottoms):
        patterns.append(multiple_tops_bottoms)

    triangle_pattern = detect_triangle_pattern(df)
    if not pd.isna(triangle_pattern):
        patterns.append(triangle_pattern)

    wedge_pattern = detect_wedge(df)
    if not pd.isna(wedge_pattern):
        patterns.append(wedge_pattern)

    channel_pattern = detect_channel(df)
    if not pd.isna(channel_pattern):
        patterns.append(channel_pattern)

    double_top_bottom = detect_double_top_bottom(df)
    if not pd.isna(double_top_bottom):
        patterns.append(double_top_bottom)

    head_shoulder_kf = detect_head_shoulder_kf(df)
    if not pd.isna(head_shoulder_kf):
        patterns.append(head_shoulder_kf)

    return patterns