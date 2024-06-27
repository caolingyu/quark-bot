import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import pywt

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
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 'Head and Shoulder'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = 'Inverse Head and Shoulder'
    return df['head_shoulder_pattern'].dropna().tolist()

def wavelet_denoise(series, wavelet='db1', level=1):
    coeff = pywt.wavedec(series, wavelet, mode="per")
    for i in range(1, len(coeff)):
        coeff[i] = pywt.threshold(coeff[i], value=np.std(coeff[i])/2, mode="soft")
    return pywt.waverec(coeff, wavelet, mode="per")

def detect_head_shoulder_wavelet(df, window=3):
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
    print(df['head_shoulder_pattern'].tolist())
    return latest_pattern
    
def detect_candlestick_patterns(df):
    patterns = []

    if is_morning_star(df):
        patterns.append('启明星')
    if is_evening_star(df):
        patterns.append('黄昏星')
    if is_doji_star(df):
        patterns.append('十字星')

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

    head_shoulder_wavelet = detect_head_shoulder_wavelet(df)
    if not pd.isna(head_shoulder_wavelet):
        patterns.append(head_shoulder_wavelet)

    return patterns