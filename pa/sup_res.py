import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def calculate_support_resistance(df, window=14, distance=5):
    """
    使用局部峰值和谷值检测支撑和阻力位
    """

    # 使用 rolling min 和 max 计算局部最低和最高
    df['RollingMin'] = df['Close'].rolling(window=window, center=True).min()
    df['RollingMax'] = df['Close'].rolling(window=window, center=True).max()

    # 使用 find_peaks 检测支撑和阻力
    peaks, _ = find_peaks(df['Close'], distance=distance)
    troughs, _ = find_peaks(-df['Close'], distance=distance)

    support_levels = df.iloc[troughs]['Close']
    resistance_levels = df.iloc[peaks]['Close']

    return support_levels, resistance_levels