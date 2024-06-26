import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def calculate_recent_support_resistance(df, distance=5):
    """
    使用局部峰值和谷值检测最近的支撑和阻力位
    """

    # 使用 find_peaks 检测支撑和阻力
    peaks, _ = find_peaks(df['High'], distance=distance)
    troughs, _ = find_peaks(-df['Low'], distance=distance)

    if len(troughs) > 0:
        recent_support_level = df.iloc[troughs[-1]]['Low']
    else:
        recent_support_level = None

    if len(peaks) > 0:
        recent_resistance_level = df.iloc[peaks[-1]]['High']
    else:
        recent_resistance_level = None

    return recent_support_level, recent_resistance_level