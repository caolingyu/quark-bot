import ccxt
import asyncio

import pandas as pd
import numpy as np
from telegram import Bot
from telegram.constants import ParseMode

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

from pa.patterns import *


# 导入形态检测函数
def detect_wedge(df, window=20):
    roll_window = window
    df['high_roll_max'] = df['high'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['low'].rolling(window=roll_window).min()
    
    # 检查滚动窗口内高点趋势
    df['trend_high'] = df['high'].rolling(window=roll_window).apply(
        lambda x: 1 if len(x) == roll_window and x.iloc[-1] > x.iloc[0] else 
                 -1 if len(x) == roll_window and x.iloc[-1] < x.iloc[0] else 0, raw=False
    )
    
    # 检查滚动窗口内低点趋势
    df['trend_low'] = df['low'].rolling(window=roll_window).apply(
        lambda x: 1 if len(x) == roll_window and x.iloc[-1] > x.iloc[0] else 
                 -1 if len(x) == roll_window and x.iloc[-1] < x.iloc[0] else 0, raw=False
    )

    mask_wedge_up = (df['high_roll_max'] >= df['high'].shift(1)) & (df['low_roll_min'] <= df['low'].shift(1)) & (df['trend_high'] == 1) & (df['trend_low'] == 1)
    mask_wedge_down = (df['high_roll_max'] <= df['high'].shift(1)) & (df['low_roll_min'] >= df['low'].shift(1)) & (df['trend_high'] == -1) & (df['trend_low'] == -1)
    
    df['wedge_pattern'] = np.nan
    df.loc[mask_wedge_up, 'wedge_pattern'] = 'Wedge Up'
    df.loc[mask_wedge_down, 'wedge_pattern'] = 'Wedge Down'
    
    return df

def detect_head_shoulder(df, window=3):
    roll_window = window
    df['High_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['Low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    
    # 检查滚动窗口内高点和低点趋势
    mask_head_shoulder = ((df['High_roll_max'] > df['High'].shift(1)) & 
                          (df['High_roll_max'] > df['High'].shift(-1)) & 
                          (df['High'] < df['High'].shift(1)) & 
                          (df['High'] < df['High'].shift(-1))) 

    mask_inv_head_shoulder = ((df['Low_roll_min'] < df['Low'].shift(1)) & 
                              (df['Low_roll_min'] < df['Low'].shift(-1)) & 
                              (df['Low'] > df['Low'].shift(1)) & 
                              (df['Low'] > df['Low'].shift(-1)))
    
    df['head_shoulder_pattern'] = np.nan
    df['inv_head_shoulder_pattern'] = np.nan
    df = df.astype({"head_shoulder_pattern": "object", "inv_head_shoulder_pattern": "object"})  # 显式设置列类型为字符串类型
    
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 'Head and Shoulder'
    df.loc[mask_inv_head_shoulder, 'inv_head_shoulder_pattern'] = 'Inverse Head and Shoulder'
    
    return df

# 配置 Telegram Bot
bot_token = TELEGRAM_BOT_TOKEN
chat_id = TELEGRAM_CHAT_ID
bot = Bot(token=bot_token)

async def send_message(message):
    try:
        await bot.send_message(chat_id=chat_id, text=message, parse_mode=ParseMode.HTML)
    except Exception as e:
        print(f"Failed to send message: {e}")


# 初始化 binance 交易所
binance = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True,
    'proxies': {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890',
    },
})

# 获取交易对信息
markets = binance.load_markets()
symbols = [symbol for symbol in markets.keys()][:100]
print(symbols)

# 获取交易对的K线数据
def fetch_ohlcv(symbol):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe='1h')
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.rename(columns=lambda x: x.capitalize(), inplace=True)
    df['Symbol'] = symbol  # 添加symbol列

    return df

# # 检测特定形态并发送消息
# for symbol in symbols:
#     df = fetch_ohlcv(symbol)
#     df = detect_head_shoulder_wavelet(df, window=20)

#     # # 检查是否有形态检测到，如果有，发送消息
#     # if df['wedge_pattern'].notna().any():
#     #     send_message(f"{symbol} detected wedge_pattern")

# # print(df[df.wedge_pattern=='Wedge Up'])
# print(df[df.head_shoulder_pattern=='Inverse Head and Shoulder'])

# 检测特定形态并发送消息
async def main():
    patterns_detected = {
        "Head and Shoulder": [],
        "Inverse Head and Shoulder": []
    }

    for symbol in symbols:
        try:
            df = fetch_ohlcv(symbol)
            df = detect_head_shoulder(df)

            if df['head_shoulder_pattern'].eq('Head and Shoulder').any():
                patterns_detected["Head and Shoulder"].append(symbol)
            if df['inv_head_shoulder_pattern'].eq('Inverse Head and Shoulder').any():
                patterns_detected["Inverse Head and Shoulder"].append(symbol)
        except Exception as e:
            print(f"Failed to process {symbol}: {e}")

    # 发送收集到的消息
    if patterns_detected["Head and Shoulder"]:
        head_shoulder_symbols = ", ".join(patterns_detected["Head and Shoulder"])
        await send_message(f"Detected Head and Shoulder pattern in symbols: {head_shoulder_symbols}")

    if patterns_detected["Inverse Head and Shoulder"]:
        inv_head_shoulder_symbols = ", ".join(patterns_detected["Inverse Head and Shoulder"])
        await send_message(f"Detected Inverse Head and Shoulder pattern in symbols: {inv_head_shoulder_symbols}")


# 运行主任务
asyncio.run(main())