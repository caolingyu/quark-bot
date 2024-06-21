import ccxt
import asyncio
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
from telegram import Bot
from telegram.constants import ParseMode
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from pa.patterns import *
from pa.pa_index import *
import os
import glob



def delete_images_in_folder(folder_path):
    # 构建搜索模式以匹配所有的.png文件
    pattern = os.path.join(folder_path, '*.png')
    # 使用glob找到所有匹配的文件
    files = glob.glob(pattern)
    # 遍历文件列表并删除每个文件
    for file in files:
        os.remove(file)
    print("All images in the folder have been deleted.")


def calculate_atr(df, period=14):
    df['TR'] = np.maximum((df['High'] - df['Low']), 
                          np.maximum(abs(df['High'] - df['Close'].shift()), 
                                     abs(df['Low'] - df['Close'].shift())))
    df['ATR'] = df['TR'].rolling(window=period).mean()
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

async def send_message_via_bot(message, chat_id, bot_token):
    bot = Bot(token=bot_token)
    await bot.send_message(chat_id=chat_id, text=message)

async def send_photo_via_bot(photo_path, chat_id, bot_token):
    bot = Bot(token=bot_token)
    await bot.send_photo(chat_id=chat_id, photo=open(photo_path, 'rb'))


async def process_and_send_patterns(patterns_detected, chat_id, bot_token):
    for pattern, symbols in patterns_detected.items():
        if symbols:
            symbols_text = ", ".join(symbols)
            await send_message_via_bot(f"Detected {pattern} in symbols: {symbols_text}", chat_id, bot_token)

            image_paths = []
            for symbol in symbols:
                df = fetch_ohlcv(symbol)
                df = detect_pa_index_patterns(df)
                image_path = f"imgs/{symbol.replace('/', '-')}_{pattern.lower()}.png"
                plot_with_oscillator(df, symbol, filename=image_path)
                image_paths.append(image_path)

            if image_paths:
                combined_image_path = f"imgs/{pattern.lower()}_combined.png"
                combine_images(image_paths, combined_image_path)
                await send_photo_via_bot(combined_image_path, chat_id, bot_token)
    # 在发送完成后删除imgs路径下的所有图片
    delete_images_in_folder("imgs")

# 初始化 binance 交易所
binance = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True,
    'proxies': {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890',
    },
})

# 获取所有交易对的 ticker 信息
tickers = binance.fetch_tickers()

# 获取所有 USDT 的交易对，并提取交易量信息
usdt_pairs = []
for symbol, ticker in tickers.items():
    if symbol.endswith('/USDT'):
        # 使用 quoteVolume 来代表交易量
        usdt_pairs.append((symbol, ticker['quoteVolume']))

# 按交易量排序，选择前50名
top_usdt_pairs = sorted(usdt_pairs, key=lambda x: x[1], reverse=True)[:50]
symbols = [symbol for symbol, _ in top_usdt_pairs]

print("Top 50 USDT pairs by market volume:")
print(symbols)

# 获取交易对的K线数据
def fetch_ohlcv(symbol, timeframe='1h', limit=100):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.rename(columns=lambda x: x.capitalize(), inplace=True)
    df.set_index('Timestamp', inplace=True)  # 确保时间戳为索引，并且是 DatetimeIndex
    df['Symbol'] = symbol  # 添加symbol列
    return df


# 计算技术指标
def calculate_indicators(df):
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = df['Close'].diff(1).apply(lambda x: np.nan if x == 0 else (x if x > 0 else 0)).rolling(window=14).sum() / df['Close'].diff(1).abs().rolling(window=14).sum() * 100
    df['MiddleBand'] = df['Close'].rolling(window=20).mean()
    df['UpperBand'] = df['MiddleBand'] + 2 * df['Close'].rolling(window=20).std()
    df['LowerBand'] = df['MiddleBand'] - 2 * df['Close'].rolling(window=20).std()
    return df

# 新的技术形态检测函数
def detect_pattern(df):
    df = calculate_indicators(df)
    long_condition = (df['Close'] > df['SMA50']) & (df['SMA50'] > df['SMA200']) & (df['RSI'] > 30) & (df['Close'] > df['MiddleBand'])
    short_condition = (df['Close'] < df['SMA50']) & (df['SMA50'] < df['SMA200']) & (df['RSI'] < 70) & (df['Close'] < df['MiddleBand'])

    df['Pattern'] = np.nan
    df = df.astype({"Pattern": "object"})
    df.loc[long_condition, 'Pattern'] = 'Long'
    df.loc[short_condition, 'Pattern'] = 'Short'
    
    return df


def combine_images(image_paths, output_path, rows=1):
    images = [Image.open(image) for image in image_paths]
    
    # 获取图片的尺寸
    widths, heights = zip(*(i.size for i in images))
    
    total_width = max(widths)
    total_height = sum(heights)
    
    # 创建一个新的空白图片对象
    combined_image = Image.new('RGB', (total_width, total_height))
    
    y_offset = 0
    for image in images:
        combined_image.paste(image, (0, y_offset))
        y_offset += image.size[1]

    # 提高图片的清晰度和对比度
    sharpness_enhancer = ImageEnhance.Sharpness(combined_image)
    clarity_enhancer = ImageEnhance.Contrast(combined_image)

    enhanced_image = sharpness_enhancer.enhance(2.0)  # 锐化处理
    enhanced_image = clarity_enhancer.enhance(1.5)  # 对比度增强

    # 保存合并后的图片并提高分辨率
    enhanced_image.save(output_path, dpi=(300, 300))


import asyncio
from datetime import datetime, timedelta

async def run_at_specific_time():
    while True:
        now = datetime.now()
        # 计算下一个整点过30秒的时间
        next_run_time = (now + timedelta(hours=1)).replace(minute=0, second=30, microsecond=0)
        # 计算当前时间到下一个运行时间的等待秒数
        wait_seconds = (next_run_time - now).total_seconds()

        print(f"Next run at: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
        await asyncio.sleep(wait_seconds)

        # 运行您的主要任务函数
        await main()


# 回测策略
def backtest_strategy(symbol, limit=240, atr_period=14, atr_multiplier=2):
    df = fetch_ohlcv(symbol, limit=limit)
    df = calculate_atr(df, period=atr_period)
    df = detect_pa_index_patterns(df)  # 检测形态

    trades = 0
    wins = 0
    print(df['Zone'].tolist())

    for i in range(1, len(df)):
        if df['Zone'].iloc[i-1] == 'Straddle' and df['Zone'].iloc[i] != 'Straddle':
            entry_price = df['Close'].iloc[i]
            atr_value = df['ATR'].iloc[i]

            print(entry_price)
            print(df['Close'].iloc[i-1])
            if entry_price < df['Close'].iloc[i-1]:
                # 假设开空单
                stop_loss = entry_price + atr_multiplier * atr_value
                take_profit = entry_price - atr_multiplier * atr_value
                trades += 1
                for j in range(i, len(df)):
                    if df['Low'].iloc[j] <= stop_loss:
                        break
                    if df['High'].iloc[j] >= take_profit:
                        wins += 1
                        break

            if entry_price > df['Close'].iloc[i-1]:
                # 假设开多单
                stop_loss = entry_price - atr_multiplier * atr_value
                take_profit = entry_price + atr_multiplier * atr_value
                trades += 1
                for j in range(i, len(df)):
                    if df['High'].iloc[j] >= take_profit:
                        wins += 1
                        break
                    if df['Low'].iloc[j] <= stop_loss:
                        break

    winrate = (wins / trades) * 100 if trades > 0 else 0
    return winrate, trades


# 检测特定形态并发送消息
async def main():
    patterns_detected = {
        "Oversold": [],
        "Overbought": [],
        "Straddle": []
    }

    for symbol in symbols:
        df = fetch_ohlcv(symbol)
        df = detect_pa_index_patterns(df)

        if df['Zone'].iloc[-1] == 'Straddle':
            patterns_detected['Straddle'].append(symbol)
        elif df['Zone'].iloc[-1] == 'Oversold':
            patterns_detected['Oversold'].append(symbol)
        elif df['Zone'].iloc[-1] == 'Overbought':
            patterns_detected['Overbought'].append(symbol)

    print(patterns_detected)
    await process_and_send_patterns(patterns_detected, chat_id=chat_id, bot_token=bot_token)


# 运行主任务
# asyncio.run(run_at_specific_time())
# asyncio.run(main())

print(backtest_strategy('BTCUSDT'))