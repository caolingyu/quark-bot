import ccxt
import time
import asyncio
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from telegram import Bot
from telegram.constants import ParseMode
from telegram.request import HTTPXRequest
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, USING_PROXY, FONT_PATH
from pa.pa_patterns import *
from pa.pa_index import *
from pa.crypto_info import *
from pa.sup_res import *
import os
import glob
import httpx
from retry import retry


HTTP_PROXY = os.getenv('HTTP_PROXY')
HTTPS_PROXY = os.getenv('HTTPS_PROXY')

FONT_PATH = os.getenv('FONT_PATH', FONT_PATH)


def draw_multiline_text(draw, text, position, font, max_width, fill):
    lines = []
    words = text.split(' ')
    line = ""
    for word in words:
        test_line = line + word + " "
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word + " "
    lines.append(line)
    
    y = position[1]
    for line in lines:
        draw.text((position[0], y), line, font=font, fill=fill)
        bbox = draw.textbbox((0, 0), line, font=font)
        y += bbox[3] - bbox[1]
    return y  # 返回绘制后的 y 坐标


def combine_images(image_paths, output_path, title=None):
    images = [Image.open(image_path) for image_path in image_paths]
    widths, heights = zip(*(image.size for image in images))

    total_height = sum(heights)
    max_width = max(widths)

    combined_image = Image.new('RGB', (max_width, total_height + 100), 'white')
    y_offset = 100

    draw = ImageDraw.Draw(combined_image)
    title_font = ImageFont.truetype(FONT_PATH, 60)

    if title:
        draw.text((10, 10), title, font=title_font, fill='black')

    for image in images:
        combined_image.paste(image, (0, y_offset))
        y_offset += image.size[1]

    combined_image.save(output_path)

def create_report_image(symbol, df, patterns, support_level, resistance_level, output_path, timeframe):
    # 设置图像尺寸，增加分辨率
    final_width = 1200
    final_height = 1800
    high_res_multiplier = 2  # 分辨率倍数
    width = final_width * high_res_multiplier
    height = final_height * high_res_multiplier

    # 创建一张高分辨率的空白图片作为报告的背景
    report_image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(report_image)
    title_font = ImageFont.truetype(FONT_PATH, int(40 * high_res_multiplier))
    font = ImageFont.truetype(FONT_PATH, int(30 * high_res_multiplier))

    # 绘制标题
    draw.text((50 * high_res_multiplier, 30 * high_res_multiplier), f'交易信号报告: {symbol} - {timeframe}', font=title_font, fill='black')

    # 添加K线图（放置在顶部）
    kline_path = f"imgs/{symbol.replace('/', '-')}_kline.png"
    plot_with_oscillator(df, symbol, filename=kline_path, support=support_level, resistance=resistance_level)

    kline_image = Image.open(kline_path)
    kline_image = kline_image.resize((1100 * high_res_multiplier, 600 * high_res_multiplier), Image.LANCZOS)
    report_image.paste(kline_image, (50 * high_res_multiplier, 100 * high_res_multiplier))

    text_y = 750 * high_res_multiplier

    draw.text((50 * high_res_multiplier, text_y), 'K线识别结果:', font=title_font, fill='black')
    text_y += 50 * high_res_multiplier
    text_y = draw_multiline_text(draw, f'检测到的K线形态: {", ".join(patterns)}', (50 * high_res_multiplier, text_y), font, 1100 * high_res_multiplier, 'black')

    text_y += 80 * high_res_multiplier
    draw.text((50 * high_res_multiplier, text_y), '支撑位和阻力位:', font=title_font, fill='black')
    text_y += 50 * high_res_multiplier
    draw.text((50 * high_res_multiplier, text_y), f'支撑位: {support_level}', font=font, fill='black')
    text_y += 30 * high_res_multiplier
    draw.text((50 * high_res_multiplier, text_y), f'阻力位: {resistance_level}', font=font, fill='black')

    text_y += 80 * high_res_multiplier

    crypto_info = get_crypto_info(binance, symbol)

    draw.text((50 * high_res_multiplier, text_y), '币种详细信息:', font=title_font, fill='black')
    text_y += 50 * high_res_multiplier
    text_y = draw_multiline_text(draw, f'市值: {crypto_info["market_cap"]}', (50 * high_res_multiplier, text_y), font, 1100 * high_res_multiplier, 'black')
    text_y += 30 * high_res_multiplier
    text_y = draw_multiline_text(draw, f'最新价格: {crypto_info["last_price"]}', (50 * high_res_multiplier, text_y), font, 1100 * high_res_multiplier, 'black')
    text_y += 30 * high_res_multiplier
    text_y = draw_multiline_text(draw, f'24小时成交量: {crypto_info["24h_volume"]}', (50 * high_res_multiplier, text_y), font, 1100 * high_res_multiplier, 'black')
    text_y += 30 * high_res_multiplier
    text_y = draw_multiline_text(draw, f'1小时涨跌幅: {crypto_info["change_1h"]:.2f}%', (50 * high_res_multiplier, text_y), font, 1100 * high_res_multiplier, 'black')
    text_y += 30 * high_res_multiplier
    text_y = draw_multiline_text(draw, f'4小时涨跌幅: {crypto_info["change_4h"]:.2f}%', (50 * high_res_multiplier, text_y), font, 1100 * high_res_multiplier, 'black')
    text_y += 30 * high_res_multiplier
    text_y = draw_multiline_text(draw, f'24小时涨跌幅: {crypto_info["change_24h"]:.2f}%', (50 * high_res_multiplier, text_y), font, 1100 * high_res_multiplier, 'black')

    # 将图片调整到最终尺寸
    final_image = report_image.resize((final_width, final_height), Image.LANCZOS)
    final_image.save(output_path)


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

# bot = Bot(token=bot_token)

# 配置 Telegram Bot 并使用代理
def get_telegram_bot():
    if USING_PROXY:
        request = HTTPXRequest(proxy_url=HTTP_PROXY)
        return Bot(token=TELEGRAM_BOT_TOKEN, request=request)
    else:
        return Bot(token=TELEGRAM_BOT_TOKEN)

bot = get_telegram_bot()


async def send_message(message):
    try:
        await bot.send_message(chat_id=chat_id, text=message, parse_mode=ParseMode.HTML)
    except Exception as e:
        print(f"Failed to send message: {e}")

from telegram.error import TimedOut

async def send_photo_via_bot(photo_path, chat_id, bot_token, caption='', max_retries=5, delay=2):
    bot = get_telegram_bot()
    for attempt in range(1, max_retries + 1):
        try:
            with open(photo_path, 'rb') as photo:
                await bot.send_photo(chat_id=chat_id, photo=photo, caption=caption)
                return
        except TimedOut:
            if attempt == max_retries:
                print(f"Failed to send photo after {max_retries} attempts due to timeout")
                raise
            else:
                print(f"Timed out. Retrying {attempt}/{max_retries}...")
                await asyncio.sleep(delay * attempt)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

async def send_message_via_bot(message, chat_id, bot_token, max_retries=5, delay=2):
    bot = get_telegram_bot()
    for attempt in range(1, max_retries + 1):
        try:
            await bot.send_message(chat_id=chat_id, text=message)
            return
        except TimedOut:
            if attempt == max_retries:
                print(f"Failed to send message after {max_retries} attempts due to timeout")
                raise
            else:
                print(f"Timed out. Retrying {attempt}/{max_retries}...")
                await asyncio.sleep(delay * attempt)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break


def convert_to_chinese_unit(value):
    value = float(value)
    units = ['', '千', '万', '亿']
    unit_index = 0
    while value >= 1000 and unit_index < len(units) - 1:
        value /= 1000.0
        unit_index += 1
    # 为了保证精度，这里对浮点数进行格式化处理，保留一位小数
    return f'{value:.1f}{units[unit_index]}'

async def process_and_send_patterns(patterns_detected, chat_id, bot_token, timeframe, pct_ranking, all_ohlcv):
    pattern_to_desc_map = {
        "Straddle": "盘整突破",
        "Overbought": "超买",
        "Oversold": "超卖"
    }

    for pattern, symbols in patterns_detected.items():
        if symbols:
            for symbol in symbols:
                # df = fetch_ohlcv(symbol, timeframe=timeframe)
                df = all_ohlcv[symbol][timeframe]
                df = detect_pa_index_patterns(df)
                image_path = f"imgs/{symbol.replace('/', '-')}_kline.png"

                candle_patterns = detect_candlestick_patterns(df)
                support_levels, resistance_levels = calculate_recent_support_resistance(df)
                fib_level = detect_fibonacci_level(df)
                ma_analysis = analyze_latest_moving_averages(df)


                # 生成K线图
                plot_with_oscillator(df, symbol, filename=image_path, support=support_levels, resistance=resistance_levels)

                # 获取文字描述
                message = f"检测到 【{timeframe}】 级别 【{pattern_to_desc_map.get(pattern)}】的币种：{symbol}\n\n"
                message += f'交易信号报告: {symbol} - {timeframe}\n\n'
                message += f'检测到的K线形态: {", ".join(candle_patterns)}\n\n'
                message += f'支撑位: {support_levels}\n'
                message += f'阻力位: {resistance_levels}\n\n'
                message += f'斐波那契: {fib_level}\n\n'
                message += f'均线组形态: {ma_analysis.get('Trend')}\n\n'
                message += f'均线组密集程度: {ma_analysis.get('Density')}\n\n'

                crypto_info = get_crypto_info(binance, symbol, all_ohlcv)
                crypto_quote_volume = convert_to_chinese_unit(crypto_info["quote_volume"])
                crypto_24h_volume = convert_to_chinese_unit(crypto_info["24h_volume"])

                message += f'币种详细信息:\n'
                message += f'Quote Volume: {crypto_quote_volume}\n'
                message += f'最新价格: {str(crypto_info["last_price"])}\n'
                message += f'24小时成交量: {crypto_24h_volume}\n'
                message += f'1小时涨跌幅: {crypto_info["change_1h"]:.2f}%\n'
                message += f'4小时涨跌幅: {crypto_info["change_4h"]:.2f}%\n'
                message += f'24小时涨跌幅: {crypto_info["change_24h"]:.2f}%\n\n'
                # 添加相对强弱排名信息
                for tf in TIMEFRAME:
                    rank = pct_ranking[tf].get(symbol, "N/A")
                    message += f'相对强弱排名 ({tf}): {rank}\n'


                # 发送图片和文字描述到Telegram
                await send_photo_via_bot(image_path, chat_id, bot_token, caption=message)

    # 在发送完成后删除imgs路径下的所有图片
    delete_images_in_folder("imgs")

# 初始化 binance 交易所
if USING_PROXY:
    binance = ccxt.binance({
        'rateLimit': 1200,
        'enableRateLimit': True,
        'proxies': {
            'http': HTTP_PROXY,
            'https': HTTPS_PROXY,
        },
    })
else:
    binance = ccxt.binance({
        'rateLimit': 1200,
        'enableRateLimit': True
    })

# 获取所有交易对的 ticker 信息
# tickers = binance.fetch_tickers()

@retry(exceptions=ccxt.NetworkError, tries=5, delay=2, backoff=2)
def fetch_tickers_with_retry():
    return binance.fetch_tickers()

tickers = fetch_tickers_with_retry()


# 获取所有 USDT 的交易对，并提取交易量信息
usdt_pairs = []
for symbol, ticker in tickers.items():
    if symbol.endswith('/USDT'):
        # 使用 quoteVolume 来代表交易量
        usdt_pairs.append((symbol, ticker['quoteVolume']))

# 按交易量排序，选择前50名
top_usdt_pairs = sorted(usdt_pairs, key=lambda x: x[1], reverse=True)[:50]
symbols = [symbol for symbol, _ in top_usdt_pairs if symbol not in ['USDC/USDT', 'EUR/USDT']]

print("Top 50 USDT pairs by market volume:")
print(symbols)

# 获取交易对的K线数据
def fetch_ohlcv(symbol, timeframe='15m', limit=200):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.rename(columns=lambda x: x.capitalize(), inplace=True)
    df.set_index('Timestamp', inplace=True)  # 确保时间戳为索引，并且是 DatetimeIndex
    df['Symbol'] = symbol  # 添加symbol列
    return df

def fetch_all_ohlcv(symbols, timeframes):
    """获取所有指定符号和时间框架的OHLCV数据并缓存"""
    ohlcv_data = {}

    for symbol in symbols:
        ohlcv_data[symbol] = {}
        for tf in timeframes:
            df = fetch_ohlcv(symbol, timeframe=tf)
            ohlcv_data[symbol][tf] = df.iloc[:-1,]

    return ohlcv_data

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


def calculate_percentage_change(df):
    """计算涨跌幅度"""
    return ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100

def get_percentage_change_rankings(symbols, timeframes, all_ohlcv):
    rankings = {tf: {} for tf in timeframes}

    for symbol in symbols:
        for tf in timeframes:
            df = all_ohlcv[symbol][tf]
            change_percent = calculate_percentage_change(df)
            rankings[tf][symbol] = change_percent

    for tf in rankings:
        sorted_rankings = sorted(rankings[tf].items(), key=lambda item: item[1], reverse=True)
        rankings[tf] = {symbol: rank for rank, (symbol, _) in enumerate(sorted_rankings, 1)}

    return rankings

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

TIMEFRAME = ['15m', '1h', '4h']

# 检测特定形态并发送消息
async def main():
    # 获取所有需要的K线数据
    all_ohlcv = fetch_all_ohlcv(symbols, TIMEFRAME)
    pct_ranking = get_percentage_change_rankings(symbols, TIMEFRAME, all_ohlcv)
    for tf in TIMEFRAME:
        patterns_detected = {
            "Oversold": [],
            "Overbought": [],
            "Straddle": []
        }
        for symbol in symbols:
            # df = fetch_ohlcv(symbol, timeframe=tf)
            try:
                df = all_ohlcv[symbol][tf]
            except:
                continue
            df = detect_pa_index_patterns(df)

            # if df['Zone'].iloc[-1] == 'Straddle':
            if all(df['Zone'].iloc[-1-j] == 'Straddle' for j in range(1, 6)) and df['Zone'].iloc[-1] != 'Straddle':
                patterns_detected['Straddle'].append(symbol)
            elif df['Zone'].iloc[-1] == 'Oversold':
                patterns_detected['Oversold'].append(symbol)
            elif df['Zone'].iloc[-1] == 'Overbought':
                patterns_detected['Overbought'].append(symbol)
            
            time.sleep(5)

        await process_and_send_patterns(patterns_detected, chat_id=chat_id, bot_token=bot_token, timeframe=tf, 
                                        pct_ranking=pct_ranking, all_ohlcv=all_ohlcv)

def test_ranking():
    print(get_percentage_change_rankings(symbols, ['4h']))



# 运行主任务
# asyncio.run(run_at_specific_time())
asyncio.run(main())

# print(backtest_strategy('BTCUSDT'))
