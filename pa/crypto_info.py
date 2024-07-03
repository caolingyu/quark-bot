import ccxt
import time

def get_crypto_info(exchange, symbol, all_ohlcv):
    """
    获取币种的详细信息，例如市值，所属概念板块、简介等
    """

    max_retries = 5

    # 获取时间段的ohlcv数据用于计算涨跌幅
    def get_percentage_change(data):
        if len(data) < 2:
            return None
        first_close = data[0][4]
        last_close = data[-1][4]
        return (last_close - first_close) / first_close * 100


    for attempt in range(1, max_retries + 1):
        try:
            ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=2)
            ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=2)
            ohlcv_24h = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=2)
            ticker_info = exchange.fetch_ticker(symbol)
            market_info = exchange.fetch_markets()
            # 获取symbol的信息
            symbol_info = next((item for item in market_info if item['symbol'] == symbol), None)
            break
        except:
            if attempt == max_retries:
                print(f"Failed to send photo after {max_retries} attempts due to timeout")
                raise
            else:
                print(f"Timed out. Retrying {attempt}/{max_retries}...")
                time.sleep(5)


    change_1h = get_percentage_change(ohlcv_1h)
    change_4h = get_percentage_change(ohlcv_4h)
    change_24h = get_percentage_change(ohlcv_24h)

    # 获取市值或交易量信息
    quote_volume = ticker_info.get('quoteVolume', None)  # 视具体情况而定

    info = {
        'symbol': symbol,
        'quote_volume': quote_volume,
        'change_1h': change_1h,
        'change_4h': change_4h,
        'change_24h': change_24h,
        'last_price': ticker_info.get('last', None),
        '24h_volume': ticker_info.get('info', {}).get('volume', None)
    }
    
    return info

# # 示例调用
# exchange = ccxt.binance({
#         'rateLimit': 1200,
#         'enableRateLimit': True,
#         'proxies': {
#             'http': 'http://127.0.0.1:7890',
#             'https': 'http://127.0.0.1:7890',
#         },
#     })
# symbol = "BTC/USDT"
# info = get_crypto_info(exchange, symbol, None)
# print(info)