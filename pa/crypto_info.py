import ccxt

def get_crypto_info(exchange, symbol):
    """
    获取币种的详细信息，例如市值，所属概念板块、简介等
    """
    ticker_info = exchange.fetch_ticker(symbol)
    market_info = exchange.fetch_markets()

    # 获取symbol的信息
    symbol_info = next((item for item in market_info if item['symbol'] == symbol), None)

    # 获取时间段的ohlcv数据用于计算涨跌幅
    def get_percentage_change(data):
        if len(data) < 2:
            return None
        first_close = data[0][4]
        last_close = data[-1][4]
        return (last_close - first_close) / first_close * 100

    ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=2)
    ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=2)
    ohlcv_24h = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=2)

    change_1h = get_percentage_change(ohlcv_1h)
    change_4h = get_percentage_change(ohlcv_4h)
    change_24h = get_percentage_change(ohlcv_24h)

    # 获取市值或交易量信息
    market_cap = ticker_info.get('quoteVolume', None)  # 视具体情况而定

    info = {
        'symbol': symbol,
        'market_cap': market_cap,
        'change_1h': change_1h,
        'change_4h': change_4h,
        'change_24h': change_24h,
        'last_price': ticker_info.get('last', None),
        '24h_volume': ticker_info.get('quoteVolume24h', None)
    }
    
    return info

# # 示例调用
# exchange = ccxt.binance()
# symbol = "BTC/USDT"
# info = get_crypto_info(exchange, symbol)
# print(info)