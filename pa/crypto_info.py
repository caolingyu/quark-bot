import ccxt

def get_crypto_info(exchange, symbol):
    """
    获取币种的详细信息，例如市值，所属概念板块、简介等
    """
    ticker_info = exchange.fetch_ticker(symbol)
    market_info = exchange.fetch_markets()
    
    symbol_info = next((item for item in market_info if item['symbol'] == symbol), None)
    
    info = {
        'symbol': symbol,
        'market_cap': ticker_info.get('quoteVolume'),  # 假设quoteVolume代表市值，需要根据具体API调整
        'info': symbol_info  # 包含有关该交易对的所有信息
    }
    
    return info