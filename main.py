import logging
import requests
import schedule
import time
from telegram import Update, Bot
from telegram.ext import Updater, CommandHandler, CallbackContext
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from config import TELEGRAM_BOT_TOKEN, CRYPTO_NEWS_API_TOKEN

# 设置日志
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# 初始化 Telegram bot
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# 加密货币新闻API
CRYPTO_NEWS_API_URL = f'https://cryptonews-api.com/api/v1?tickers=all&items=10&token={CRYPTO_NEWS_API_TOKEN}'

# 加密货币行情数据API
COINGECKO_API_URL = 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd'

def get_crypto_news():
    response = requests.get(CRYPTO_NEWS_API_URL)
    news_data = response.json()
    return news_data['data']

def get_market_data():
    response = requests.get(COINGECKO_API_URL)
    market_data = response.json()
    return market_data

def summarize_news_and_market_data(news_data, market_data):
    # 模拟 LangChain 和 LLM 的实现
    news_summary = "Summary of latest news: ..."
    market_summary = f"Market data: BTC: ${market_data['bitcoin']['usd']}, ETH: ${market_data['ethereum']['usd']}"

    # 构造提示
    prompt = PromptTemplate(
        input_variables=["news", "market"],
        template="News: {news}\nMarket Data: {market}\nSummarize this information."
    )
    chain = LLMChain(prompt)

    collective_data = {"news": news_summary, "market": market_summary}
    summary = chain(collective_data)
    return summary

def send_update(update: Update, context: CallbackContext) -> None:
    news_data = get_crypto_news()
    market_data = get_market_data()
    summary = summarize_news_and_market_data(news_data, market_data)
    bot.send_message(chat_id=update.effective_chat.id, text=summary)

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Hello! This bot provides hourly summaries of the latest cryptocurrency news and market data.')

def main():
    updater = Updater(TELEGRAM_BOT_TOKEN)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))

    # 使用Schedule每小时执行任务
    schedule.every().hour.do(send_update)

    updater.start_polling()
    updater.idle()

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    main()