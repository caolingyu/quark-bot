from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# 用你的 Token 替换这里的 'YOUR_BOT_TOKEN'
TOKEN = TELEGRAM_BOT_TOKEN

# 用你要展示的图片链接替换这里的 'YOUR_PHOTO_URL'
PHOTO_URL = 'assets/logo.png'
# 用你的介绍文字替换这里的 'YOUR_INTRO_TEXT'
INTRO_TEXT = """
 ▎欢迎加入夸克量化！

    夸克量化是一个专注于价格行为学交易技术和量化技术的社区。我们致力于开发和应用先进的量化交易算法，帮助交易者更好地理解市场行为，制定高效的交易策略。

 ▎❕ 重要提醒：本系统是辅助交易系统，不提供完全自动交易功能。自动交易功能目前不公开，你需要自己掌控交易节奏，制定交易计划，盈亏自负。求带单服务的请另寻他处。

 ▎ 我们的独特优势：

    1️⃣. 夸克量化专注于价格行为学，利用先进的量化技术，从多个交易所获取原始价格信息，实现高效且精准的交易模型，完全独立开发，避免对第三方平台的依赖。

    2️⃣. 系统由经验丰富的程序员和交易员共同开发，确保高效、稳定，有效捕捉市场波动。

    3️⃣. 本系统自动化程度高，从信号扫描到自动交易全面覆盖，运行流畅，无需人工介入。

 ℹ️ 更多功能，请在左下角menu中查看。
"""
# 定义处理 /start 命令的函数
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    # 发送顶部图片
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=PHOTO_URL)

    # 发送介绍文字
    await context.bot.send_message(chat_id=update.effective_chat.id, text=INTRO_TEXT)

    # 创建按钮
    keyboard = [
        [
            InlineKeyboardButton("交流群", url="https://t.me/+6onLm2P5LehlN2U9"),
            InlineKeyboardButton("信号群", url="https://t.me/+l57clUvMW0s2NmY1"),
        ]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # 给用户发送信息并附上按钮
    await context.bot.send_message(chat_id=update.effective_chat.id, text="请选择一个群组加入:", reply_markup=reply_markup)

def main() -> None:
    # 创建 Application 对象并传入 Token
    application = Application.builder().token(TOKEN).build()

    # 获取调度器以注册处理器
    application.add_handler(CommandHandler("start", start))

    # 启动 Bot
    application.run_polling()

if __name__ == '__main__':
    main()