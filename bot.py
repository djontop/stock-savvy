import asyncio
import logging
from datetime import datetime, timedelta
import yfinance as yf
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler, ConversationHandler, MessageHandler, filters
from telegram.error import TelegramError
from typing import Dict, List, Tuple
import json
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import aiohttp

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

BOT_TOKEN = '7541281361:AAEKcMwMYRsfu5Ba0tx3-U0Se4ZdvAnJDvo'
ALERT_CHECK_INTERVAL = 30
ANNOUNCEMENT_CHECK_INTERVAL = 300 
DATA_FILE = 'user_alerts.json'
ANNOUNCEMENT_FILE = 'last_announcements.json'


NSE_BASE_URL = "https://www.nseindia.com"
NSE_ANNOUNCEMENT_URL = f"{NSE_BASE_URL}/companies-listing/corporate-filings-announcements"


NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

SEARCH_SYMBOL, SET_PRICE, SUBSCRIBE_SYMBOL, UNSUBSCRIBE_SYMBOL = range(4)

class StockAlertBot:
    def __init__(self):
        self.user_alerts: Dict[str, List[Tuple[str, float]]] = {}
        self.user_temp_data: Dict[int, Dict] = {}
        self.user_subscriptions: Dict[int, List[str]] = {}
        self.last_announcements: Dict[str, List[Dict]] = {}
        self.load_alerts()
        self.load_subscriptions()
        self.load_last_announcements()

    def load_alerts(self) -> None:
        """Load saved alerts from file"""
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'r') as f:
                    data = json.load(f)
                    self.user_alerts = {int(k): v for k, v in data.items()}
                logger.info("Alerts loaded from file")
        except Exception as e:
            logger.error(f"Error loading alerts: {e}")

    def save_alerts(self) -> None:
        """Save alerts to file"""
        try:
            with open(DATA_FILE, 'w') as f:
                json.dump(self.user_alerts, f)
            logger.info("Alerts saved to file")
        except Exception as e:
            logger.error(f"Error saving alerts: {e}")

    def load_subscriptions(self) -> None:
        """Load saved announcement subscriptions from file"""
        try:
            if os.path.exists('user_subscriptions.json'):
                with open('user_subscriptions.json', 'r') as f:
                    self.user_subscriptions = json.load(f)
                logger.info("Subscriptions loaded from file")
        except Exception as e:
            logger.error(f"Error loading subscriptions: {e}")

    def save_subscriptions(self) -> None:
        """Save announcement subscriptions to file"""
        try:
            with open('user_subscriptions.json', 'w') as f:
                json.dump(self.user_subscriptions, f)
            logger.info("Subscriptions saved to file")
        except Exception as e:
            logger.error(f"Error saving subscriptions: {e}")

    def load_last_announcements(self) -> None:
        """Load cached announcements from file"""
        try:
            if os.path.exists(ANNOUNCEMENT_FILE):
                with open(ANNOUNCEMENT_FILE, 'r') as f:
                    self.last_announcements = json.load(f)
                logger.info("Last announcements loaded from file")
        except Exception as e:
            logger.error(f"Error loading last announcements: {e}")

    def save_last_announcements(self) -> None:
        """Save cached announcements to file"""
        try:
            with open(ANNOUNCEMENT_FILE, 'w') as f:
                json.dump(self.last_announcements, f)
            logger.info("Last announcements saved to file")
        except Exception as e:
            logger.error(f"Error saving last announcements: {e}")

    async def fetch_nse_announcements(self, symbol: str) -> List[Dict]:
        """Fetch latest announcements for a given symbol from NSE"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{NSE_ANNOUNCEMENT_URL}/filtered?symbol={symbol}",
                    headers=NSE_HEADERS
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        announcements = []
                        for item in data.get('data', []):
                            announcement = {
                                'symbol': symbol,
                                'date': item.get('date'),
                                'subject': item.get('subject'),
                                'category': item.get('category'),
                                'url': f"{NSE_BASE_URL}{item.get('url', '')}"
                            }
                            announcements.append(announcement)
                        return announcements
                    else:
                        logger.error(f"Failed to fetch announcements for {symbol}: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching announcements for {symbol}: {e}")
            return []

    async def subscribe_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Start the subscription conversation"""
        await update.message.reply_text(
            "Please enter a company name or stock symbol to search for (e.g., 'RELIANCE' or 'TCS'):"
        )
        return SUBSCRIBE_SYMBOL
    
    async def unsubscribe_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Start the unsubscription conversation"""
        chat_id = update.effective_chat.id
        
        if chat_id not in self.user_subscriptions or not self.user_subscriptions[chat_id]:
            await update.message.reply_text("You don't have any active subscriptions to remove.")
            return ConversationHandler.END
            
        keyboard = []
        for symbol in self.user_subscriptions[chat_id]:
            keyboard.append([InlineKeyboardButton(symbol, callback_data=f"unsub_{symbol}")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "Select a stock to unsubscribe from:",
            reply_markup=reply_markup
        )
        return UNSUBSCRIBE_SYMBOL
    
    async def subscribe_announcements(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /subscribe command"""
        if not context.args:
            await update.message.reply_text(
                "Usage: /subscribe <stock_symbol>\n"
                "Example: /subscribe RELIANCE"
            )
            return

        chat_id = update.effective_chat.id
        symbol = context.args[0].upper()

        announcements = await self.fetch_nse_announcements(symbol)
        if not announcements:
            await update.message.reply_text(
                f"Could not find announcements for {symbol}. "
                "Please check if the symbol is correct."
            )
            return

        if chat_id not in self.user_subscriptions:
            self.user_subscriptions[chat_id] = []

        if symbol not in self.user_subscriptions[chat_id]:
            self.user_subscriptions[chat_id].append(symbol)
            self.save_subscriptions()
            await update.message.reply_text(
                f"Successfully subscribed to announcements for {symbol}!"
            )
        else:
            await update.message.reply_text(
                f"You are already subscribed to announcements for {symbol}"
            )

    async def unsubscribe_announcements(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /unsubscribe command"""
        if not context.args:
            await update.message.reply_text(
                "Usage: /unsubscribe <stock_symbol>\n"
                "Example: /unsubscribe RELIANCE"
            )
            return

        chat_id = update.effective_chat.id
        symbol = context.args[0].upper()

        if chat_id in self.user_subscriptions and symbol in self.user_subscriptions[chat_id]:
            self.user_subscriptions[chat_id].remove(symbol)
            self.save_subscriptions()
            await update.message.reply_text(
                f"Successfully unsubscribed from announcements for {symbol}"
            )
        else:
            await update.message.reply_text(
                f"You are not subscribed to announcements for {symbol}"
            )

    async def list_subscriptions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /mysubscriptions command"""
        chat_id = update.effective_chat.id
        
        if chat_id not in self.user_subscriptions or not self.user_subscriptions[chat_id]:
            await update.message.reply_text("You don't have any active announcement subscriptions")
            return

        subscriptions_text = "Your announcement subscriptions:\n\n"
        for symbol in self.user_subscriptions[chat_id]:
            subscriptions_text += f"• {symbol}\n"
        
        await update.message.reply_text(subscriptions_text)

    async def handle_subscribe_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
            """Handle the stock symbol search for subscription"""
            query = update.message.text
            results = await self.search_stock(query)

            if not results:
                await update.message.reply_text(
                    "No stocks found matching your search. Please try again with a different name:"
                )
                return SUBSCRIBE_SYMBOL

            keyboard = []
            for result in results:
                text = f"{result['symbol']} - {result['name']} ({result['exchange']})"
                keyboard.append([InlineKeyboardButton(text, callback_data=f"sub_{result['symbol']}")])

            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "Select a stock to subscribe to announcements:",
                reply_markup=reply_markup
            )
            return ConversationHandler.END

    async def handle_subscription_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            """Handle the selected stock symbol for subscription"""
            query = update.callback_query
            await query.answer()
            
            symbol = query.data.replace("sub_", "")
            chat_id = query.message.chat_id

            if chat_id not in self.user_subscriptions:
                self.user_subscriptions[chat_id] = []

            if symbol not in self.user_subscriptions[chat_id]:
                self.user_subscriptions[chat_id].append(symbol)
                self.save_subscriptions()
                await query.message.reply_text(
                    f"Successfully subscribed to announcements for {symbol}! "
                    f"You'll receive notifications when new announcements are posted."
                )
            else:
                await query.message.reply_text(
                    f"You are already subscribed to announcements for {symbol}"
                )

    async def handle_unsubscribe_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            """Handle the selected stock symbol for unsubscription"""
            query = update.callback_query
            await query.answer()
            
            symbol = query.data.replace("unsub_", "")
            chat_id = query.message.chat_id

            if chat_id in self.user_subscriptions and symbol in self.user_subscriptions[chat_id]:
                self.user_subscriptions[chat_id].remove(symbol)
                self.save_subscriptions()
                await query.message.reply_text(
                    f"Successfully unsubscribed from announcements for {symbol}"
                )
            else:
                await query.message.reply_text(
                    f"You are not subscribed to announcements for {symbol}"
                )

    async def check_announcements(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Check for new announcements and notify subscribed users"""
        for chat_id, symbols in self.user_subscriptions.items():
            for symbol in symbols:
                try:
                    new_announcements = await self.fetch_nse_announcements(symbol)
                    
                    last_sent = self.last_announcements.get(symbol, [])
                    
                    for announcement in new_announcements:
                        if announcement not in last_sent:
                            message = (
                                f"🔔 New announcement for {symbol}\n\n"
                                f"Date: {announcement['date']}\n"
                                f"Category: {announcement['category']}\n"
                                f"Subject: {announcement['subject']}\n"
                                f"Link: {announcement['url']}"
                            )
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text=message,
                                disable_web_page_preview=True
                            )
                    
                    self.last_announcements[symbol] = new_announcements
                    self.save_last_announcements()
                
                except Exception as e:
                    logger.error(f"Error checking announcements for {symbol}: {e}")
                    continue

    async def search_stock(self, query: str) -> List[Dict]:
        """Search for stocks based on query"""
        try:
            url = f"https://query2.finance.yahoo.com/v1/finance/search"
            params = {
                'q': query,
                'quotesCount': 5,
                'newsCount': 0,
                'enableFuzzyQuery': True,
                'quotesQueryId': 'tss_match_phrase_query'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers)
            data = response.json()
            
            if 'quotes' in data:
                return [
                    {
                        'symbol': quote['symbol'],
                        'name': quote.get('shortname', quote.get('longname', 'N/A')),
                        'exchange': quote.get('exchange', 'N/A')
                    }
                    for quote in data['quotes']
                    if 'symbol' in quote
                ]
            return []
        except Exception as e:
            logger.error(f"Error searching stocks: {e}")
            return []

    @staticmethod
    async def validate_stock_symbol(symbol: str) -> bool:
        """Validate if the stock symbol exists"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            return 'regularMarketPrice' in info
        except Exception:
            return False

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /start command"""
        help_text = (
            "Welcome to the Stock Alert Bot! 📈\n\n"
            "Available commands:\n"
            "/setalert - Set a new price alert (with stock search)\n"
            "/updatealert <symbol> <price> - Update existing alert\n"
            "/myalerts - View your current alerts\n"
            "/deletealert <symbol> - Delete an alert\n"
            "/subscribe - Subscribe to NSE announcements (with stock search)\n"
            "/unsubscribe - Unsubscribe from NSE announcements\n"
            "/mysubscriptions - View your announcement subscriptions\n"
            "/help - Show this help message"
        )
        await update.message.reply_text(help_text)
        logger.info(f"Start command used by user {update.effective_user.id}")

    async def setalert_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Start the set alert conversation"""
        await update.message.reply_text(
            "Please enter a company name or stock symbol to search for (e.g., 'Apple' or 'AAPL'):"
        )
        return SEARCH_SYMBOL

    async def handle_symbol_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle the stock symbol search"""
        query = update.message.text
        results = await self.search_stock(query)

        if not results:
            await update.message.reply_text(
                "No stocks found matching your search. Please try again with a different name:"
            )
            return SEARCH_SYMBOL

        keyboard = []
        for result in results:
            text = f"{result['symbol']} - {result['name']} ({result['exchange']})"
            keyboard.append([InlineKeyboardButton(text, callback_data=f"symbol_{result['symbol']}")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "Select a stock from the results:",
            reply_markup=reply_markup
        )
        return SET_PRICE

    async def handle_symbol_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle the selected stock symbol"""
        query = update.callback_query
        await query.answer()
        
        symbol = query.data.replace("symbol_", "")
        chat_id = query.message.chat_id
        
        self.user_temp_data[chat_id] = {'symbol': symbol}

        try:
            stock = yf.Ticker(symbol)
            current_price = stock.history(period="1m")['Close'].iloc[-1]
            await query.message.reply_text(
                f"Current price of {symbol} is ${current_price:.2f}\n"
                f"Please enter your desired alert price:"
            )
        except Exception as e:
            await query.message.reply_text(
                f"Selected symbol: {symbol}\n"
                f"Please enter your desired alert price:"
            )
        
        return SET_PRICE

    async def handle_price_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle the price input"""
        chat_id = update.message.chat_id
        
        try:
            price_threshold = float(update.message.text)
        except ValueError:
            await update.message.reply_text(
                "Please enter a valid number for the price alert:"
            )
            return SET_PRICE

        if chat_id not in self.user_temp_data:
            await update.message.reply_text(
                "Something went wrong. Please start over with /setalert"
            )
            return ConversationHandler.END

        symbol = self.user_temp_data[chat_id]['symbol']

        if chat_id not in self.user_alerts:
            self.user_alerts[chat_id] = []

        for alert in self.user_alerts[chat_id]:
            if alert[0] == symbol:
                await update.message.reply_text(
                    f"Alert already exists for {symbol}. Use /updatealert to modify."
                )
                del self.user_temp_data[chat_id]
                return ConversationHandler.END

        self.user_alerts[chat_id].append((symbol, price_threshold))
        self.save_alerts()
        
        await update.message.reply_text(
            f"Alert set for {symbol} at price ${price_threshold:.2f}"
        )
        logger.info(f"Alert set for {symbol} by user {chat_id}")
        
        del self.user_temp_data[chat_id]
        return ConversationHandler.END

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Cancel the conversation"""
        await update.message.reply_text('Operation cancelled.')
        return ConversationHandler.END

    async def update_alert(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /updatealert command"""
        if not context.args or len(context.args) != 2:
            await update.message.reply_text("Usage: /updatealert <stock_symbol> <new_price_threshold>")
            return

        chat_id = update.effective_chat.id
        stock_symbol = context.args[0].upper()
        
        try:
            new_threshold = float(context.args[1])
        except ValueError:
            await update.message.reply_text("Price must be a valid number!")
            return

        if chat_id not in self.user_alerts:
            await update.message.reply_text("You don't have any alerts set.")
            return

        for i, (symbol, _) in enumerate(self.user_alerts[chat_id]):
            if symbol == stock_symbol:
                self.user_alerts[chat_id][i] = (stock_symbol, new_threshold)
                self.save_alerts()
                await update.message.reply_text(
                    f"Alert for {stock_symbol} updated to ${new_threshold:.2f}"
                )
                logger.info(f"Alert updated for {stock_symbol} by user {chat_id}")
                return

        await update.message.reply_text(f"No alert found for {stock_symbol}")

    async def delete_alert(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /deletealert command"""
        if not context.args:
            await update.message.reply_text("Usage: /deletealert <stock_symbol>")
            return

        chat_id = update.effective_chat.id
        stock_symbol = context.args[0].upper()

        if chat_id in self.user_alerts:
            original_length = len(self.user_alerts[chat_id])
            self.user_alerts[chat_id] = [
                alert for alert in self.user_alerts[chat_id] 
                if alert[0] != stock_symbol
            ]
            
            if len(self.user_alerts[chat_id]) < original_length:
                self.save_alerts()
                await update.message.reply_text(f"Alert for {stock_symbol} deleted")
                logger.info(f"Alert deleted for {stock_symbol} by user {chat_id}")
            else:
                await update.message.reply_text(f"No alert found for {stock_symbol}")
        else:
            await update.message.reply_text("You don't have any alerts set")

    async def my_alerts(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /myalerts command"""
        chat_id = update.effective_chat.id
        
        if chat_id not in self.user_alerts or not self.user_alerts[chat_id]:
            await update.message.reply_text("You don't have any active alerts")
            return

        alert_text = "Your active alerts:\n\n"
        for symbol, price in self.user_alerts[chat_id]:
            try:
                stock = yf.Ticker(symbol)
                current_price = stock.history(period="1m")['Close'].iloc[-1]
                alert_text += f"• {symbol}: Target ${price:.2f} (Current: ${current_price:.2f})\n"
            except Exception:
                alert_text += f"• {symbol}: Target ${price:.2f} (Current price unavailable)\n"
        
        await update.message.reply_text(alert_text)
        logger.info(f"Alerts listed for user {chat_id}")

    async def check_alerts(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Check stock prices and send alerts"""
        for chat_id, alerts in list(self.user_alerts.items()):
            alerts_to_remove = []
            
            for stock_symbol, price_threshold in alerts:
                try:
                    stock = yf.Ticker(stock_symbol)
                    current_price = stock.history(period="1m")['Close'].iloc[-1]
                    
                    if current_price >= price_threshold:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=f"🚨 Alert: {stock_symbol} has reached ${current_price:.2f} "
                                f"(threshold: ${price_threshold:.2f})"
                        )
                        alerts_to_remove.append((stock_symbol, price_threshold))
                        logger.info(f"Alert triggered for {stock_symbol} for user {chat_id}")
                
                except Exception as e:
                    logger.error(f"Error checking price for {stock_symbol}: {e}")
                    continue

            for alert in alerts_to_remove:
                self.user_alerts[chat_id].remove(alert)
            
            if alerts_to_remove:
                self.save_alerts()

def main():
    """Main function to run the bot"""
    logger.info("Starting bot...")
    
    bot = StockAlertBot()
    application = Application.builder().token(BOT_TOKEN).build()

    setalert_handler = ConversationHandler(
        entry_points=[CommandHandler('setalert', bot.setalert_start)],
        states={
            SEARCH_SYMBOL: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_symbol_search)
            ],
            SET_PRICE: [
                CallbackQueryHandler(bot.handle_symbol_selection, pattern=r'^symbol_'),
                MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_price_input)
            ],
        },
        fallbacks=[CommandHandler('cancel', bot.cancel)]
    )

    subscribe_handler = ConversationHandler(
        entry_points=[CommandHandler('subscribe', bot.subscribe_start)],
        states={
            SUBSCRIBE_SYMBOL: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_subscribe_search)
            ],
        },
        fallbacks=[CommandHandler('cancel', bot.cancel)]
    )

    unsubscribe_handler = ConversationHandler(
        entry_points=[CommandHandler('unsubscribe', bot.unsubscribe_start)],
        states={
            UNSUBSCRIBE_SYMBOL: [
                CallbackQueryHandler(bot.handle_unsubscribe_selection, pattern=r'^unsub_')
            ],
        },
        fallbacks=[CommandHandler('cancel', bot.cancel)]
    )

    application.add_handler(setalert_handler)
    application.add_handler(subscribe_handler)
    application.add_handler(unsubscribe_handler)
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.start))
    application.add_handler(CommandHandler("updatealert", bot.update_alert))
    application.add_handler(CommandHandler("deletealert", bot.delete_alert))
    application.add_handler(CommandHandler("myalerts", bot.my_alerts))
    

    application.add_handler(CallbackQueryHandler(
        bot.handle_subscription_selection, 
        pattern=r'^sub_'
    ))

    application.job_queue.run_repeating(
        bot.check_alerts,
        interval=ALERT_CHECK_INTERVAL,
        first=10
    )
    
    application.job_queue.run_repeating(
        bot.check_announcements,
        interval=ANNOUNCEMENT_CHECK_INTERVAL,
        first=15
    )

    logger.info("Bot started. Polling for updates...")
    application.run_polling()

if __name__ == "__main__":
    main()