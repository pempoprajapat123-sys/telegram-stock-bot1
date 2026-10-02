import os
import requests
import yfinance as yf
import pandas as pd
import nltk
import feedparser
from datetime import datetime, timezone
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER Sentiment Engine
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# Load Secrets from GitHub Environment
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

# 1. CORE WATCHLIST (13 Tickers)
CORE_WATCHLIST = [
    "LITE", "RKLB", "SNDK", "MSFT", "GOOGL",
    "AMZN", "NVDA", "PLTR", "CLS", "SHOP", "RY", "NOK", "AEM"
]

# 2. ACTIVE SWING WATCHLIST (9 Tickers)
ACTIVE_SWING_WATCHLIST = [
    "INTC", "SPCX", "IOVA", "NU", "CCL", "AAL", "ONDS", "WBD", "PCG"
]

# Designated High-Beta / High-Volatility Tickers requiring wider 2.2x ATR stop-losses
HIGH_BETA_TICKERS = ["RKLB", "NVDA", "INTC", "IOVA", "TSLA", "SMCI", "MARA", "SPCX", "ONDS"]

def send_telegram_alert(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("\n -> Telegram Error: BOT_TOKEN or CHAT_ID environment variables missing.")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        res = requests.post(url, data=payload)
        if res.status_code == 200:
            print("\n -> Telegram Alert Sent Successfully! ✅")
        else:
            print(f"\n -> Telegram Error ({res.status_code}): {res.text}")
    except Exception as e:
        print(f"\n -> Telegram Exception: {e}")

def get_news_sentiment(symbol, ticker_obj):
    headlines = []
    try:
        news_items = ticker_obj.news
        if news_items:
            for item in news_items[:5]:
                title = item.get('title', '')
                if title:
                    headlines.append(title)
    except Exception:
        pass

    if not headlines:
        try:
            rss_url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:5]:
                headlines.append(entry.title)
        except Exception:
            pass

    if not headlines:
        return 0.0, "🟡 NEUTRAL"

    scores = [sia.polarity_scores(text)['compound'] for text in headlines]
    avg_score = sum(scores) / len(scores)

    if avg_score >= 0.10:
        return avg_score, "🟢 BULLISH"
    elif avg_score <= -0.10:
        return avg_score, "🔴 BEARISH"
    else:
        return avg_score, "🟡 NEUTRAL"

def check_vix_risk():
    try:
        vix = yf.Ticker("^VIX").history(period="5d")
        if not vix.empty:
            vix_val = vix['Close'].iloc[-1]
            if vix_val > 25:
                return vix_val, "🔴 HIGH FEAR (VIX &gt; 25)"
            elif vix_val > 18:
                return vix_val, "🟡 MODERATE RISK (VIX 18-25)"
            else:
                return vix_val, "🟢 LOW RISK (VIX &lt; 18)"
    except Exception:
        pass
    return 20.0, "🟡 UNKNOWN VOLATILITY"

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_macd(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def check_market_trend():
    try:
        spy = yf.Ticker("SPY").history(period="1mo", interval="5m")
        if not spy.empty:
            spy['SMA_50'] = spy['Close'].rolling(window=50).mean()
            latest = spy.iloc[-1]
            if latest['Close'] > latest['SMA_50']:
                return "🟢 BULLISH (Safe Market)", "BUY MODE"
            else:
                return "🔴 BEARISH (Market Falling)", "HOLD CASH / WAIT"
    except Exception:
        pass
    return "🟡 UNKNOWN", "CAUTION"

def calculate_dynamic_atr_levels(symbol, price, atr_val, sma_50):
    if pd.isna(atr_val) or atr_val <= 0:
        atr_val = price * 0.02

    if symbol in HIGH_BETA_TICKERS:
        stop_mult = 2.2
        target_mult = 4.5
    else:
        stop_mult = 1.5
        target_mult = 3.0

    raw_target = price + (target_mult * atr_val)
    raw_stop = price - (stop_mult * atr_val)

    if sma_50 < price and raw_stop > (sma_50 * 0.98):
        final_stop = max(0.01, sma_50 * 0.985)
    else:
        final_stop = max(0.01, raw_stop)

    return raw_target, final_stop

def scan_watchlist_group(symbols, phase, market_action, vix_val):
    rows = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1mo", interval="5m")
            if df.empty or len(df) < 50:
                print(f"Skipping {symbol}: Insufficient price data.")
                continue

            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['ATR'] = calculate_atr(df)

            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            macd, signal = calculate_macd(df)
            df['MACD'] = macd
            df['MACD_Signal'] = signal

            latest = df.iloc[-1]
            price = latest['Close']
            rsi = latest['RSI']
            sma = latest['SMA_50']
            atr = latest['ATR']
            avg_volume = df['Volume'].mean()

            sent_score, sent_label = get_news_sentiment(symbol, ticker)
            target_price, stop_price = calculate_dynamic_atr_levels(symbol, price, atr, sma)

            bull_score = 0
            if price > sma: bull_score += 1
            if 50 <= rsi <= 70: bull_score += 1
            if latest['Volume'] > (avg_volume * 1.2): bull_score += 1
            if latest['MACD'] > latest['MACD_Signal']: bull_score += 1

            if phase == "PREMARKET":
                prediction = f"RANGE WATCH: ${stop_price:.2f} - ${target_price:.2f}"
                action = "PREPARE WATCHLIST / WAIT FOR OPEN"
            elif phase == "POST_OPEN":
                if bull_score >= 3:
                    prediction = "🟢 CONFIRMED UPWARD BREAKOUT"
                    action = "BUY NOW / ENTER LONG POSITION"
                elif price < sma:
                    prediction = "🔴 OPENING SLIDE (PRICE DROP RISK)"
                    action = "AVOID BUYING / EXPECT LOWER DIPS"
                else:
                    prediction = "🟡 CONSOLIDATION AT OPEN"
                    action = "WAIT FOR DIRECTIONAL BREAKOUT"
            elif phase == "BEFORE_CLOSE":
                if price > sma and latest['MACD'] > latest['MACD_Signal']:
                    prediction = "🟢 STRONG CLOSE EXPECTED (BULLISH OVERNIGHT)"
                    action = "HOLD OVERNIGHT / TAKE PROFIT AT TARGET"
                elif price < sma:
                    prediction = "🔴 WEAK CLOSE EXPECTED (DOWNTREND RISK)"
                    action = "SELL / EXIT BEFORE CLOSE / DO NOT HOLD OVERNIGHT"
                else:
                    prediction = "🟡 NEUTRAL CLOSE"
                    action = "HOLD CASH / CLOSE SWING TRADES"
            else:
                prediction = "INTRADAY MONITORING"
                action = "BUY NOW" if bull_score >= 3 else "HOLD CASH"

            tv_link = f"https://www.tradingview.com/chart/?symbol={symbol}"

            row_entry = (
                f"<b>{symbol}</b> (<a href='{tv_link}'>Chart</a>)\n"
                f"💵 <b>Current:</b> <code>${price:.2f}</code> | 🎯 <b>Target:</b> <code>${target_price:.2f}</code> | 🛑 <b>Stop:</b> <code>${stop_price:.2f}</code>\n"
                f"🔮 <b>Predict:</b> {prediction}\n"
                f"💡 <b>Action:</b> <b>{action}</b> | 📰 <b>News:</b> {sent_label}"
            )
            rows.append(row_entry)

        except Exception as err:
            print(f"Error scanning {symbol}: {err}")
    return rows

def run_master_scheduled_scan():
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    minute = now_utc.minute

    if hour == 13 and minute < 20:
        phase = "PREMARKET"
        phase_title = "🌅 PRE-MARKET REPORT (9:00 AM EST)"
    elif hour == 14 and minute < 20:
        phase = "POST_OPEN"
        phase_title = "⚡ 30-MIN POST-OPEN OBSERVATION (10:00 AM EST)"
    elif hour == 19 and minute >= 20:
        phase = "BEFORE_CLOSE"
        phase_title = "🔔 30-MIN BEFORE CLOSE ALERT (3:30 PM EST)"
    else:
        phase = "INTRADAY"
        phase_title = "📊 INTRADAY SCANNER REPORT"

    market_text, market_action = check_market_trend()
    vix_val, vix_text = check_vix_risk()

    core_rows = scan_watchlist_group(CORE_WATCHLIST, phase, market_action, vix_val)
    swing_rows = scan_watchlist_group(ACTIVE_SWING_WATCHLIST, phase, market_action, vix_val)

    if core_rows:
        msg_core = f"<b>🔵 CORE WATCHLIST — {phase_title}</b>\n"
        msg_core += f"<b>Market Trend:</b> {market_text}\n"
        msg_core += f"<b>Volatility:</b> {vix_text}\n"
        msg_core += "─────────────────────────────\n\n"
        msg_core += "\n\n".join(core_rows)
        send_telegram_alert(msg_core)

    if swing_rows:
        msg_swing = f"<b>⚡ ACTIVE SWING WATCHLIST — {phase_title}</b>\n"
        msg_swing += f"<b>Market Trend:</b> {market_text}\n"
        msg_swing += f"<b>Volatility:</b> {vix_text}\n"
        msg_swing += "─────────────────────────────\n\n"
        msg_swing += "\n\n".join(swing_rows)
        send_telegram_alert(msg_swing)

if __name__ == "__main__":
    run_master_scheduled_scan()
