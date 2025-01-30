import streamlit as st
import yfinance as yf
import pandas as pd
import schedule
import time
import ollama
from datetime import datetime, timedelta

# Streamlit UI Setup
st.title("AI Trading Advisor")
logtxtbox = st.empty()  # Placeholder for updating log timestamps
logtxt = datetime.now().strftime("%H:%M:%S")  # Start with current timestamp
logtxtbox.caption(logtxt)

# Fetching historical data for Apple (AAPL) and Dow Jones (DJI) for yesterday (1-minute intervals)
stock = yf.Ticker("BTC-USD")
dow_jones = yf.Ticker("^DJI")
data = stock.history(period="12mo", interval="1d")  # AAPL stock data
dow_data = dow_jones.history(period="12mo", interval="1d")  # Dow Jones data
#data.reset_index(inplace=True, drop=True)
#dow_data.reset_index(inplace=True, drop=True)

# Fetching historical data for Apple (AAPL) and Dow Jones (DJI) for the last 12 months
# data = yf.download("AAPL", period="12mo", interval="1d")
# dow_data = yf.download("^DJI", period="12mo", interval="1d")

# Global variables to store rolling data for analysis
rolling_window = pd.DataFrame()
dow_rolling_window = pd.DataFrame()

# Variables to track daily context
daily_high = float('-inf')  # Track the highest price of the day
daily_low = float('inf')  # Track the lowest price of the day
buying_momentum = 0  # Track accumulation of buying pressure
selling_momentum = 0  # Track accumulation of selling pressure

# Function to process a new stock update every minute
def process_stock_update():
    global rolling_window, data, dow_rolling_window, dow_data
    global daily_high, daily_low, buying_momentum, selling_momentum

    if not data.empty and not dow_data.empty:

        available_data = len(data)-1
        available_dow_data = len(dow_data)-1

        # Simulate receiving the latest data point for AAPL and Dow Jones
        update = data.iloc[available_data].to_frame().T  # Get the first row
        time_str = update.index[0].time()  # Extract timestamp
        logtxtbox.caption(time_str)  # Update UI with current time
        dow_update = dow_data.iloc[available_dow_data].to_frame().T

        # Remove processed row from dataset
        data = data.iloc[1:]
        dow_data = dow_data.iloc[1:]

        # Keep all the previous data for processing
        rolling_window = data
        dow_rolling_window = dow_data

        # Append new data points to rolling window for tracking recent history
        # rolling_window = pd.concat([rolling_window, update], ignore_index=False)
        # dow_rolling_window = pd.concat([dow_rolling_window, dow_update], ignore_index=False)

        # Update daily high and low
        daily_high = max(daily_high, update['High'].values[0])
        daily_low = min(daily_low, update['Low'].values[0])

        # Calculate price movement momentum
        if len(rolling_window) >= 2:
            price_change = update['Close'].values[0] - rolling_window['Close'].iloc[-2]
            if price_change > 0:
                buying_momentum += price_change
            else:
                selling_momentum += abs(price_change)
                
        # Limit rolling window to the last 12 months of data
        rolling_window = rolling_window.tail(365)  # 365 trading days in a year
        dow_rolling_window = dow_rolling_window.tail(365)

        # Perform analysis on new data
        calculate_insights(rolling_window, dow_rolling_window)


# Function to determine how long the market has been open
# def get_market_open_duration(window):
#     current_time = datetime.now().time()
#     market_start_time = datetime.strptime("09:30:00", "%H:%M:%S").time()
#     return (datetime.combine(datetime.today(), current_time) - datetime.combine(datetime.today(), market_start_time)).total_seconds() / 60  # Compute elapsed minutes


# Function to analyze stock trends and generate insights
def calculate_insights(window, dow_window):
    if len(window) >= 5:
        last_price = window['Close'].iloc[-1]
        rolling_avg = window['Close'].rolling(window=7).mean().iloc[-1]  # 7-day rolling average
        price_change = window['Close'].iloc[-1] - window['Close'].iloc[-2] if len(window) >= 2 else 0
        volume_change = window['Volume'].iloc[-1] - window['Volume'].iloc[-2] if len(window) >= 2 else 0

        dow_price_change = dow_window['Close'].iloc[-1] - dow_window['Close'].iloc[-2] if len(dow_window) >= 2 else 0
        dow_volume_change = dow_window['Volume'].iloc[-1] - dow_window['Volume'].iloc[-2] if len(dow_window) >= 2 else 0

        ema7 = window['Close'].ewm(span=7, adjust=False).mean().iloc[-1]  # Exponential Moving Average
        ema20 = window['Close'].ewm(span=20, adjust=False).mean().iloc[-1]  # Exponential Moving Average
        ema50 = window['Close'].ewm(span=50, adjust=False).mean().iloc[-1]  # Exponential Moving Average
        ema100 = window['Close'].ewm(span=100, adjust=False).mean().iloc[-1]  # Exponential Moving Average
        ema200 = window['Close'].ewm(span=200, adjust=False).mean().iloc[-1]  # Exponential Moving Average
        std = window['Close'].rolling(window=5).std().iloc[-1]  # Standard deviation for Bollinger Bands
        bollinger_upper = rolling_avg + (2 * std)
        bollinger_lower = rolling_avg - (2 * std)

        # Calculate Relative Strength Index (RSI)
        delta = window['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean().iloc[-1]
        avg_loss = loss.rolling(window=14, min_periods=1).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else float('nan')
        rsi = 100 - (100 / (1 + rs))

        dow_rolling_avg = dow_window['Close'].rolling(window=5).mean().iloc[-1]
        # market_open_duration = get_market_open_duration(window)

        # Calculate Ichimoku indicators
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
        nine_period_high = window['High'].rolling(window= 9).max().iloc[-1]
        nine_period_low = window['Low'].rolling(window= 9).min().iloc[-1]
        tenkan_sen = (nine_period_high + nine_period_low) /2
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
        period26_high = window['High'].rolling(window=26).max().iloc[-1]
        period26_low = window['Low'].rolling(window=26).min().iloc[-1]
        kijun_sen = (period26_high + period26_low) / 2
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
        # window['senkou_span_a'] = ((window['tenkan_sen'] + window['kijun_sen']) / 2).shift(26)
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
        # period52_high = window['High'].rolling(window=52).max()
        # period52_low = window['Low'].rolling(window=52).min()
        # window['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
        # The most current closing price plotted 26 time periods behind (optional)
        # window['chikou_span'] = window['Close'].shift(-26)

        # Print the calculated insights
        print("***")
        print(f"Last price: {last_price:.2f}")
        print(f"7-day Rolling Average: {rolling_avg:.2f}")
        print(f"EMA7: {ema7:.2f}, EMA20: {ema20:.2f}, EMA50: {ema50:.2f}, EMA100: {ema100:.2f}, EMA200: {ema200:.2f},")
        print(f"RSI: {rsi:.2f}")
        print(f"Bollinger Upper Band: {bollinger_upper:.2f}, Lower Band: {bollinger_lower:.2f}")
        print(f"Price Change: {price_change:.2f}")
        print(f"Volume Change: {volume_change}")
        print(f"DOW Price Change: {dow_price_change:.2f}")
        print(f"DOW Volume Change: {dow_volume_change}")
        print(f"Dow Jones 7-day Rolling Average: {dow_rolling_avg:.2f}")
        print(f"Daily High: {daily_high:.2f}, Daily Low: {daily_low:.2f}")
        print(f"Buying Momentum: {buying_momentum:.2f}, Selling Momentum: {selling_momentum:.2f}")
        print(f"Tenkan Sen: {tenkan_sen:.2f}")
        print(f"Kijun Sen: {kijun_sen:.2f}")
        # print(f"Market has been open for {market_open_duration:.2f} minutes")
        print("***")
        
        # Generate natural language insights every 5 minutes based on the system clock
        current_minute = datetime.now().minute
        if current_minute % 5 == 0:
            get_natural_language_insights(
                rolling_avg, ema7, ema20, ema50, ema100, ema200, tenkan_sen, kijun_sen, rsi, bollinger_upper, bollinger_lower,
                price_change, volume_change, dow_rolling_avg,
                dow_price_change, dow_volume_change, daily_high, daily_low, buying_momentum, selling_momentum,
                window.index[-1].time().strftime("%H:%M:%S")
            )

# Function to generate stock insights using an LLM
def get_natural_language_insights(
    rolling_avg, ema7, ema20, ema50, ema100, ema200, tenkan_sen, kijun_sen, rsi, bollinger_upper, bollinger_lower,
    price_change, volume_change, dow_rolling_avg, dow_price_change, dow_volume_change, daily_high, daily_low, buying_momentum, selling_momentum, timestamp

):
    prompt = f"""
    You are a professional crypto trader with advanced skills in Technical Analysis and Ichimoku methodology. 
    Bitcoin's price has a 7-day rolling average of {rolling_avg:.2f}.
    The Exponential Moving Average are EMA7: {ema7:.2f}, EMA20: {ema20:.2f}, EMA50: {ema50:.2f}, EMA100: {ema100:.2f}, EMA200: {ema200:.2f}. 
    The Relative Strength Index (RSI) is {rsi:.2f}.
    The Bollinger Bands are set with an upper band of {bollinger_upper:.2f} and a lower band of {bollinger_lower:.2f}.
    The price has changed by {price_change:.2f}, and the volume has shifted by {volume_change}.
    Ichimoku Indicators are : Tenkan Sen: {tenkan_sen:.2f}, Kijun Sen: {kijun_sen:.2f}.
    The DOW price has changed by {dow_price_change:.2f}, and the volume has shifted by {dow_volume_change}.
    Meanwhile, the Dow Jones index has a 7-day rolling average of {dow_rolling_avg:.2f}.
    Yesterday's high was {daily_high:.2f} and low was {daily_low:.2f}.
    The buying momentum is {buying_momentum:.2f} and selling momentum is {selling_momentum:.2f}.
    Based on this data, provide insights into the current crypto trend and the general market sentiment.
    Provide all the necessary information to decide if we should buy or sell or nothing at the moment.
    Your answer should be structured in a bullet-point mode. 
    The insights should not be longer than 250 words and should not have an introduction. 
    """
    response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
    response_text = response['message']['content'].strip()
    message = st.chat_message("assistant")
    message.write(timestamp)
    message.write(response_text)
    print("Natural Language Insight:", response_text)

# Schedule job to simulate receiving updates 6000 (10) seconds
schedule.every(60).seconds.do(process_stock_update)  


message = st.chat_message("assistant")
message.write("Starting real-time simulation for BTC token updates. First update will be processed in 5 minutes...")    
# Run the scheduled jobs
print("Starting real-time simulation for BTC token updates...")
while True:
    schedule.run_pending()
    time.sleep(1)

