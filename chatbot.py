import streamlit as st
import pandas as pd
import yfinance as yf
import nltk
from nltk.tokenize import word_tokenize

page = st.sidebar.selectbox("Select a page", ["Home", "Stock Analysis", "Predection","Chattbot","Stock News"])

def fetch_stock_data(symbol):
    # Fetch data from Yahoo Finance
    stock_data = yf.Ticker(symbol)
    return stock_data

def main():
    st.title("Stock Chatbot")

    # User input for stock symbol
    symbol = st.text_input("Enter the stock symbol (e.g., AAPL):")

    if symbol:
        # Fetch historical data
        try:
            stock = fetch_stock_data(symbol)

            st.write(f"### {symbol} Stock Information")
            st.write(stock.info)

            # Chatbot interaction
            stock = fetch_stock_data(symbol)

            st.write(f"### {symbol} Stock Information")
            st.write(stock.info)

            # Tokenize the keys and values of stock.info
            info_tokens = {}
            for key, value in stock.info.items():
                key_tokens = word_tokenize(str(key).lower())
                value_tokens = word_tokenize(str(value).lower())
                info_tokens[key] = key_tokens + value_tokens

            # Fetch historical data for risk calculation
            stock_history = stock.history(period="1y")

            # Calculate daily returns
            stock_history['Daily Return'] = stock_history['Close'].pct_change()

            # Calculate risk (standard deviation of daily returns)
            risk_level = stock_history['Daily Return'].std()

            st.write(f"### Estimated Risk Level")
            st.write(f"The estimated risk level of {symbol} based on historical volatility is {risk_level:.2f}")

            # Chatbot interaction loop
            while True:
                st.write("### Ask a question (type 'bye' to exit):")
                question = st.text_input("Type here...")

                if question.lower() == 'bye':
                    break

                if st.button("Ask"):
                    found_answer = False

                    # Tokenize the user question
                    tokens = word_tokenize(question.lower())

                    # Check for relevant keywords in stock.info
                    for key, key_tokens in info_tokens.items():
                        if all(token in key_tokens for token in tokens):
                            st.write(f"{key.capitalize()}: {stock.info[key]}")
                            found_answer = True
                            break

                    if not found_answer:
                        # Check for relevant keywords in other questions
                        if 'closing' in tokens and 'price' in tokens:
                            st.write(f"The closing price of {symbol} is ${stock.history(period='1d')['Close'].iloc[-1]}")
                            found_answer = True

                        elif 'opening' in tokens and 'price' in tokens:
                            st.write(f"The opening price of {symbol} was ${stock.history(period='1d')['Open'].iloc[0]}")
                            found_answer = True

                        elif 'volume' in tokens:
                            st.write(f"The volume of {symbol} traded today is {stock.history(period='1d')['Volume'].iloc[-1]}")
                            found_answer = True

                        elif 'top' in tokens and 'gainer' in tokens:
                            top_gainer = stock.history(period='1d').nlargest(1, 'Close')
                            st.write(f"The top gainer today is {top_gainer.index[0]} with a closing price of ${top_gainer['Close'].iloc[0]}")
                            found_answer = True

                        elif 'top' in tokens and 'loser' in tokens:
                            top_loser = stock.history(period='1d').nsmallest(1, 'Close')
                            st.write(f"The top loser today is {top_loser.index[0]} with a closing price of ${top_loser['Close'].iloc[0]}")
                            found_answer = True

                        elif 'long' in tokens and 'term' in tokens and 'investment' in tokens:
                            if stock.info['dividendYield'] > 0.05:  # Example condition for considering long-term investment
                                st.write("Considering the high dividend yield, this stock might be suitable for long-term investment.")
                            else:
                                st.write("Based on current information, this stock might not be suitable for long-term investment.")
                            found_answer = True

                        elif 'risk' in tokens and 'level' in tokens:
                            st.write(f"The estimated risk level of {symbol} based on historical volatility is {risk_level:.2f}")
                            found_answer = True

                    if not found_answer:
                        st.write("I'm sorry, I don't understand that question.")

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
if __name__ == "__main__":
    main()
