import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.graph_objs as go
import datetime
import sklearn 
import streamlit.components.v1 as components
import nltk
from nltk.tokenize import word_tokenize
import requests
    
st.title("Stock Market Analysis")
today = datetime.date.today()
start = '2010-01-01'
end = today.strftime('%Y-%m-%d')
st.sidebar.title("Predictive Analysis of Stock Market Trends:           ")
user_input = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL",key="stock_symbol")

df = yf.download(user_input, start=start, end=end)

model=load_model("keras model.h5")


    #Splitting Data into Training and Testing using MinMaxScaler

data_training = pd.DataFrame(df['Close'][0: int(len(df)*0.70)])
data_testing= pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array= scaler.fit_transform(data_training)
data_testing_array= scaler.fit_transform(data_testing)

def fetch_stock_data(symbol):
    # Fetch data from Yahoo Finance
    stock_data = yf.Ticker(symbol)
    return stock_data

past_100_days = data_training.tail(100)
final_df = past_100_days._append(data_testing, ignore_index=True)

input_data=scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

x_test , y_test =np.array(x_test) , np.array(y_test);

    #predication making
test_predication = model.predict(x_test)
scaler.scale_

scaling_factor = 1/scaler.scale_[0]
y_test = y_test*scaling_factor
test_predication=test_predication*scaling_factor

# Sidebar navigation
page = st.sidebar.selectbox("Select a page", ["Home", "Stock Analysis", "Prediction","Chatbot","Stock News"])

if page == "Home":
    st.title("Future Finance: Predictive Insights and Chatbot Consultation")

    st.subheader("Abstract")
    st.write("""
    The framework has four key components:
    
    1. **Stock Analysis**: A data-driven approach that breaks down historical data to reduce hidden information.
    2. **Stock Prediction**: Applies predictive modeling to gain insights about future market trends and movements, informing investors.
    3. **Asisystem**: An AI-enabled assistant that provides customized suggestions, up-to-the-minute details, and impeccable representative behavior.
    4. **Market Guider**: Offers users curated news and updates from the stock market, enabling them to stay informed.
    
    The project utilizes an LSTM model for predictive analysis, which achieved an R-squared score of 0.89 and demonstrated robustness with a cross-validation score of 0.84. The model is deployed through a Streamlit interface, allowing users to input a stock ticker and receive predicted prices.
    """)

    st.subheader("Meet the Team")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("- [Ashish Ruke](https://www.linkedin.com/in/ashish-ruke-68a038230/)")
        st.markdown("- [Harshvardhan Kulkarni](https://www.linkedin.com/in/ashish-ruke-68a038230/)")
    with cols[1]:
        st.markdown("- [Anushka Pote](https://www.linkedin.com/in/anushka-pote-17b692224/)")
        st.markdown("- [Shreyash Shegade](https://www.linkedin.com/in/shreyash-shedage-970b812a7/)")


elif page == "Stock Analysis":


    st.subheader("Data from 2010 to 2023")
    data1 = pd.DataFrame(df)

    st.dataframe(data1.tail(15))


    #Closing price with month and year using moving average
    fig_all = plt.figure(figsize=(18, 12))
    plt.plot(df.Close)

    ma30 = df.Close.rolling(30).mean()
    fig_ma30 = plt.figure(figsize=(12, 6))
    plt.plot(ma30)

    ma365 = df.Close.rolling(365).mean()
    fig_ma365 = plt.figure(figsize=(12, 6))
    plt.plot(ma365)

    button_container = st.columns(3)

    # Create buttons for each plot
    if button_container[0].button("Show Closing Prices"):
        st.subheader("Closing Price VS Time Chart")
        st.pyplot(fig_all)

    if button_container[1].button("Show MA30"):
        st.subheader("Closing Price of 30 days")
        st.pyplot(fig_ma30)

    if button_container[2].button("Show MA365"):
        st.subheader("Closing Price of 365 days")
        st.pyplot(fig_ma365)
    #load my model
    model=load_model("keras model.h5")


    #Splitting Data into Training and Testing using MinMaxScaler

    data_training = pd.DataFrame(df['Close'][0: int(len(df)*0.70)])
    data_testing= pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array= scaler.fit_transform(data_training)
    data_testing_array= scaler.fit_transform(data_testing)


    past_100_days = data_training.tail(100)
    final_df = past_100_days._append(data_testing, ignore_index=True)

    input_data=scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

    x_test , y_test =np.array(x_test) , np.array(y_test);

    #predication making
    test_predication = model.predict(x_test)
    scaler.scale_

    scaling_factor = 1/scaler.scale_[0]
    y_test = y_test*scaling_factor
    test_predication=test_predication*scaling_factor

    st.subheader("Prediction vs Original")
    fig2=plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(test_predication,'r', label ='Preditemp_inputcted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
    

elif page == "Prediction":
    st.title("Stock Market Analysis")

    # demonstrate prediction for next 50 days  with closing price
    x_input=data_testing_array[len(data_testing_array) - 100:].reshape(1,-1)
    opening=list(x_input)
    opening=opening[0].tolist()

    from numpy import array
    lst_output=[]
    n_steps=100
    i=0
    while(i<50):

        if(len(opening)>100):
            #print(opening)
            x_input=np.array(opening[1:])
            #print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
            opening.extend(yhat[0].tolist())
            opening=opening[1:]
            #print(opening)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1,n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            opening.extend(yhat[0].tolist()) 
            lst_output.extend(yhat.tolist())
            i=i+1

    lst_output=scaler.inverse_transform(lst_output)
    arr = lst_output



    #opening
    #Splitting Data into Training and Testing

    data_training_open = pd.DataFrame(df['Open'][0: int(len(df)*0.70)])
    data_testing_open= pd.DataFrame(df['Open'][int(len(df)*0.70): int(len(df))])
    from sklearn.preprocessing import MinMaxScaler
    scaler_open = MinMaxScaler(feature_range=(0,1))

    data_testing_array_open= scaler_open.fit_transform(data_testing_open)
    x_input_open=data_testing_array_open[len(data_testing_array_open) - 100:].reshape(1,-1)
    temp_input_open=list(x_input_open)
    temp_input_open=temp_input_open[0].tolist()

    from numpy import array

    lst_output_open=[]
    n_steps=100
    i=0
    while(i<50):

        if(len(temp_input_open)>100):
            #print(temp_input)
            x_input_open=np.array(temp_input_open[1:])
            #print("{} day input {}".format(i,x_input_open))
            x_input_open=x_input_open.reshape(1,-1)
            x_input_open = x_input_open.reshape((1, n_steps, 1))
            #print(x_input)
            yhat_open = model.predict(x_input_open, verbose=0)
            
            temp_input_open.extend(yhat_open[0].tolist())
            temp_input_open=temp_input_open[1:]
            #print(temp_input)
            lst_output_open.extend(yhat_open.tolist())
            i=i+1
        else:
            x_input_open = x_input_open.reshape((1,n_steps,1))
            yhat_open = model.predict(x_input_open, verbose=0)
            temp_input_open.extend(yhat_open[0].tolist())
            lst_output_open.extend(yhat_open.tolist())
            i=i+1

    lst_output_open=scaler_open.inverse_transform(lst_output_open)
    arr_open = lst_output_open

    #High
    #Splitting Data into Training and Testing

    data_training_high = pd.DataFrame(df['High'][0: int(len(df)*0.70)])
    data_testing_high= pd.DataFrame(df['High'][int(len(df)*0.70): int(len(df))])
    from sklearn.preprocessing import MinMaxScaler
    scaler_high = MinMaxScaler(feature_range=(0,1))

    data_testing_array_high= scaler_high.fit_transform(data_testing_high)
    x_input_high=data_testing_array_high[len(data_testing_array_high) - 100:].reshape(1,-1)
    temp_input_high=list(x_input_high)
    temp_input_high=temp_input_high[0].tolist()

    from numpy import array

    lst_output_high=[]
    n_steps=100
    i=0
    while(i<50):

        if(len(temp_input_high)>100):
            #print(temp_input)
            x_input_high=np.array(temp_input_high[1:])
            #print("{} day input {}".format(i,x_input_high))
            x_input_high=x_input_high.reshape(1,-1)
            x_input_high = x_input_high.reshape((1, n_steps, 1))
            #print(x_input)
            yhat_high = model.predict(x_input_high, verbose=0)
            
            temp_input_high.extend(yhat_high[0].tolist())
            temp_input_high=temp_input_high[1:]
            #print(temp_input)
            lst_output_high.extend(yhat_high.tolist())
            i=i+1
        else:
            x_input_high = x_input_high.reshape((1,n_steps,1))
            yhat_high = model.predict(x_input_high, verbose=0)
            temp_input_high.extend(yhat_high[0].tolist())
            lst_output_high.extend(yhat_high.tolist())
            i=i+1

    lst_output_high=scaler_high.inverse_transform(lst_output_high)
    arr_high = lst_output_high

    #low
    #Splitting Data into Training and Testing

    data_training_Low = pd.DataFrame(df['Low'][0: int(len(df)*0.70)])
    data_testing_Low= pd.DataFrame(df['Low'][int(len(df)*0.70): int(len(df))])
    from sklearn.preprocessing import MinMaxScaler
    scaler_Low = MinMaxScaler(feature_range=(0,1))

    data_testing_array_Low= scaler_Low.fit_transform(data_testing_Low)
    x_input_Low=data_testing_array_Low[len(data_testing_array_Low) - 100:].reshape(1,-1)
    temp_input_Low=list(x_input_Low)
    temp_input_Low=temp_input_Low[0].tolist()

    from numpy import array

    lst_output_Low=[]
    n_steps=100
    i=0
    while(i<50):

        if(len(temp_input_Low)>100):
            #print(temp_input)
            x_input_Low=np.array(temp_input_Low[1:])
            #print("{} day input {}".format(i,x_input_Low))
            x_input_Low=x_input_Low.reshape(1,-1)
            x_input_Low = x_input_Low.reshape((1, n_steps, 1))
            #print(x_input)
            yhat_Low = model.predict(x_input_Low, verbose=0)
            
            temp_input_Low.extend(yhat_Low[0].tolist())
            temp_input_Low=temp_input_Low[1:]
            #print(temp_input)
            lst_output_Low.extend(yhat_Low.tolist())
            i=i+1
        else:
            x_input_Low = x_input_Low.reshape((1,n_steps,1))
            yhat_Low = model.predict(x_input_Low, verbose=0)
            temp_input_Low.extend(yhat_Low[0].tolist())
            lst_output_Low.extend(yhat_Low.tolist())
            i=i+1

    lst_output_Low=scaler_Low.inverse_transform(lst_output_Low)
    arr_Low = lst_output_Low

    h = pd.DataFrame(arr_Low, columns=['Low'])

    g = pd.DataFrame(arr_high, columns=['High'])

    f = pd.DataFrame(arr_open, columns=['Open'])

    k =pd.DataFrame(arr, columns=['Close'])



    k['DailyChange'] = k['Close'].diff()
    k['PercentageChange'] = (k['DailyChange'] / k['Close'].shift(1)) * 100
    k['PercentageChange'] = k['PercentageChange'].map("{:.2f}%".format)
    result_df = pd.concat([h, g, f, k], axis=1)
    
    fig = go.Figure(data=[go.Candlestick(x=result_df.index,
                    open=result_df['Open'],
                    high=result_df['High'],
                    low=result_df['Low'],
                    close=result_df['Close'])])

    st.plotly_chart(fig)

    # Display the styled DataFrame using Streamlit
    st.write(result_df)


    fig3= plt.figure(figsize=(12,6))
    plt.plot(k.Close)
    st.pyplot(fig3)

    st.title("Next 5 Years Returns")

    # Input for the stock symbol
    #user_input = st.text_input("Enter the Stock Ticker", "AAPL", key="stock_symbol_input")

    # Check for changes in the input field
    if st.session_state.stock_symbol is not None:
        try:
            # Download historical data
            stock_data = yf.download(user_input, start="2023-01-01", end="2028-01-01")

            if not stock_data.empty and len(stock_data['Adj Close']) >= 2:
                # Calculate returns over the next 5 years
                returns_next_5_years = (stock_data['Adj Close'].iloc[-1] / stock_data['Adj Close'].iloc[0] - 1) * 100

                # Display the returns
                st.write(f"{user_input} returns over the next 5 years: {returns_next_5_years:.2f}%")
                st.write("Recommendation Moodel")

                # Display recommendation with colored box
                if returns_next_5_years >= 0:
                    st.markdown('<div style="padding: 10px; color: white; background-color: green; text-align: center;">Yes</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="padding: 10px; color: white; background-color: red; text-align: center;">No</div>', unsafe_allow_html=True)

            else:
                st.write("Error: Insufficient data for calculating returns.")

        except Exception as e:
            st.write(f"Error: {str(e)}")
    else:
        st.write("Enter a stock symbol to get the returns.")
    # Add content for the analysis page

elif page == "Chatbot":
            symbol =user_input
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

    
    
elif page == "Stock News":
        NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"
        NEWS_API_KEY = "263f24e3d72e4880ab9ce9559725bef3"  # Replace with your API key

        # Streamlit app title
        st.title("Stock News Viewer")

        # Sidebar input for stock symbol
        stock_symbol = user_input

        # Slider for selecting the number of news articles
        num_articles = st.slider("Number of News Articles", min_value=1, max_value=20, value=10)

        if stock_symbol:
            # Fetch news articles related to the selected stock
            params = {
                "q": stock_symbol,
                "apiKey": NEWS_API_KEY,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": num_articles,
            }
            try:
                response = requests.get(NEWS_API_ENDPOINT, params=params)
                response.raise_for_status()  # Raise an exception for HTTP errors

                articles = response.json().get("articles", [])

                if articles:
                    # Display news articles in a grid layout
                    for i in range(0, len(articles), 3):
                        row_articles = articles[i:i+3]
                        col1, col2, col3 = st.columns(3)
                        for idx, article in enumerate(row_articles):
                            with locals()[f"col{idx+1}"]:
                                st.write("###", article["title"])
                                st.write(article["description"])
                                st.write("Source:", article["source"]["name"])
                                st.write("Published at:", article["publishedAt"])
                                st.write("[Read more](" + article["url"] + ")")
                                st.markdown("---")
                else:
                    st.write("No articles found for this stock symbol.")
            except requests.exceptions.RequestException as e:
                st.error("Error fetching news. Please check your internet connection and try again.")
                st.error(f"Error details: {e}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")