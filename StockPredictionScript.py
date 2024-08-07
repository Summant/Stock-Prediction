import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to fetch stock data
def fetch_stock_data(ticker, period='1y'):
    stock_data = yf.download(ticker, period=period)
    return stock_data

# Function to prepare data for model
def prepare_data(stock_data, prediction_days=30):
    stock_data['Prediction'] = stock_data['Close'].shift(-prediction_days)
    X = np.array(stock_data.drop(['Prediction'], axis=1))
    X = X[:-prediction_days]
    y = np.array(stock_data['Prediction'])
    y = y[:-prediction_days]
    return X, y

# Function to train and predict stock prices
def predict_stock_prices(ticker, prediction_days=30):
    stock_data = fetch_stock_data(ticker)
    X, y = prepare_data(stock_data, prediction_days)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    future_prices = stock_data['Close'][-prediction_days:]
    future_dates = pd.date_range(start=future_prices.index[-1], periods=prediction_days + 1)[1:]
    future_predictions = model.predict(np.array(stock_data.drop(['Prediction'], axis=1))[-prediction_days:])

    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'], label='Actual Prices')
    plt.plot(future_dates, future_predictions, label='Predicted Prices', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.legend()
    plt.show()

# Example usage
ticker = 'AAPL'
prediction_days = 30  # Change this value for different prediction periods
predict_stock_prices(ticker, prediction_days)
