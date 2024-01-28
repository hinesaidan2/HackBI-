
import yfinance as yf
#import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, jsonify







def download_stock_data(ticker):
    """Download historical stock data from Yahoo Finance. If the stock doesn't exist, print a message."""

    data = yf.download(ticker, start='2010-01-01')
    return data

def generate_features(data):
    """Generate technical indicators as features."""
    data['Return'] = data['Adj Close'].pct_change()
    data['MA50'] = data['Adj Close'].rolling(window=50).mean()
    data['MA200'] = data['Adj Close'].rolling(window=200).mean()
    data['Volume_Change'] = data['Volume'].pct_change()
    data.dropna(inplace=True)
    return data

def prepare_data(data):
    """Prepare data for modeling, including feature/target creation."""
    data['Target'] = (data['Return'].shift(-1) > 0).astype(int)
    features = ['Return', 'MA50', 'MA200', 'Volume_Change']
    X = data[features]
    y = data['Target']
    return train_test_split(X, y, test_size=0.2, shuffle=False)

def train_model(X_train, y_train):
    """Train a machine learning model."""
    #scalar helps standardize data so all prices are 0s and 1s
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler #returns scalar that can be used later to predict

def evaluate_model(model, scaler, X_test, y_test):
    """Evaluate the model's performance."""
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy:.2%}')
    return predictions


def predict_future_rises(model, scaler, data, features):
    """Predict future stock price rises."""
    # Use the most recent data to predict the future
    future_data = data.tail(1)[features]
    future_data_scaled = scaler.transform(future_data)
    prediction = model.predict(future_data_scaled)
    if prediction == 1: #returns 1 for a rise pred in next few days
       return(1)
    else:   #returns 0 for decline
       return(0)


# Main function to run the script
#combines all the functions returns 0 if negative expected and 1 if positive
def main1(stock_ticker):

    data = download_stock_data(stock_ticker)
    data_with_features = generate_features(data)
    X_train, X_test, y_train, y_test = prepare_data(data_with_features)
    model, scaler = train_model(X_train, y_train)
    evaluate_model(model, scaler, X_test, y_test)
    if(predict_future_rises(model, scaler, data_with_features, ['Return', 'MA50', 'MA200', 'Volume_Change'])==0):

        return(0)
    else:
        return(1)




#stock_ticker = input("Enter stock ticker: ")
#main1(stock_ticker)





