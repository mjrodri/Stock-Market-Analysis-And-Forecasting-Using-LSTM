import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tkinter as tk
from tkinter import ttk
import os

# Set TF_ENABLE_ONEDNN_OPTS to 0
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Your existing code here

class StockPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Prediction App")

        # Variables
        self.symbol_var = tk.StringVar()
        self.start_date_var = tk.StringVar()
        self.end_date_var = tk.StringVar()

        # GUI Elements
        ttk.Label(root, text="Stock Symbol:").grid(row=0, column=0, padx=10, pady=5)
        ttk.Entry(root, textvariable=self.symbol_var).grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(root, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=10, pady=5)
        ttk.Entry(root, textvariable=self.start_date_var).grid(row=1, column=1, padx=10, pady=5)

        ttk.Label(root, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=10, pady=5)
        ttk.Entry(root, textvariable=self.end_date_var).grid(row=2, column=1, padx=10, pady=5)

        ttk.Button(root, text="Predict", command=self.predict_stock).grid(row=3, column=0, columnspan=2, pady=10)

    def predict_stock(self):
        # Get input values
        symbol = self.symbol_var.get()
        start_date = self.start_date_var.get()
        end_date = self.end_date_var.get()

        # Retrieve stock data
        stock_data = self.get_stock_data(symbol, start_date, end_date)

        # Data Preprocessing
        scaled_data = self.preprocess_data(stock_data)

        # Prepare data for LSTM
        X_train, y_train = self.prepare_lstm_data(scaled_data)

        # Build and train LSTM model
        model = self.build_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=10, batch_size=32)

        # Forecasting
        input_data = scaled_data[-60:]
        input_data = np.reshape(input_data, (1, input_data.shape[0], 1))
        predicted_price = model.predict(input_data)

        # Visualization of Predictions
        self.plot_predictions(stock_data, predicted_price)

    def get_stock_data(self, symbol, start_date, end_date):
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        return stock_data['Close']

    def preprocess_data(self, data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
        return scaled_data

    def prepare_lstm_data(self, scaled_data):
        X_train, y_train = [], []
        for i in range(60, len(scaled_data)):
            X_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])
        return np.array(X_train), np.array(y_train)

    def build_lstm_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def plot_predictions(self, stock_data, predicted_price):
        plt.figure(figsize=(10, 6))
        plt.plot(stock_data, label='Actual Stock Price')
        plt.plot(len(stock_data), predicted_price, marker='o', markersize=8, label='Predicted Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Normalized Stock Price')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictionApp(root)
    root.mainloop()