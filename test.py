# Import required libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Step 1: Fetch Time Series Data


def fetch_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Step 2: Preprocessing


def preprocess_data(data):
    """
    Preprocess the data by setting the index and selecting the 'Close' prices.
    """
    data.index = pd.to_datetime(data.index)
    ts = data['Close']
    return ts

# Step 3: ARIMA Forecasting


def forecast_with_arima(ts, order, forecast_periods):
    """
    Forecast using ARIMA model.

    Args:
        ts (pd.Series): Time series data.
        order (tuple): ARIMA (p, d, q) order.
        forecast_periods (int): Number of periods to forecast.

    Returns:
        pd.Series: Forecasted values.
    """
    model = ARIMA(ts, order=order)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(forecast_periods)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ts, label="Historical")
    plt.plot(forecast, label="ARIMA Forecast", color="orange")
    plt.title("ARIMA Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()
    return forecast

# Step 4: PyTorch LSTM Model


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def prepare_lstm_data(ts, sequence_length):
    """
    Prepare time series data for LSTM.
    """
    scaler = MinMaxScaler()
    ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1))

    data = []
    for i in range(len(ts_scaled) - sequence_length):
        data.append(ts_scaled[i:i + sequence_length])

    data = np.array(data)
    X, y = data[:, :-1], data[:, -1]
    return train_test_split(X, y, test_size=0.2, shuffle=False), scaler


def train_lstm_model(X_train, y_train, input_size=1, hidden_size=50, num_layers=1, epochs=50, lr=0.001):
    """
    Train LSTM model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM(input_size=input_size, hidden_size=hidden_size,
                 num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return model


def forecast_with_lstm(model, X_test, scaler):
    """
    Forecast using the trained LSTM model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model(X_test).cpu().numpy()

    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions)
    return predictions


# Main Pipeline
if __name__ == "__main__":
    # Parameters
    TICKER = "AAPL"
    START_DATE = "2018-01-01"
    END_DATE = "2023-01-01"
    ARIMA_ORDER = (5, 1, 0)
    FORECAST_PERIODS = 30
    SEQUENCE_LENGTH = 50
    EPOCHS = 50
    LR = 0.001

    # Step 1: Fetch Data
    data = fetch_data(TICKER, START_DATE, END_DATE)

    # Step 2: Preprocess Data
    ts = preprocess_data(data)

    # Step 3: ARIMA Forecasting
    arima_forecast = forecast_with_arima(ts, ARIMA_ORDER, FORECAST_PERIODS)

    # Step 4: LSTM Forecasting
    (X_train, X_test, y_train, y_test), scaler = prepare_lstm_data(ts, SEQUENCE_LENGTH)
    lstm_model = train_lstm_model(X_train, y_train, epochs=EPOCHS, lr=LR)
    lstm_forecast = forecast_with_lstm(lstm_model, X_test, scaler)

    # Plot LSTM Forecast
    plt.figure(figsize=(10, 6))
    plt.plot(ts.index[-len(y_test):], scaler.inverse_transform(
        y_test.reshape(-1, 1)), label="Actual", color="blue")
    plt.plot(ts.index[-len(y_test):], lstm_forecast,
             label="LSTM Forecast", color="green")
    plt.title("LSTM Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()
