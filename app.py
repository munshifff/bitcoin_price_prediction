import gradio as gr
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import io
import base64

def train_and_predict(days):
    # Load data
    df = yf.download("BTC-USD", period="max")
    df['Close'] = df['Close'].fillna(method='ffill')

    # Feature engineering
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=30).std()

    for i in range(1, 8):
        df[f'Lag_{i}'] = df['Close'].shift(i)

    df.dropna(inplace=True)
    X = df.drop(['Close'], axis=1)
    y = df['Close']

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Forecast future (naive loop)
    future_inputs = X_test[-1:].values
    future_preds = []
    for _ in range(days):
        pred = model.predict(future_inputs)[0]
        future_preds.append(pred)
        # roll inputs
        future_inputs = np.roll(future_inputs, -1)
        future_inputs[0, -1] = pred

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(predictions)), predictions, label='Predicted')
    plt.title(f'Bitcoin Price Forecast for Next {days} Days\nRMSE: {rmse:.2f}')
    plt.xlabel('Days')
    plt.ylabel('BTC Price')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"<img src='data:image/png;base64,{encoded}'/>"

# Gradio UI
iface = gr.Interface(
    fn=train_and_predict,
    inputs=gr.Slider(1, 30, step=1, label="Days to Predict"),
    outputs="html",
    title="Bitcoin Price Predictor",
    description="Predict future Bitcoin prices using XGBoost + technical indicators"
)

iface.launch()
