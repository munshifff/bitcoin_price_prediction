import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import gradio as gr
import io
import base64

plt.style.use('fivethirtyeight')

def prepare_data():
    end = datetime.now()
    start = datetime(end.year-15, end.month, end.day)
    stock = 'BTC-USD'
    stock_data = yf.download(stock, start=start, end=end)
    
    # Close price data
    closing_price = stock_data[['Close']]
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_price[['Close']].dropna())
    
    return closing_price, scaler, scaled_data

def create_model():
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(100, 1)),
        LSTM(64, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_model():
    closing_price, scaler, scaled_data = prepare_data()
    
    # Prepare data for LSTM
    x_data = []
    y_data = []
    base_days = 100
    for i in range(base_days, len(scaled_data)):
        x_data.append(scaled_data[i-base_days:i])
        y_data.append(scaled_data[i])
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    # Split into train and test sets
    train_size = int(len(x_data) * 0.9)
    x_train, y_train = x_data[:train_size], y_data[:train_size]
    
    model = create_model()
    model.fit(x_train, y_train, batch_size=5, epochs=10)
    
    return model, scaler, scaled_data, closing_price, base_days, train_size

def predict_future(days_to_predict):
    model, scaler, scaled_data, closing_price, base_days, train_size = train_model()
    
    # Prepare the plot for historical predictions
    x_test = np.array([scaled_data[i-base_days:i] for i in range(base_days, len(scaled_data))])[train_size:]
    y_test = np.array([scaled_data[i] for i in range(base_days, len(scaled_data))])[train_size:]
    
    predictions = model.predict(x_test)
    inv_predictions = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_test)
    
    plotting_data = pd.DataFrame(
        {
            'Original': inv_y_test.flatten(), 
            'Prediction': inv_predictions.flatten(),
        }, index=closing_price.index[train_size + base_days:]
    )
    
    # Predict future days
    last_100 = scaled_data[-100:].reshape(1, -1, 1)
    future_predictions = []
    for _ in range(days_to_predict):
        next_day = model.predict(last_100)
        future_predictions.append(scaler.inverse_transform(next_day)[0][0])
        last_100 = np.append(last_100[:, 1:, :], next_day.reshape(1, 1, -1), axis=1)
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.plot(plotting_data.index, plotting_data['Original'], label='Historical Actual', color='blue', linewidth=2)
    plt.plot(plotting_data.index, plotting_data['Prediction'], label='Historical Predictions', color='red', linewidth=2)
    
    # Plot future predictions
    future_dates = pd.date_range(start=plotting_data.index[-1], periods=days_to_predict+1)[1:]
    plt.plot(future_dates, future_predictions, 'g--o', label=f'Next {days_to_predict} Days Prediction', linewidth=2)
    
    # Add value labels for future predictions
    for i, (date, val) in enumerate(zip(future_dates, future_predictions)):
        plt.text(date, val, f'{val:.2f}', fontsize=10, ha='center', va='bottom', color='black')
    
    plt.title("Bitcoin Price Prediction with LSTM", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel('Price (USD)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    
    # Convert plot to HTML image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f"<img src='data:image/png;base64,{encoded}'/>"

# Gradio UI
iface = gr.Interface(
    fn=predict_future,
    inputs=gr.Slider(1, 30, step=1, value=10, label="Days to Predict"),
    outputs="html",
    title="Bitcoin Price Predictor with LSTM",
    description="Predict future Bitcoin prices using LSTM neural network"
)

iface.launch()