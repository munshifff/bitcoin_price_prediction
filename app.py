import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

st.set_page_config(page_title="Bitcoin Price Predictor", layout="wide")
st.title("📈 Bitcoin Price Prediction with LSTM")

# --- Download BTC data ---
end = datetime.now()
start = datetime(end.year - 15, end.month, end.day)
stock = "BTC-USD"

stock_data = yf.download(stock, start=start, end=end)

# Fix MultiIndex columns if exist
if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = [col[0] for col in stock_data.columns]

st.subheader("Raw Bitcoin Data (last 10 rows)")
st.dataframe(stock_data.tail(10))

# --- Closing price ---
closing_price = stock_data[['Close']]

st.subheader("Closing Price Over Time")
st.line_chart(closing_price)

# --- Scaling ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_price.dropna())

# --- Prepare data for LSTM ---
x_data, y_data = [], []
base_days = 100
for i in range(base_days, len(scaled_data)):
    x_data.append(scaled_data[i-base_days:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# --- Train-test split ---
train_size = int(len(x_data) * 0.9)
x_train, y_train = x_data[:train_size], y_data[:train_size]
x_test, y_test = x_data[train_size:], y_data[train_size:]

# --- Build LSTM model ---
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(64, return_sequences=False),
    Dense(25),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
with st.spinner("Training LSTM model..."):
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
st.success("Model training completed!")

# --- Predictions ---
predictions = model.predict(x_test)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_test)

st.subheader("Prediction vs Actual Closing Price")
chart_data = pd.DataFrame({
    "Actual": inv_y_test.flatten(),
    "Predicted": inv_predictions.flatten()
}, index=closing_price.index[train_size + base_days:])
st.line_chart(chart_data)

# --- User input for future forecast ---
st.subheader("🔮 Future Price Prediction")
num_days = st.number_input("Enter number of days to forecast:", min_value=1, max_value=365, value=10, step=1)

last_100 = scaled_data[-100:].reshape(1, -1, 1)
future_preds = []

for _ in range(num_days):
    next_day = model.predict(last_100)
    future_price = scaler.inverse_transform(next_day)[0, 0]
    future_preds.append(future_price)
    last_100 = np.append(last_100[:, 1:, :], next_day.reshape(1, 1, 1), axis=1)

# --- Generate actual dates for future ---
last_date = stock_data.index[-1].normalize()  # remove time
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, num_days + 1)]
future_dates = [d.date() for d in future_dates]  # only date

future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": future_preds
})

future_df.set_index("Date", inplace=True)

st.line_chart(future_df)
st.table(future_df)
