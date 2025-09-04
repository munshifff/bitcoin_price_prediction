📈 Bitcoin Price Predictor with LSTM

This project predicts future Bitcoin (BTC-USD) prices using a Long Short-Term Memory (LSTM) neural network.
The model is trained on 15 years of historical Bitcoin price data fetched from Yahoo Finance (yfinance).
Predictions are visualized in an interactive Streamlit web interface.

Features

Fetches 15 years of historical Bitcoin price data

Preprocesses data using MinMax scaling

Trains an LSTM model to learn temporal price patterns

Predicts future Bitcoin prices for user-defined days (1 to 365 days)

Visualizes historical actual prices, model predictions, and future price forecasts

Interactive and easy-to-use Streamlit web interface

Project Structure

Data Preparation: Downloads historical closing prices and scales them

Model Architecture: Two-layer LSTM network with dense layers for regression

Training: Trains on 90% of the available historical data

Prediction: Predicts historical test data and future days

Visualization: Plots historical vs predicted prices and future forecasts

UI: Streamlit input for number of days to predict, output as line charts and tables

Requirements

Python 3.7+

Libraries:

yfinance

pandas

numpy

scikit-learn

keras

streamlit

Install dependencies with:

pip install yfinance pandas numpy scikit-learn keras streamlit

How to Run

Clone the repository:

git clone https://github.com/munshifff/bitcoin_price_prediction.git
cd bitcoin_price_prediction


Run the Streamlit app:

streamlit run app.py


The Streamlit web interface will launch locally and provide an input to select the number of future days to predict. The output displays:

Historical closing prices

Predicted prices vs actual for test data

Future forecasted prices with only the date (no time)

Deployed App

You can access the live app on Hugging Face Spaces here:
Bitcoin Price Predictor on Hugging Face

(Replace your-username with your actual Hugging Face username.)

Code Explanation
Data Preparation

Downloads BTC-USD data from Yahoo Finance for the past 15 years

Extracts closing prices

Scales closing prices to the [0,1] range for neural network input

Model Creation

Builds a Sequential Keras model with:

LSTM layer with 128 units (returns sequences)

LSTM layer with 64 units

Dense layer with 25 units

Final Dense layer outputting one value (price prediction)

Compiled using Adam optimizer and mean squared error loss

Training

Converts scaled closing prices into sequences of 100 days as input, and next day price as label

Splits data into training (90%) and testing (10%)

Trains model for 10 epochs with batch size 32

Prediction & Plotting

Predicts on historical test data

Uses the last 100 days data to recursively predict the next N days (user input)

Plots actual historical prices, predicted prices, and future forecast

Displays future predictions in a table with only dates (no timestamps)


