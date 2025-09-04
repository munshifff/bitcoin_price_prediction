# 📈 Bitcoin Price Predictor with LSTM

This project predicts future Bitcoin (BTC-USD) prices using a **Long Short-Term Memory (LSTM)** neural network.  
The model is trained on 15 years of historical Bitcoin price data fetched from **Yahoo Finance** (`yfinance`).  
Predictions are visualized in an interactive **Streamlit** web interface.

---

## Project Structure

- **Data Preparation:** Downloads historical closing prices and scales them
- **Model Architecture:** Two-layer LSTM network with dense layers for regression
- **Training:** Trains on 90% of the available historical data
- **Prediction:** Predicts historical test data and future days
- **Visualization:** Plots historical vs predicted prices and future forecasts
- **UI:** Streamlit input for number of days to predict, output as line charts and tables

---

## Requirements

- Python 3.7+
- Libraries:
  - `yfinance`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `keras`
  - `streamlit`

Install dependencies with:

```bash
pip install yfinance pandas numpy scikit-learn keras streamlit
```

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/munshifff/bitcoin_price_prediction.git
cd bitcoin_price_prediction
```

Run the Streamlit app:

```bash
streamlit run app.py
```

The Streamlit web interface will launch locally and provide an input to select the number of future days to predict. The output displays:

- Historical closing prices
- Predicted prices vs actual for test data
- Future forecasted prices with only the date (no time)

---

## Deployed App

You can access the live app on Hugging Face Spaces here:  
**[Bitcoin Price Predictor on Hugging Face](https://huggingface.co/spaces/Munshifff/bitcoin-price-prediction-lstm)**  

---

