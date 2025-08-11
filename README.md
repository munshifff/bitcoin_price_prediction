Bitcoin Price Predictor with LSTM
This project predicts future Bitcoin (BTC-USD) prices using a Long Short-Term Memory (LSTM) neural network. The model is trained on 15 years of historical Bitcoin price data fetched from Yahoo Finance (yfinance). The predictions are visualized in an interactive Gradio web interface.

Features
Fetches 15 years of historical Bitcoin price data.

Preprocesses data using MinMax scaling.

Trains an LSTM model to learn temporal price patterns.

Predicts future Bitcoin prices for user-defined days (1 to 30 days).

Visualizes historical actual prices, model predictions, and future price forecasts.

Interactive and easy-to-use Gradio web interface.

Project Structure
Data Preparation: Downloads historical closing prices and scales them.

Model Architecture: Two-layer LSTM network with dense layers for regression.

Training: Trains on 90% of the available historical data.

Prediction: Predicts historical test data and future days.

Visualization: Plots historical vs predicted prices and future forecasts with value labels.

UI: Gradio slider input for days to predict, output as embedded plot image.

Requirements
Python 3.7+

yfinance

pandas

numpy

matplotlib

keras

scikit-learn

gradio

Install dependencies with:

bash
Copy
Edit
pip install yfinance pandas numpy matplotlib keras scikit-learn gradio
How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/munshifff/bitcoin_price_prediction.git
cd bitcoin_price_prediction
Run the Python script:

bash
Copy
Edit
python app.py
The Gradio web interface will launch locally and provide a slider to select the number of future days to predict. The output will display a plot with historical prices, model predictions, and future forecasted prices.

Hugging Face Model Link
The trained LSTM model and associated files are also hosted on Hugging Face for easy access and sharing:

https://huggingface.co/your-username/bitcoin-lstm-price-predictor

You can download or integrate the model weights and code from this repository.

Code Explanation
Data Preparation (prepare_data)
Downloads BTC-USD data from Yahoo Finance for the past 15 years.

Extracts closing prices.

Scales closing prices to the [0,1] range for neural network input.

Model Creation (create_model)
Builds a Sequential Keras model with:

LSTM layer with 128 units (returns sequences).

LSTM layer with 64 units.

Dense layer with 25 units.

Final Dense layer outputting one value (price prediction).

Compiled using Adam optimizer and mean squared error loss.

Training (train_model)
Converts scaled closing prices into sequences of 100 days as input, and next day price as label.

Splits data into training (90%) and testing (10%).

Trains model for 10 epochs with batch size 5.

Prediction & Plotting (predict_future)
Predicts on historical test data.

Uses the last 100 days data to recursively predict the next N days (user input).

Plots actual historical prices, predicted prices, and future forecast.

Displays values on future prediction points.

Converts plot to a base64-encoded image for embedding in Gradio.

Gradio Interface
Slider to select days to predict (1 to 30).

Output: HTML image embedding the matplotlib plot.
