

<h1>Bitcoin Price Predictor with LSTM</h1>
<p>This project predicts future Bitcoin (BTC-USD) prices using a Long Short-Term Memory (LSTM) neural network. The model is trained on 15 years of historical Bitcoin price data fetched from Yahoo Finance (yfinance). The predictions are visualized in an interactive Gradio web interface.</p>

<h2>Features</h2>
<ul>
    <li>Fetches 15 years of historical Bitcoin price data</li>
    <li>Preprocesses data using MinMax scaling</li>
    <li>Trains an LSTM model to learn temporal price patterns</li>
    <li>Predicts future Bitcoin prices for user-defined days (1 to 30 days)</li>
    <li>Visualizes historical actual prices, model predictions, and future price forecasts</li>
    <li>Interactive and easy-to-use Gradio web interface</li>
</ul>

<h2>Project Structure</h2>
<ul>
    <li><strong>Data Preparation:</strong> Downloads historical closing prices and scales them</li>
    <li><strong>Model Architecture:</strong> Two-layer LSTM network with dense layers for regression</li>
    <li><strong>Training:</strong> Trains on 90% of the available historical data</li>
    <li><strong>Prediction:</strong> Predicts historical test data and future days</li>
    <li><strong>Visualization:</strong> Plots historical vs predicted prices and future forecasts with value labels</li>
    <li><strong>UI:</strong> Gradio slider input for days to predict, output as embedded plot image</li>
</ul>

<h2>Requirements</h2>
<p>Python 3.7+</p>
<ul>
    <li>yfinance</li>
    <li>pandas</li>
    <li>numpy</li>
    <li>matplotlib</li>
    <li>keras</li>
    <li>scikit-learn</li>
    <li>gradio</li>
</ul>

<p>Install dependencies with:</p>
<pre><code>pip install yfinance pandas numpy matplotlib keras scikit-learn gradio</code></pre>

<h2>How to Run</h2>
<p>Clone the repository:</p>
<pre><code>git clone https://github.com/munshifff/bitcoin_price_prediction.git
cd bitcoin_price_prediction</code></pre>

<p>Run the Python script:</p>
<pre><code>python app.py</code></pre>

<p>The Gradio web interface will launch locally and provide a slider to select the number of future days to predict. The output will display a plot with historical prices, model predictions, and future forecasted prices.</p>


<h2>Code Explanation</h2>

<h3>Data Preparation (prepare_data)</h3>
<ul>
    <li>Downloads BTC-USD data from Yahoo Finance for the past 15 years</li>
    <li>Extracts closing prices</li>
    <li>Scales closing prices to the [0,1] range for neural network input</li>
</ul>

<h3>Model Creation (create_model)</h3>
<ul>
    <li>Builds a Sequential Keras model with:
        <ul>
            <li>LSTM layer with 128 units (returns sequences)</li>
            <li>LSTM layer with 64 units</li>
            <li>Dense layer with 25 units</li>
            <li>Final Dense layer outputting one value (price prediction)</li>
        </ul>
    </li>
    <li>Compiled using Adam optimizer and mean squared error loss</li>
</ul>

<h3>Training (train_model)</h3>
<ul>
    <li>Converts scaled closing prices into sequences of 100 days as input, and next day price as label</li>
    <li>Splits data into training (90%) and testing (10%)</li>
    <li>Trains model for 10 epochs with batch size 5</li>
</ul>

<h3>Prediction & Plotting (predict_future)</h3>
<ul>
    <li>Predicts on historical test data</li>
    <li>Uses the last 100 days data to recursively predict the next N days (user input)</li>
    <li>Plots actual historical prices, predicted prices, and future forecast</li>
    <li>Displays values on future prediction points</li>
    <li>Converts plot to a base64-encoded image for embedding in Gradio</li>
</ul>

<h3>Gradio Interface</h3>
<ul>
    <li>Slider to select days to predict (1 to 30)</li>
    <li>Output: HTML image embedding the matplotlib plot</li>
</ul>

