<h1>📈 Bitcoin Price Predictor with LSTM</h1>
    <p>
        This project predicts future Bitcoin (BTC-USD) prices using a <strong>Long Short-Term Memory (LSTM)</strong> neural network.
        The model is trained on 15 years of historical Bitcoin price data fetched from <strong>Yahoo Finance (yfinance)</strong>.
        Predictions are visualized in an interactive <strong>Streamlit</strong> web interface.
    </p>

    <hr>

    <h2>Features</h2>
    <ul>
        <li>Fetches 15 years of historical Bitcoin price data</li>
        <li>Preprocesses data using MinMax scaling</li>
        <li>Trains an LSTM model to learn temporal price patterns</li>
        <li>Predicts future Bitcoin prices for user-defined days (1 to 365 days)</li>
        <li>Visualizes historical actual prices, model predictions, and future price forecasts</li>
        <li>Interactive and easy-to-use Streamlit web interface</li>
    </ul>

    <hr>

    <h2>Project Structure</h2>
    <ul>
        <li><strong>Data Preparation:</strong> Downloads historical closing prices and scales them</li>
        <li><strong>Model Architecture:</strong> Two-layer LSTM network with dense layers for regression</li>
        <li><strong>Training:</strong> Trains on 90% of the available historical data</li>
        <li><strong>Prediction:</strong> Predicts historical test data and future days</li>
        <li><strong>Visualization:</strong> Plots historical vs predicted prices and future forecasts</li>
        <li><strong>UI:</strong> Streamlit input for number of days to predict, output as line charts and tables</li>
    </ul>

    <hr>

    <h2>Requirements</h2>
    <ul>
        <li>Python 3.7+</li>
        <li>Libraries:
            <ul>
                <li>yfinance</li>
                <li>pandas</li>
                <li>numpy</li>
                <li>scikit-learn</li>
                <li>keras</li>
                <li>streamlit</li>
            </ul>
        </li>
    </ul>
    <p>Install dependencies with:</p>
    <pre><code>pip install yfinance pandas numpy scikit-learn keras streamlit</code></pre>

    <hr>

    <h2>How to Run</h2>
    <p>Clone the repository:</p>
    <pre><code>git clone https://github.com/munshifff/bitcoin_price_prediction.git
cd bitcoin_price_prediction</code></pre>

    <p>Run the Streamlit app:</p>
    <pre><code>streamlit run app.py</code></pre>

    <p>The Streamlit web interface will launch locally and provide an input to select the number of future days to predict. The output displays:</p>
    <ul>
        <li>Historical closing prices</li>
        <li>Predicted prices vs actual for test data</li>
        <li>Future forecasted prices with only the date (no time)</li>
    </ul>

    <hr>

    <h2>Deployed App</h2>
    <p>
        You can access the live app on Hugging Face Spaces here:<br>
        <a href="https://huggingface.co/spaces/your-username/bitcoin-price-predictor" target="_blank">Bitcoin Price Predictor on Hugging Face</a><br>
        <em>(Replace <code>your-username</code> with your actual Hugging Face username.)</em>
    </p>

    <hr>

    <h2>Code Explanation</h2>

    <h3>Data Preparation</h3>
    <ul>
        <li>Downloads BTC-USD data from Yahoo Finance for the past 15 years</li>
        <li>Extracts closing prices</li>
        <li>Scales closing prices to the [0,1] range for neural network input</li>
    </ul>

    <h3>Model Creation</h3>
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

    <h3>Training</h3>
    <ul>
        <li>Converts scaled closing prices into sequences of 100 days as input, and next day price as label</li>
        <li>Splits data into training (90%) and testing (10%)</li>
        <li>Trains model for 10 epochs with batch size 32</li>
    </ul>

    <h3>Prediction & Plotting</h3>
    <ul>
        <li>Predicts on historical test data</li>
        <li>Uses the last 100 days data to recursively predict the next N days (user input)</li>
        <li>Plots actual historical prices, predicted prices, and future forecast</li>
        <li>Displays future predictions in a table with only dates (no timestamps)</li>
    </ul>


