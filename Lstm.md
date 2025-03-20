# How to Use Recurrent Neural Networks for Time Series Forecasting in Trading

## A Practical Guide to Implementing RNNs and LSTMs for Market Movement Prediction

Time series forecasting is one of the biggest challenges in the financial market. Traditional statistical models, such as ARIMA and linear regression, often fail to capture complex patterns and temporal dependencies in financial data.

Recurrent Neural Networks (RNNs) and their advanced variants, such as LSTMs (Long Short-Term Memory), are designed to handle sequential data and can capture subtle temporal patterns, making them powerful tools for asset price forecasting.

In this article, we will explore how to use RNNs and LSTMs to predict financial market movements. We will implement a model in Python using the TensorFlow/Keras library and apply the technique to real market data.


---

1. Fundamental Concepts

1.1 Recurrent Neural Networks (RNNs)

Unlike traditional neural networks (MLPs), which treat each input independently, RNNs maintain an internal state that allows capturing temporal dependencies. This is crucial for modeling time series, where past events influence future events.

Problem with RNNs:

RNNs suffer from the vanishing gradient problem, which makes it difficult to learn long-term dependencies.

1.2 Long Short-Term Memory (LSTM)

LSTMs are a variant of RNNs designed to overcome the vanishing gradient problem. They use input, output, and forget gates to control the flow of information and retain relevant information over long periods.


---

2. Data Preparation

2.1 Data Collection

We will use historical financial market data, which can be obtained from APIs like Yahoo Finance, Alpha Vantage, or Binance API. For this example, we will use Yahoo Finance.

```py
import yfinance as yf
import pandas as pd

# Download stock data (example: Apple - AAPL)
ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", end="2024-01-01")

# Display the first rows
print(data.head())
```

---

2.2 Data Processing

The data must be normalized and transformed into a suitable sequence for the model.

```py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Select only the closing price
df = data[['Close']]

# Normalize prices to the range [0,1]
scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df)

# Function to create sequences
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

# Define sequence length
SEQ_LENGTH = 50  # Number of days used for prediction

# Create sequences
X, y = create_sequences(df_scaled, SEQ_LENGTH)

# Split into training and testing
split = int(len(X) * 0.8)  # 80% training, 20% testing
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
```

---

3. Building the RNN/LSTM Model

Now that the data is prepared, we will build an LSTM model using TensorFlow/Keras.

```py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Create the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),  # First LSTM layer
    Dropout(0.2),  # Regularization to avoid overfitting
    LSTM(50, return_sequences=False),  # Second LSTM layer
    Dropout(0.2),
    Dense(25),  # Fully connected layer
    Dense(1)  # Output layer (price prediction)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Model summary
model.summary()
```

---

4. Training the Model

Now we train the model with the training data.

```py
# Train the model
EPOCHS = 20
BATCH_SIZE = 32

history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=1)
```

---

5. Model Evaluation

Let's visualize the model's performance in price prediction.

```py
import matplotlib.pyplot as plt

# Make predictions
y_pred = model.predict(X_test)

# Reverse normalization to the original scale
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Plot results
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label='Real Price', color='blue')
plt.plot(y_pred_inv, label='Prediction', color='red')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Price Prediction using LSTM")
plt.show()
```

---

6. Implementing a Trading Indicator with LSTM

Now, let's transform our prediction into a simple trading strategy. If the model predicts a price higher than the current price, we assume a buy position. If it predicts a lower price, we assume a sell position.

```py
# Create buy/sell signals
signals = np.where(y_pred_inv > y_test_inv, 1, -1)  # 1 = Buy, -1 = Sell

# Plot buy/sell signals
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label='Real Price', color='blue', alpha=0.5)
plt.scatter(range(len(y_test_inv)), y_test_inv, c=signals, cmap='coolwarm', marker='o', label='Trading Signals')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Trading Signals based on LSTM")
plt.show()
```

---

Although LSTMs capture temporal patterns, they do not guarantee perfect predictions, as the financial market is influenced by numerous external factors. More sophisticated strategies can include attention mechanisms, transformers, and reinforcement learning.

If you want to improve this model, try adding technical indicators such as Moving Averages, RSI, and MACD, and test the robustness of predictions on different assets.


---

Now Let's Get BTCUSDT Data from Binance and Implement a Backtest

Installing Dependencies

Install the necessary libraries:

```py
pip install requests pandas numpy tensorflow matplotlib backtrader
```


Now let's train our model and test it with Backtest:

```py
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import backtrader as bt

# === 1. COLLECTING DATA FROM BINANCE USING DIRECT API ===

def get_binance_klines(symbol="BTCUSDT", interval="1m", limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    return df[["open", "high", "low", "close", "volume"]]

df = get_binance_klines()
print(df.head())
```


Now, you can test different timeframes to improve or validate the model!

