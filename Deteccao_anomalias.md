# Anomaly Detection in Financial Data Using Autoencoders

How to identify anomalous patterns and arbitrage opportunities using unsupervised neural networks.


---

1. Introduction

The financial market generates vast amounts of real-time data, making it crucial to apply advanced techniques for anomaly detection and identifying arbitrage opportunities. Traditional statistical approaches struggle to capture the complexity and non-linearity of financial time series.

Autoencoders, a type of unsupervised neural network, can detect anomalies by learning normal data patterns and highlighting deviations. This article presents multiple Python implementations using TensorFlow/Keras, covering data preprocessing, model training, anomaly detection, and visualization techniques.


---

2. Understanding Anomalies in Financial Data

Financial anomalies can result from:

Fraud or market manipulation (e.g., wash trading, spoofing)

Unexpected price movements due to economic news

Errors in data feeds causing price spikes or dips

Rare events (e.g., flash crashes)

Arbitrage opportunities from price discrepancies across exchanges


Example 1: Generating Synthetic Financial Data with Anomalies

Before diving into deep learning, we generate synthetic data to simulate normal and anomalous patterns.

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic financial data
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", periods=300)
prices = np.cumsum(np.random.randn(300)) + 100  # Normal price movements
prices[100] += 10  # Inject an anomaly
prices[200] -= 15  # Another anomaly

# Convert to DataFrame
df = pd.DataFrame({'date': dates, 'price': prices})
df.set_index('date', inplace=True)

# Plot data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['price'], label="Asset Price")
plt.scatter(df.index[[100, 200]], df['price'].iloc[[100, 200]], color='red', label="Anomalies", marker="x")
plt.legend()
plt.show()
```

---

3. Autoencoder for Anomaly Detection

Autoencoders consist of two parts:

- Encoder: Compresses data into a lower-dimensional representation.
- Decoder: Reconstructs data from the compressed version.


Anomalies have higher reconstruction errors, allowing us to set a threshold for detection.

Example 2: Data Preprocessing

```py
from sklearn.preprocessing import MinMaxScaler

# Normalize data (prices between 0 and 1)
scaler = MinMaxScaler()
df['scaled_price'] = scaler.fit_transform(df[['price']])

# Convert to NumPy array
data = df['scaled_price'].values.reshape(-1, 1)
```

Example 3: Building an Autoencoder with TensorFlow/Keras

```py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define model architecture
input_dim = 1  # One feature (price)
autoencoder = keras.Sequential([
    layers.Dense(32, activation="relu", input_shape=(input_dim,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(input_dim, activation="sigmoid")  # Output same shape as input
])

# Compile and train
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(data, data, epochs=100, batch_size=16, shuffle=True, validation_split=0.1)
```

Example 4: Detecting Anomalies

```py
# Reconstruct data
reconstructed = autoencoder.predict(data)
reconstruction_errors = np.abs(data - reconstructed)

# Define anomaly threshold
threshold = np.percentile(reconstruction_errors, 95)

# Flag anomalies
df['reconstruction_error'] = reconstruction_errors
df['anomaly'] = df['reconstruction_error'] > threshold

# Plot anomalies
plt.figure(figsize=(12,6))
plt.plot(df.index, df['reconstruction_error'], label="Reconstruction Error")
plt.axhline(y=threshold, color="r", linestyle="--", label="Anomaly Threshold")
plt.scatter(df.index[df['anomaly']], df['reconstruction_error'][df['anomaly']], color='red', label="Detected Anomalies")
plt.legend()
plt.show()
```

---

4. Advanced Autoencoder Variations

4.1. LSTM Autoencoder for Sequential Data

LSTMs (Long Short-Term Memory networks) are useful for time-series anomaly detection.

```py
from tensorflow.keras.models import Model

# Reshape data for LSTM (samples, timesteps, features)
time_steps = 10
X = np.array([data[i-time_steps:i] for i in range(time_steps, len(data))])

# Define LSTM Autoencoder
inputs = layers.Input(shape=(time_steps, 1))
encoded = layers.LSTM(32, activation='relu', return_sequences=True)(inputs)
encoded = layers.LSTM(16, activation='relu', return_sequences=False)(encoded)
decoded = layers.RepeatVector(time_steps)(encoded)
decoded = layers.LSTM(16, activation='relu', return_sequences=True)(decoded)
decoded = layers.LSTM(32, activation='relu', return_sequences=True)(decoded)
decoded = layers.TimeDistributed(layers.Dense(1))(decoded)

lstm_autoencoder = Model(inputs, decoded)
lstm_autoencoder.compile(optimizer="adam", loss="mse")

# Train model
lstm_autoencoder.fit(X, X, epochs=50, batch_size=16, shuffle=True, validation_split=0.1)
```

---

5. Application in Arbitrage Trading

Once anomalies are detected, traders can take action.

Example 5: Detecting Arbitrage Opportunities Across Markets

```py
# Simulating price differences between two exchanges
df['exchange_1'] = df['price'] + np.random.randn(len(df)) * 0.5  # Small noise
df['exchange_2'] = df['price'] + np.random.randn(len(df)) * 0.5

# Compute price difference
df['price_diff'] = np.abs(df['exchange_1'] - df['exchange_2'])

# Set arbitrage threshold (e.g., significant difference)
arb_threshold = df['price_diff'].mean() + 2 * df['price_diff'].std()
df['arbitrage_opportunity'] = df['price_diff'] > arb_threshold

# Plot arbitrage opportunities
plt.figure(figsize=(12,6))
plt.plot(df.index, df['price_diff'], label="Price Difference")
plt.axhline(y=arb_threshold, color="r", linestyle="--", label="Arbitrage Threshold")
plt.scatter(df.index[df['arbitrage_opportunity']], df['price_diff'][df['arbitrage_opportunity']], color='green', label="Arbitrage Opportunities")
plt.legend()
plt.show()
```

---

6. Conclusion and Next Steps

We explored multiple ways to detect anomalies in financial data using Autoencoders, covering:

Basic and LSTM-based Autoencoders

Reconstruction error analysis

Arbitrage detection using inter-exchange price differences

