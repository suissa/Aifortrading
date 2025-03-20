# Machine Learning

1. Linear Regression for Price Prediction
2. Random Forest for Trend Classification
3. Reinforcement Learning with Q-Learning for Trading Strategy



Each of these algorithms can be integrated into a trading bot to improve decision-making. Let's implement each one.


---

1. Linear Regression for Price Prediction

This model uses historical data to predict the future price of BTC.

Implementation:

```py
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Function to fetch historical data from Binance
def get_binance_data(symbol='BTCUSDT', interval='1h', limit=1000):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                     'close_time', 'quote_asset_volume', 'trades', 
                                     'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'])
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['open'] = df['open'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# Fetching data
df = get_binance_data()

# Creating features and target
df['return'] = df['close'].pct_change()
df['volatility'] = df['return'].rolling(10).std()
df.dropna(inplace=True)

X = df[['open', 'high', 'low', 'volume', 'volatility']]
y = df['close']

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Predicting the next closing price
next_price = model.predict([X.iloc[-1]])
print(f'Next closing price prediction: {next_price[0]:.2f}')
```

---

2. Random Forest for Trend Classification

This model classifies BTC's trend as uptrend or downtrend based on historical data.

Implementation:

```py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Creating the trend variable (1 = Uptrend, 0 = Downtrend)
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df.dropna(inplace=True)

X = df[['open', 'high', 'low', 'volume', 'volatility']]
y = df['target']

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predicting the trend for the next candle
next_trend = model.predict([X.iloc[-1]])
print(f'Next trend: {"Uptrend" if next_trend[0] == 1 else "Downtrend"}')
```

---

3. Reinforcement Learning with Q-Learning for Trading Strategy

Here, we use Q-Learning to make buy and sell decisions.

Implementation:

```py
import numpy as np
import random

# Defining actions (0 = Hold, 1 = Buy, 2 = Sell)
actions = [0, 1, 2]

# Initializing the Q-Table
q_table = np.zeros((len(df), len(actions)))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration probability

# Training simulation
for episode in range(1000):
    state = 0  # Start at the first day
    while state < len(df) - 1:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)  # Exploration
        else:
            action = np.argmax(q_table[state])  # Exploitation

        # Defining the reward
        next_state = state + 1
        reward = (df.iloc[next_state]['close'] - df.iloc[state]['close']) if action == 1 else 0

        # Updating the Q-Table
        q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]))

        state = next_state

# Testing the strategy
position = None
profit = 0
for i in range(len(df) - 1):
    action = np.argmax(q_table[i])
    if action == 1 and position is None:  # Buy
        position = df.iloc[i]['close']
    elif action == 2 and position is not None:  # Sell
        profit += df.iloc[i]['close'] - position
        position = None

print(f'Total profit: {profit:.2f} USDT')
```



- Linear Regression: Predicts the next price.
- Random Forest: Classifies the trend (uptrend or downtrend).
- Q-Learning: Reinforcement learning-based trading strategy.


These models can be integrated into trading bots to automate decision-making. Do you want to enhance any of them?

 
