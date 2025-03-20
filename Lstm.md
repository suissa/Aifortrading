# Como Utilizar Redes Neurais Recorrentes para Previsão de Séries Temporais em Trading

Um Guia Prático sobre a Implementação de RNNs e LSTMs para Previsão de Movimentos no Mercado

## Introdução

A previsão de séries temporais é um dos maiores desafios no mercado financeiro. Modelos estatísticos tradicionais, como ARIMA e regressão linear, muitas vezes falham ao capturar padrões complexos e dependências temporais nos dados financeiros.

Redes Neurais Recorrentes (RNNs) e suas variantes avançadas, como LSTMs (Long Short-Term Memory), são projetadas para lidar com dados sequenciais e podem capturar padrões temporais sutis, tornando-se ferramentas poderosas para previsão de preços de ativos.

Neste artigo, exploraremos como utilizar RNNs e LSTMs para prever movimentos do mercado financeiro. Implementaremos um modelo em Python utilizando a biblioteca TensorFlow/Keras e aplicaremos a técnica a dados reais do mercado.


---

1. Conceitos Fundamentais

1.1 Redes Neurais Recorrentes (RNNs)

Diferentemente das redes neurais tradicionais (MLPs), que tratam cada entrada de forma independente, as RNNs mantêm um estado interno que permite capturar dependências temporais. Isso é crucial para modelar séries temporais, onde eventos passados influenciam eventos futuros.

Problema das RNNs:

As RNNs sofrem com desvanecimento do gradiente, o que dificulta o aprendizado de dependências de longo prazo.


1.2 Long Short-Term Memory (LSTM)

LSTMs são uma variante das RNNs projetadas para superar o problema do desvanecimento do gradiente. Elas utilizam portas de entrada, saída e esquecimento para controlar o fluxo de informações e reter informações relevantes por períodos longos.


---

2. Preparação dos Dados

2.1 Coleta de Dados

Utilizaremos dados históricos do mercado financeiro. Podemos obter esses dados de APIs como Yahoo Finance, Alpha Vantage ou Binance API. Para este exemplo, utilizaremos o Yahoo Finance.

```py
import yfinance as yf
import pandas as pd

# Baixar dados do ativo (exemplo: Apple - AAPL)
ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", end="2024-01-01")

# Exibir as primeiras linhas
print(data.head())
```

---

2.2 Processamento dos Dados

Os dados devem ser normalizados e transformados em uma sequência adequada para o modelo.

```py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Selecionar apenas o preço de fechamento
df = data[['Close']]

# Normalizar os preços para a faixa [0,1]
scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df)

# Função para criar sequências
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

# Definir tamanho da sequência
SEQ_LENGTH = 50  # Número de dias usados para previsão

# Criar as sequências
X, y = create_sequences(df_scaled, SEQ_LENGTH)

# Dividir em treino e teste
split = int(len(X) * 0.8)  # 80% treino, 20% teste
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

print(f"Formato dos dados de treino: {X_train.shape}, {y_train.shape}")
print(f"Formato dos dados de teste: {X_test.shape}, {y_test.shape}")
```

---

3. Construção do Modelo de RNN/LSTM

Agora que os dados estão preparados, construiremos um modelo LSTM usando TensorFlow/Keras.

```py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Criar o modelo LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),  # Primeira camada LSTM
    Dropout(0.2),  # Regularização para evitar overfitting
    LSTM(50, return_sequences=False),  # Segunda camada LSTM
    Dropout(0.2),
    Dense(25),  # Camada totalmente conectada
    Dense(1)  # Camada de saída (previsão do preço)
])

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Resumo do modelo
model.summary()
```

---

4. Treinamento do Modelo

Agora treinamos o modelo com os dados de treino.

```py
# Treinar o modelo
EPOCHS = 20
BATCH_SIZE = 32

history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=1)
```

---

5. Avaliação do Modelo

Vamos visualizar o desempenho do modelo na previsão de preços.

```py
import matplotlib.pyplot as plt

# Fazer previsões
y_pred = model.predict(X_test)

# Reverter normalização para escala original
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Plotar resultados
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label='Preço Real', color='blue')
plt.plot(y_pred_inv, label='Previsão', color='red')
plt.legend()
plt.xlabel("Tempo")
plt.ylabel("Preço")
plt.title("Previsão de Preço usando LSTM")
plt.show()
```

---

6. Implementação de um Indicador de Trading com a LSTM

Agora, vamos transformar nossa previsão em uma estratégia de trading simples. Se o modelo prevê um preço maior que o preço atual, assumimos uma posição de compra. Se prevê um preço menor, assumimos uma posição de venda.

```py
# Criar sinais de compra/venda
signals = np.where(y_pred_inv > y_test_inv, 1, -1)  # 1 = Compra, -1 = Venda

# Plotar sinais de compra/venda
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label='Preço Real', color='blue', alpha=0.5)
plt.scatter(range(len(y_test_inv)), y_test_inv, c=signals, cmap='coolwarm', marker='o', label='Sinais de Trading')
plt.legend()
plt.xlabel("Tempo")
plt.ylabel("Preço")
plt.title("Sinais de Trading baseados em LSTM")
plt.show()
```

---

Apesar dos LSTMs capturarem padrões temporais, eles não garantem previsões perfeitas, pois o mercado financeiro é influenciado por inúmeros fatores externos. Estratégias mais sofisticadas podem incluir atenção, transformers e aprendizado por reforço.

Se quiser aprimorar este modelo, experimente adicionar indicadores técnicos como Média Móvel, RSI e MACD, e testar a robustez das previsões em diferentes ativos.


---


Agora vamos pegar os dados do BTCUSDT da Binance e implementar Backtest. 

Instalação de Dependências

Instale as bibliotecas necessárias:

```py
pip install requests pandas numpy tensorflow matplotlib backtrader
```

Agora vamos treinar nosso modelo e testar ele com Backtest:

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

# === 1. COLETAR DADOS DA BINANCE USANDO API DIRETA ===

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
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    return df[["open", "high", "low", "close", "volume"]]

df = get_binance_klines()
print(df.head())

# === 2. PRÉ-PROCESSAMENTO DOS DADOS ===

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df[["close"]])

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 50
X, y = create_sequences(df_scaled, SEQ_LENGTH)

# Separar treino e teste
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# === 3. CONSTRUIR E TREINAR O MODELO LSTM ===

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# === 4. FAZER PREVISÕES ===

y_pred = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Plotar previsões vs valores reais
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label="Preço Real", color="blue")
plt.plot(y_pred_inv, label="Previsão", color="red")
plt.legend()
plt.xlabel("Tempo")
plt.ylabel("Preço")
plt.title("Previsão de Preço usando LSTM")
plt.show()

# === 5. BACKTEST COM BACKTRADER ===

class LSTMStrategy(bt.Strategy):
    params = (("threshold", 0.001),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.signal = self.datas[0].pred

    def next(self):
        if self.signal[0] > self.dataclose[0] * (1 + self.params.threshold):
            self.buy()
        elif self.signal[0] < self.dataclose[0] * (1 - self.params.threshold):
            self.sell()

# Criar DataFrame para backtest
df_bt = df.copy()
df_bt["pred"] = np.nan
df_bt.iloc[-len(y_pred_inv):, df_bt.columns.get_loc("pred")] = y_pred_inv.flatten()

class PandasDataExtended(bt.feeds.PandasData):
    lines = ("pred",)
    params = (("pred", -1),)

cerebro = bt.Cerebro()
data = PandasDataExtended(dataname=df_bt)
cerebro.adddata(data)
cerebro.addstrategy(LSTMStrategy)
cerebro.broker.set_cash(10000)
cerebro.run()

print(f"Saldo final: {cerebro.broker.getvalue()}")
cerebro.plot()
```

## Explicação do Código

1. Obtenção dos dados via API da Binance

O script consulta 1000 candles de 1 minuto para BTC/USDT.

Ele armazena os valores de open, high, low, close e volume.



2. Pré-processamento

Os preços de fechamento são normalizados com MinMaxScaler.

São criadas sequências de 50 candles para alimentar a LSTM.



3. Criação e treinamento do modelo LSTM

O modelo usa 2 camadas LSTM com Dropout para evitar overfitting.

O treinamento ocorre em 20 épocas com batch size de 32.



4. Previsões e visualização

As previsões são convertidas de volta para a escala original.

O gráfico exibe preços reais x previsões.



5. Backtest com Backtrader

- Sinal de Compra: Se o modelo prevê preço maior que o atual.
- Sinal de Venda: Se o modelo prevê preço menor que o atual.
- O saldo final do backtest é exibido.

Agora você pode testar com timeframes diferentes para melhorar ou validar. 
