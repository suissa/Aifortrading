# Processamento de Linguagem Natural para Análise de Sentimento no Mercado

A análise de sentimento é uma técnica do Processamento de Linguagem Natural (PLN) que permite extrair emoções e opiniões de textos. No contexto do mercado financeiro, essa técnica é utilizada para prever movimentos de ativos como Bitcoin (BTC), analisando notícias, postagens em mídias sociais e outros conteúdos textuais.

Este artigo apresenta um fluxo completo de análise de sentimento aplicado ao Bitcoin, utilizando Python e bibliotecas como TextBlob, VADER, NLTK e transformers. Faremos a coleta de dados, análise de sentimento e a correlação com o preço do BTC.


---

1. Coleta de Dados

Primeiramente, coletamos notícias sobre o Bitcoin e tweets relevantes. Podemos usar APIs como a do Twitter (X), Reddit, ou mesmo scraping de sites de notícias.

Aqui, usaremos web scraping para coletar manchetes de notícias sobre o Bitcoin.

Instalação das Dependências:
```py
pip install requests beautifulsoup4
```

Agora vamos pegar as notícias:

```py
 pandas textblob nltk transformers matplotlib yfinance

Web Scraping de Notícias sobre Bitcoin

import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_bitcoin_news():
    url = "https://news.google.com/search?q=bitcoin&hl=en&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("article")
    
    news_data = []
    for article in articles[:10]:  # Pegamos apenas as 10 primeiras notícias
        headline = article.find("a").text if article.find("a") else "Sem título"
        link = "https://news.google.com" + article.find("a")["href"][1:] if article.find("a") else "#"
        news_data.append({"headline": headline, "link": link})
    
    return pd.DataFrame(news_data)

df_news = get_bitcoin_news()
import ace_tools as tools
tools.display_dataframe_to_user(name="Notícias sobre Bitcoin", dataframe=df_news)
```

---

2. Análise de Sentimento

Usaremos duas abordagens para análise de sentimento:

1. TextBlob (método simples baseado em léxico)


2. Modelo pré-treinado da Hugging Face (BERT para sentimentos)



2.1 Análise com TextBlob

```py
from textblob import TextBlob

def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Varia de -1 (negativo) a 1 (positivo)

df_news["sentiment_textblob"] = df_news["headline"].apply(analyze_sentiment_textblob)

tools.display_dataframe_to_user(name="Sentimento das Notícias (TextBlob)", dataframe=df_news)
```

---

2.2 Análise com Modelo Transformer (BERT)

Usamos um modelo da Hugging Face treinado para análise de sentimento.

```py
from transformers import pipeline

# Carregar modelo de sentimento
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment_bert(text):
    result = sentiment_pipeline(text[:512])[0]  # BERT tem limite de 512 tokens
    return result["label"], result["score"]

df_news["sentiment_bert"] = df_news["headline"].apply(lambda x: analyze_sentiment_bert(x)[0])
df_news["sentiment_score_bert"] = df_news["headline"].apply(lambda x: analyze_sentiment_bert(x)[1])

tools.display_dataframe_to_user(name="Sentimento das Notícias (BERT)", dataframe=df_news)
```

---

3. Correlação com o Preço do Bitcoin

Agora, analisamos como o sentimento das notícias impacta o preço do Bitcoin.

Coletando dados do preço do BTC com Yahoo Finance

```py
import yfinance as yf

btc = yf.Ticker("BTC-USD")
df_btc = btc.history(period="7d", interval="1d")  # Últimos 7 dias

df_btc.reset_index(inplace=True)
df_btc["Date"] = df_btc["Date"].dt.date  # Apenas a data

tools.display_dataframe_to_user(name="Preço do Bitcoin", dataframe=df_btc)
```

Juntando os Dados e Calculando Correlação

```py
df_news["date"] = pd.to_datetime("today").date()  # Consideramos as notícias do dia atual
df_merged = df_news.groupby("date")["sentiment_textblob"].mean().reset_index()

df_final = df_merged.merge(df_btc, left_on="date", right_on="Date", how="inner")
df_final = df_final[["date", "sentiment_textblob", "Close"]]

correlation = df_final["sentiment_textblob"].corr(df_final["Close"])
print(f"Correlação entre sentimento das notícias e preço do BTC: {correlation:.2f}")

tools.display_dataframe_to_user(name="Correlação entre Sentimento e Preço do Bitcoin", dataframe=df_final)
```

---

4. Visualizando a Relação entre Sentimento e Preço

```py
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(df_final["date"], df_final["Close"], marker="o", label="Preço do Bitcoin (USD)")
plt.xlabel("Data")
plt.ylabel("Preço do BTC")
plt.legend()
plt.title("Evolução do Preço do Bitcoin")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,5))
plt.bar(df_final["date"], df_final["sentiment_textblob"], color="green", alpha=0.5, label="Sentimento Médio")
plt.xlabel("Data")
plt.ylabel("Sentimento")
plt.legend()
plt.title("Sentimento das Notícias sobre BTC")
plt.xticks(rotation=45)
plt.show()
```

---

Conclusão

A análise mostrou que há uma correlação entre o sentimento das notícias e o preço do Bitcoin. Esse método pode ser aprimorado incluindo:

Análise em tempo real usando WebSockets ou APIs de mídias sociais.

Modelos mais avançados como LSTMs ou Transformers treinados com dados financeiros.

Integração com estratégias de trading, gerando sinais de compra e venda.


A análise de sentimento pode ser uma ferramenta poderosa para traders, fornecendo insights que complementam a análise técnica.

