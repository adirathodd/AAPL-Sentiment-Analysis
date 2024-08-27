# Apple Stock Price Prediction with Sentiment Analysis

This project combines sentiment analysis of news headlines with technical analysis indicators to predict stock prices using a deep learning model. The project leverages LSTM layers for time-series forecasting and uses a variety of techniques to enhance prediction accuracy.

## Project Overview

This project is structured into two primary components:

1. **Sentiment Analysis**: Uses a pre-trained NLP model from Hugging Face's `transformers` library to analyze the sentiment of news headlines. The sentiment score is then incorporated as a feature in the stock price prediction model. The news headlines are from https://www.kaggle.com/datasets/BidecInnovations/stock-price-and-news-realted-to-it?select=AppleNewsStock.csv.

2. **Stock Price Prediction**: Integrates sentiment scores along with technical indicators like RSI (Relative Strength Index), SMA (Simple Moving Average), and EMA (Exponential Moving Average) to predict stock prices using an LSTM-based deep learning model.

## Project Structure

- **`sentiment_analysis.py`**: Contains the code for extracting sentiment scores from news headlines using a transformer model. The sentiment scores are saved as a CSV file.
- **`stock_prediction.py`**: Implements the core logic for:
  - Fetching historical stock data.
  - Computing technical indicators like RSI, SMA, and EMA.
  - Incorporating sentiment scores into the dataset.
  - Building and training an LSTM model for stock price prediction.
  - Evaluating the model using error metrics.

## How It Works

### 1. Sentiment Analysis
- The project loads news headlines from a CSV file and applies a sentiment analysis pipeline using the `distilbert-base-uncased-finetuned-sst-2-english` model.
- Sentiment scores are saved in a CSV file (`Scores.csv`) for integration into the prediction model.

### 2. Stock Price Prediction
- Historical stock data for Apple (AAPL) is downloaded using the `yfinance` library.
- Technical indicators (RSI, SMA, EMA) are calculated and added to the dataset.
- Sentiment scores are merged with the stock data.
- The data is preprocessed, scaled, and split into sequences for training.
- A Bidirectional LSTM model is trained to predict the stock's closing price based on the processed data.

### 3. Model Evaluation
- The model is evaluated using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
- Mean Squared Error (MSE): 0.23537902553384724
- Root Mean Squared Error (RMSE): 0.4851587632248306
- Mean Absolute Error (MAE): 0.36229681475497794