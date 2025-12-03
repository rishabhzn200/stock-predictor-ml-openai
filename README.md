# Stock Predictor (Machine Learning + OpenAI Explanation)

## Overview
This project implements a complete end-to-end system for stock movement prediction. It takes a stock ticker as an input and predicts whether the price will go up or down the next day. The pipeline includes data ingestion, feature engineering, machine-learning classification, and an optional AI-generated explanation.

## Key Features:
- Real market data ingestion using yfinance (historical OHLCV data).
- Technical indicator computation such as RSI, EMA-10, EMA-50, and MACD.
- Machine-learning model (Random Forest classifier) trained to predict next-day price direction.
- AI explanation layer using an LLM to summarize indicators and trends in simple language. 
- Reusable prediction pipeline, automatically training a new model if none exists.
- FastAPI service layer exposing /analyze and /health endpoints.
- Interactive API documentation provided via Swagger/OpenAPI at /docs.

## Technologies Used

- Python 
- pandas, numpy
- scikit-learn
- FastAPI
- yfinance
- OpenAI API (>=1.0 client)
- Logging for clean trace output

## Running the Project

### Install dependencies from environment.yml:
```
conda env create -f environment.yml
conda activate stock_predictor_env
```


### Set your OpenAI API key:
```
export OPENAI_API_KEY="openapi_key_here"
```


### Run the prediction script using CLI:
```
cd src
python run.py AAPL
```

### Run FastAPI
```
cd src
uvicorn api_main:app --reload
```

The API will start at:
```
http://127.0.0.1:8000
```

Swagger UI is available at:
```
http://127.0.0.1:8000/docs
```

/analyze endpoint can be used either using swagger or curl
```
For Curl, use command below:
curl -X POST "http://127.0.0.1:8000/analyze" \
    -H "Content-Type: application/json" \
    -d '{"ticker": "AAPL", "explain": false}'

For Swagger, use an example request body below:
{
    "ticker": "AAPL",
    "explain": true
}
```

## Example Output
```
Command: python run.py MSFT
[INFO] Validating ticker: MSFT
[INFO] Starting prediction for MSFT...
[INFO] Running prediction for MSFT...
[INFO] Downloading MSFT data for period=2y...
[INFO] Download complete. Rows: 500
[INFO] Loading existing model...
[INFO] Prediction complete for MSFT.
[INFO] Prediction: UP
[INFO] Indicators: {'RSI': 38.8474210902874, 'EMA_10': 487.6570394488056, 'EMA_50': 502.92834301491445, 'MACD': -8.408960787525189}
[INFO] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
[INFO] AI Explanation:
[INFO] The model predicts that Microsoft (MSFT) stock will go up tomorrow based on several technical indicators. Here’s a brief explanation of what each indicator means:

1. **RSI (Relative Strength Index)**: The RSI is a momentum oscillator that measures the speed and change of price movements. It ranges from 0 to 100. An RSI below 30 typically indicates that a stock is oversold (potentially undervalued), while above 70 indicates it's overbought. With an RSI of 38.85, MSFT is approaching the oversold territory, suggesting it may be due for a price increase.

2. **EMA (Exponential Moving Average)**: EMAs are used to identify the trend direction by smoothing out price data. The EMA_10 (short-term) is 487.66, while the EMA_50 (long-term) is 502.93. When the short-term EMA is below the long-term EMA, it often indicates a bearish trend. However, the model might predict an upward move due to potential reversal signals or market sentiment changing.

3. **MACD (Moving Average Convergence Divergence)**: The MACD is a trend-following momentum indicator that shows the relationship between two EMAs. A negative MACD value, like -8.41 here, suggests that the stock is in a downtrend. However, if momentum is shifting, the stock may be preparing for a bounce back, prompting the prediction of an upward movement.

### Investor Considerations:
While the model suggests an upward movement, investors should be cautious and watch for:

- **Market Sentiment**: External factors such as news events, market trends, or economic data can influence stock prices significantly.
- **Confirmation of Trend**: Look for signs of actual price movement confirming the prediction, such as a strong opening or upward momentum in the first hour of trading.
- **Stop Loss**: Implement a stop-loss strategy to manage risk, especially in a volatile environment.
- **Broader Market Trends**: Keep an eye on overall market performance as it can impact individual stocks like MSFT.

In summary, while the indicators support a potential upward movement, caution is advised due to the overall market conditions and the inherent risk in trading.
```
