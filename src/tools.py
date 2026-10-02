import yfinance as yf
from langchain_core.tools import tool
from statsmodels.tsa.arima.model import ARIMA
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
# from langchain.tools import tool
from pathlib import Path

@tool
def get_stock_price(ticker: str):
    """Get the latest available stock price for a given ticker."""

    stock = yf.Ticker(ticker)
    data = stock.history(period="1d")

    if data.empty:
        return f"No price data found for {ticker}."

    price = data["Close"].iloc[-1]

    return {
        "ticker": ticker.upper(),
        "price": float(price)
    }

@tool
def get_historical_prices(ticker: str, period: str = "1y") -> str:
    """
    Get historical daily stock prices for a ticker.
    Period can be things like 1mo, 3mo, 6mo, 1y, 2y, 5y, or max.
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)

    if data.empty:
        return f"No historical data found for {ticker.upper()}."

    data = data.reset_index()

    result = data[
        ["Date", "Open", "High", "Low", "Close", "Volume"]
    ].copy()

    result["Date"] = result["Date"].dt.strftime("%Y-%m-%d")

    return result.to_json(orient="records")

@tool
def get_financials(
    ticker: str,
    statement: str = "income",
    metrics: list[str] | None = None
) -> str:
    """
    Get a company's financial statement data.

    statement must be 'income', 'balance_sheet', or 'cash_flow'.

    metrics is an optional list of financial metrics to return.
    Common aliases such as 'eps', 'revenue', and 'net income'
    are supported.
    """

    stock = yf.Ticker(ticker)

    if statement == "income":
        data = stock.income_stmt
    elif statement == "balance_sheet":
        data = stock.balance_sheet
    elif statement == "cash_flow":
        data = stock.cashflow
    else:
        return "Invalid statement. Choose income, balance_sheet, or cash_flow."

    if data.empty:
        return f"No {statement} data found for {ticker.upper()}."

    if metrics:
        metric_aliases = {
            "eps": "Diluted EPS",
            "diluted eps": "Diluted EPS",
            "basic eps": "Basic EPS",
            "revenue": "Total Revenue",
            "total revenue": "Total Revenue",
            "net income": "Net Income",
            "gross profit": "GrossProfit",
            "operating income": "Operating Income",
            "ebit": "EBIT",
            "ebitda": "EBITDA",
            "r&d": "Research And Development",
            "research and development": "Research And Development",
            "operating expenses": "Operating Expense",
        }

        requested_metrics = []

        for metric in metrics:
            metric_lower = metric.lower().strip()
            actual_metric = metric_aliases.get(metric_lower, metric)

            if actual_metric in data.index:
                requested_metrics.append(actual_metric)

        if not requested_metrics:
            return (
                f"None of the requested metrics were found for "
                f"{ticker.upper()}."
            )

        data = data.loc[requested_metrics]

    return data.to_json()

@tool
def forecast_stock_price(
    ticker: str,
    forecast_days: int = 30
) -> str:
    """
    Forecast a stock's future closing prices using an ARIMA model.
    forecast_days is the number of trading days to forecast.
    The returned prices are forecasted stock prices. Do not modify or
    reinterpret the forecast values.
    """

    stock = yf.Ticker(ticker)
    data = stock.history(period="2y")

    if data.empty:
        return f"No historical data found for {ticker.upper()}."

    prices = data["Close"].dropna().reset_index(drop=True)

    if len(prices) < 100:
        return f"Not enough historical data for {ticker.upper()}."

    # Fit ARIMA model
    model = ARIMA(prices, order=(5, 1, 0))
    fitted_model = model.fit()

    # Forecast
    forecast = fitted_model.forecast(steps=forecast_days)

    # Format forecast for the agent
    results = []

    for i, price in enumerate(forecast, start=1):
        results.append(
            f"Day {i}: ${price:.2f}"
        )

    return (
        f"Ticker: {ticker.upper()}\n"
        f"Model: ARIMA(5,1,0)\n"
        f"Forecast horizon: {forecast_days} trading days\n"
        f"Forecasted closing prices:\n"
        + "\n".join(results)
    )


@tool
def get_company_news(ticker: str) -> str:
    """
    Get recent news articles about a company using its stock ticker.
    Returns article titles, publishers, and links.
    """
    stock = yf.Ticker(ticker)
    news = stock.news

    if not news:
        return f"No recent news found for {ticker.upper()}."

    results = []

    for article in news[:10]:
        content = article.get("content", {})

        title = content.get("title", "No title")
        publisher = content.get("provider", {}).get("displayName", "Unknown")
        url = content.get("canonicalUrl", {}).get("url", "")

        results.append({
            "title": title,
            "publisher": publisher,
            "url": url
        })

    return str(results)

@tool
def get_company_profile(ticker: str) -> str:
    """
    Get basic company profile information for a stock.
    Includes company name, sector, industry, employees, location,
    business summary, and website.
    """
    stock = yf.Ticker(ticker)
    info = stock.info

    profile = {
        "Name": info.get("longName"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Employees": info.get("fullTimeEmployees"),
        "Location": (
            f"{info.get('city')}, {info.get('state')}, "
            f"{info.get('country')}"
        ),
        "Website": info.get("website"),
        "Business Summary": info.get("longBusinessSummary"),
    }

    return str(profile)


# rag
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

BASE_DIR = Path(__file__).resolve().parent.parent

vectorstore = Chroma(
    persist_directory=str(BASE_DIR / "rag" / "vectorstore"),
    embedding_function=embeddings
)
print("VECTORSTORE COUNT:", vectorstore._collection.count())
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)
@tool
def search_financial_documents(query: str) -> str:
    """
    Search financial filings and documents for information relevant
    to a financial question.
    """

    documents = retriever.invoke(query)

    if not documents:
        return "No relevant financial documents found."

    results = []

    for i, document in enumerate(documents, start=1):
        results.append(
            f"Document chunk {i}:\n{document.page_content}"
        )

    return "\n\n".join(results)