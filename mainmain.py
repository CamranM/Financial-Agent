from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import tools
import time



model = ChatOllama(model="qwen3:4b",temperature=0)


agent_tools = [tools.get_stock_price, tools.get_historical_prices,tools.get_financials, tools.forecast_stock_price]

system_prompt = """
You are a financial analysis assistant designed to help users analyze
individual stocks and financial markets. 

Your responsibilities include Basic and Advanced tasks:
- Basic tasks: Finding stock prices, providing company summaries, report financial data and financial news, etc.
- Advanced tasks: Forecasting future stock prices, providing buy/hold/sell ratings, performing analysis on stocks, etc.

The following are guidelines:
- Answering financial questions using reliable data.
- Using available tools when current or quantitative information is needed.
- Clearly distinguishing factual data from analysis or estimates.

When discussing a forecast, clearly identify:
1. The latest actual price, if provided by a tool.
2. The forecasted prices.
3. The forecasting model used.
4. That the forecast is an estimate rather than a guaranteed future price.
.
"""


agent = create_agent(model=model, tools=agent_tools, system_prompt=system_prompt)


response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Find the price to earnings ratio of AAPL."
        }
    ]
})

for i, message in enumerate(response["messages"]):
    print("\n--- MESSAGE", i, "---")
    print("Type:", type(message).__name__)
    print("Content:", message.content)

    if hasattr(message, "tool_calls"):
        print("Tool calls:", message.tool_calls)


'''
response = model.invoke(
    "Find the price to earnings ratio of AAPL."
)
'''
# print(response.content)

print(response["messages"][-1].content)