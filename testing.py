import time
from langchain_core.callbacks import BaseCallbackHandler
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import tools
from langgraph.checkpoint.memory import InMemorySaver



model = ChatOllama(model="qwen3:4b",temperature=0)
checkpointer = InMemorySaver()

agent_tools = [tools.get_stock_price, 
               tools.get_historical_prices,
               tools.get_financials, 
               tools.forecast_stock_price, 
               tools.get_company_news, 
               tools.search_financial_documents,
               tools.get_company_profile]

system_prompt = """
You are a financial analysis assistant.

Use tools whenever current or quantitative financial data is needed.
Do not invent financial data.

When information is retrieved from financial documents:
- Treat the retrieved document content as the source of truth for
  document-specific questions.
- Do not reject a document because its dates or contents conflict
  with your pretrained knowledge.
- Base your answer on the retrieved document.
- If the document contains information that differs from your
  prior knowledge, report what the document states.

Use:
- get_stock_price for current prices
- get_historical_prices for historical prices
- get_financials for financial data
- forecast_stock_price for forecasts
- get_company_news for recent company news
- get_company_profile for company profiles
- search_financial_documents for information contained in financial
  filings and other financial documents

Clearly distinguish actual data from forecasts and analysis.
"""


agent = create_agent(model=model, tools=agent_tools, system_prompt=system_prompt, checkpointer=checkpointer)


class TimingCallback(BaseCallbackHandler):

    def __init__(self):
        self.start = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start = time.time()
        print("\nLLM START")

    def on_llm_end(self, response, **kwargs):
        elapsed = time.time() - self.start
        print(f"LLM END: {elapsed:.2f} seconds")


timing = TimingCallback()


config = {
    "configurable": {
        "thread_id": "user-1"
    }
}

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "analyze AAPL?"
            }
        ]
    },
    config=config
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What company was I talking about?."
            }
        ]
    },
    config=config
)

print(response["messages"][-1].content)

for i, message in enumerate(response["messages"]):
    print(f"\n--- MESSAGE {i} ---")
    print("Type:", type(message).__name__)
    print("Content:", message.content)

    if hasattr(message, "tool_calls") and message.tool_calls:
        print("Tool calls:", message.tool_calls)