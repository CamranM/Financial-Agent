from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
#from langchain_core.prompts 
from pydantic import BaseModel
#from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
# from tools import search_tool, wiki_tool, save_tool
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
import tools

# print("I GOT HERE")

model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    return_full_text=False,
)

def generate(prompt: str) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    out = gen(
        text,
        max_new_tokens=150,
        do_sample=False
    )

    return out[0]["generated_text"]

###
llm = HuggingFacePipeline(pipeline=gen)
chat_model = ChatHuggingFace(llm=llm)

#generation_config = GenerationConfig(max_new_tokens=1000, do_sample=False)

class ResearchResponse(BaseModel):
    answer: str
    sources: list[str]
    tools_used: list[str]
    
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

system_prompt = f"""
You are a financial forecasting assistant tasked with aiding a retail investor
with individual stock analysis. 

Answer the user's query and use necessary tools when they are available.

YOU MUST WRAP YOUR RESPONSE in this format and provide no other text, if you don't, your response will be considered invalid:

{parser.get_format_instructions()}
"""

agent_tools = [tools.get_stock_price]
agent = create_agent(
    model=chat_model,
    system_prompt= system_prompt,
    tools=agent_tools
)

# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
print(chat_model.bind_tools([tools.get_stock_price]))

for i in range(10):
    query = input("How can I help? ")
    raw_response = agent.invoke({
        "messages": [
            {"role": "user", "content": query}
        ]
    })
    print("Agent response!:", raw_response["messages"][-1].content)

#response = generate("What is your favorite football team?")
#print(response)