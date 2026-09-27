# %% [markdown]
# ### Lets start wotking with LLM

# %%
import os 
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

import warnings 
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

# %% [markdown]
# ## Get the llm model

# %%
from langchain.chat_models import init_chat_model

model = init_chat_model(model="gemma4:latest", model_provider="ollama")
gpt = init_chat_model(model="gpt-5.4-mini",model_provider="openai")
model_fallback1 = init_chat_model(model="openai/gpt-oss-120b", model_provider="groq")
model_fallback2 = init_chat_model(model="qwen/qwen3.8-27b", model_provider="groq")

#print(gpt.profile.get("max_input_tokens"))


# %%
from langchain.tools import tool
import requests
url = "https://api.openweathermap.org/data/2.5/weather?"

@tool
def get_weather(location:str)->str:
    """take the location and return the current weather"""

    params={
        "q":location,
        "appid":OPENWEATHER_API_KEY
    }

    try:
        response = requests.get(url,params=params,timeout=10)
        return response.json()
    except Exception as e:
        return f"Weather fetching is failed. {e}"

# ## Add internet search tool

# %%
from langchain_tavily import TavilySearch 
primary_internet_search = TavilySearch(
    topic="news",
    search_depth="basic",
    max_results=5,
    time_range="week"
)

# %%
#primary_internet_search.invoke("US top news today")



# %%
from langchain_community.tools import DuckDuckGoSearchResults
ddg_search = DuckDuckGoSearchResults()

# %%
#ddg_search.invoke("US top news today")

# %% [markdown]
# ## Add list of tools to the llm

# %%
available_tools = [get_weather,primary_internet_search,ddg_search]
#available_tools = [get_weather,ddg_search]

# %%
prompt = """You are a helpful assistant. 
    For regular conversion use your own knowledge.
    For weather related query use getWeather tool. 
    For news, stocks, current affairs and internet search use primary_internet_search tool. 
    Use ddg_search tool when primary_internet_search tool is failed or not available
    """

# %% [markdown]
# ## Add short-term memory to agent




# %%
from langgraph.checkpoint.memory import InMemorySaver


##Agent creation
from langchain.agents.middleware import ModelFallbackMiddleware, ModelCallLimitMiddleware, ModelRetryMiddleware, ToolCallLimitMiddleware
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    middleware=[
        #ModelFallbackMiddleware(model_fallback1, model_fallback2, gpt),
        # #ModelCallLimitMiddleware(thread_limit=2,
        #                          run_limit=1,
        #                          exit_behavior="end"),
        #ModelRetryMiddleware(max_retries=4, backoff_factor=2, initial_delay=1, max_delay=60), #back_off_delay = initial_delay*fack_off_factor**n
        ToolCallLimitMiddleware(
            tool_name=get_weather.name, 
            thread_limit=3,
            run_limit=1,
            exit_behavior="continue"
        )
    ],
    checkpointer=InMemorySaver(),
    tools=available_tools,
    system_prompt=prompt
)

# %% [markdown]
# ## add messages 

# %%
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from groq import RateLimitError
from ollama import ResponseError

from uuid import uuid4 
thread_id = str(uuid4())
#print(thread_id)
thread_configure  = {"configurable":{"thread_id":thread_id}}

# create an infinite loop
while True:
    messages = [] 
    user_query = input("User: ")
    if(user_query.lower().strip() in ["bye","exit"]):
        print("Bye!")
        exit()

    if(user_query.lower().strip() == '/new'):
        thread_id = str(uuid4())
        print("New thread ID: ",thread_id)
        thread_configure  = {"configurable":{"thread_id":thread_id}}
        continue

    messages.append(HumanMessage(content=user_query))

    try:
        result = agent.invoke({"messages":messages},thread_configure)
        print("AI:", result["messages"][-1].content)
        print("\nUsage Metadata:", result["messages"][-1].response_metadata)
        # for message in result["messages"]:
        #     message.pretty_print()
    except (RateLimitError,ResponseError) as Err:
        print(Err)
    except Exception as e:
        print(e)


