from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain.agents import initialize_agent , tool
import os
import datetime

load_dotenv()

GROQ_API_KEY= os.getenv('GROQ_API_KEY1')
TAVILY_API_KEY= os.getenv('TAVILY_API_KEY')

model = ChatGroq(model_name="llama3-70b-8192",
                api_key=GROQ_API_KEY,
                temperature=0.7)

search_tool  = TavilySearchResults(search_depth='basic')

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current Date and time in the specified format"""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


tools = [search_tool , get_system_time]

agent = initialize_agent(tools=tools , llm=model , agent='zero-shot-react-description' , verbose=True)

agent.invoke("What is the date today")
