from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from chains import reflection_chain, generation_chain
from langgraph.graph import END, MessageGraph
from langchain_groq import ChatGroq
import os
from langchain_core.messages import AIMessage


load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY1')

model = ChatGroq(model_name="llama3-70b-8192", api_key=GROQ_API_KEY, temperature=0.7)

REFLECT = "reflect"
GENERATE = "generate"

graph = MessageGraph()

def generate_node(state):
    return generation_chain.invoke({"messages": state})


def reflect_node(messages):
    response = reflection_chain.invoke({"messages": messages})
    if isinstance(response, str):
        return [HumanMessage(content=response)]
    elif hasattr(response, 'content'):
        return [HumanMessage(content=response.content)]
    else:
        return [HumanMessage(content=str(response))]



def should_continue(state):
    if len(state) > 2:
        return END 
    return REFLECT       

# Add nodes to the graph
graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)
graph.set_entry_point(GENERATE)

# Add conditional transition based on should_continue
graph.add_conditional_edges(GENERATE, should_continue)

# Add transition from REFLECT back to GENERATE
graph.add_edge(REFLECT, GENERATE)

# Compile and run
app = graph.compile()

response = app.invoke(HumanMessage(content="AI agents taking over content creating"))
print(response)
