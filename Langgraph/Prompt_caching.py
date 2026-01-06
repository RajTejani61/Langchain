from langgraph.graph import StateGraph, START, END
from langgraph.types import CachePolicy
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from typing import TypedDict, List
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


class AppState(TypedDict):
    messages: List


def system_and_tools(state: AppState):
    return {
        "messages": [
            SystemMessage(content="You are a strict finance assistant. Return JSON only."),
            SystemMessage(content="""
							Tools:
							- search(query: string)
							- calculate(a: number, b: number)
							""")
            ]
    }

def user_input(state: AppState):
    return {
        "messages": state["messages"]
    }

def call_llm(state: AppState):
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


graph = StateGraph(AppState)

graph.add_node("cached_prefix", system_and_tools, cache_policy=CachePolicy(ttl=3600))
graph.add_node("user_input", user_input)
graph.add_node("llm", call_llm)

graph.add_edge(START, "cached_prefix")
graph.add_edge("cached_prefix", "user_input")
graph.add_edge("user_input", "llm")
graph.add_edge("llm", END)

app = graph.compile()


result = app.invoke({
    "messages": [HumanMessage(content="What is EBITDA?")]
})
print(result)