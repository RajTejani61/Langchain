"""
Project 1: 
	Large Language Models often produce better results when they "think" before they write. 
    Create a sequential workflow that mimics a human writer's process: planning the content first, then generating the final text.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv() 

class MessageState(MessagesState):
    plan : str | None
    approved_plan: str | None
    final_answer: str | None


def planner(state: MessageState):
    prompt = f"""
    Create only 5 bullet points containing 1 sentence each for answering the following question.
	Do NOT answer the question.
	Question:
	{state['messages'][-1].content}
	"""
    response = model.invoke(prompt)
    return {
		"plan": response.content
	}

def humanInTheLoop(state: MessageState):
    print("\nGenerated Plan:\n", state['plan'])
    decision = input("Approve? (y/n/edit): ").strip().lower()

    approved_plan = state['plan']
    if decision == "edit":
        approved_plan = input("Enter revised plan:\n")
    elif decision == "n":
        approved_plan = None
    
    return {
		"approved_plan": approved_plan
	}

def detailed_answer_node(state: MessageState):
    prompt = f"""
	Using ONLY the approved plan below, write the final answer.
	Approved Plan:
	{state['approved_plan']}
	"""
    response = model.invoke(prompt)
    return {
        "final_answer": response.content,
		"messages": state["messages"] + [AIMessage(content=response.content)]
    }

llm = HuggingFaceEndpoint(
	model="openai/gpt-oss-120b",
	task="text-generation",
)
model = ChatHuggingFace(llm=llm)

# model = ChatOllama(model="llama3")

graph = StateGraph(MessageState)

graph.add_node("planner", planner) 
graph.add_node("feedback", humanInTheLoop) 
graph.add_node("writer", detailed_answer_node) 

graph.add_edge(START, "planner")
graph.add_edge("planner", "feedback")
graph.add_edge("feedback", "writer")
graph.add_edge("writer", END)

app = graph.compile()

messages = [HumanMessage(content="can you give me road map of deep learning?")]
response = app.invoke({"messages": messages}) #type: ignore


for m in response["messages"]:
	m.pretty_print()

