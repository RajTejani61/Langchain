"""
Project 1: 
	Large Language Models often produce better results when they "think" before they write. 
    Create a sequential workflow that mimics a human writer's process: planning the content first, then generating the final text.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv() 

class MessageState(MessagesState):
    final_answer: str | None

def plan_and_write(state: MessageState):
    system_prompt = """
        You are an expert writer.

        Follow this exact process internally:
        1. First, generate a concise plan of exactly 5 bullet points.
        - Each bullet must be ONE sentence.
        - Do NOT answer the question yet.
        2. Then, using ONLY that plan, write a detailed, clear final answer.

        Output format (strict):
        PLAN:
        - bullet 1
        - bullet 2
        - bullet 3
        - bullet 4
        - bullet 5

        FINAL ANSWER:
        <final answer here>
    """
    messages = [
        SystemMessage(content=system_prompt),
        state["messages"][-1]  # last user question
    ]
    response = model.invoke(messages)

    return {
        "final_answer": response.content,
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }


llm = HuggingFaceEndpoint(
	model="openai/gpt-oss-120b",
	task="text-generation",
)
model = ChatHuggingFace(llm=llm)

graph = StateGraph(MessageState)

graph.add_node("plan_and_write", plan_and_write)

graph.add_edge(START, "plan_and_write")
graph.add_edge("plan_and_write", END)

app = graph.compile()

messages = [HumanMessage(content="can you give me road map of deep learning?")]

for response in app.stream({"messages": messages}, stream_mode="values"): # type: ignore
    for m in response["messages"]:
        m.pretty_print()
