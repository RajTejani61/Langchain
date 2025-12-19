"""
Project 1: 
	Large Language Models often produce better results when they "think" before they write. 
    Create a sequential workflow that mimics a human writer's process: planning the content first, then generating the final text.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv() 

class MessageState(MessagesState):
    plan: str | None
    approved_plan: str | None
    final_answer: str | None
    is_human_approved : bool = False # type: ignore


planner_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
answer_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


def planner(state: MessageState):
    prompt = f"""
		Create EXACTLY 5 bullet points (one sentence each) general plan of below question.
		Do NOT write the final answer.

		Question:
		{state['messages'][-1].content}
		"""
    result = planner_model.invoke(prompt)
    return {
        "plan": result.content,
        "messages": state["messages"] + [AIMessage(content=result.content)]
    }

def human_review(state: MessageState):
    print("\nGenerated plan : \n")
    print(state["plan"])

    user_input = input("\nEdit plan or type 'approve': ")
    approval_list = ["approve", "y", "yes"]
    
    approved = state["plan"] if user_input.lower() in approval_list else user_input
    is_approved = True if user_input.lower() in approval_list else False
    
    return {
        "approved_plan": approved,
        "is_human_approved": is_approved,
        "messages": state["messages"] + [
            HumanMessage(content=f"Approved Plan:\n{approved}")
        ]
    }

def detailed_answer(state: MessageState):
    prompt = f"""
		Write 5-10 sentence about each bullet point or topic in the final answer usning ONLY this plan which provided below:

        PLAN : 
		{state['approved_plan']}
		"""
    result = answer_model.invoke(prompt)
    return {
        "final_answer": result.content,
        "messages": state["messages"] + [AIMessage(content=result.content)]
    }


graph = StateGraph(MessageState)

graph.add_node("planner", planner)
graph.add_node("human_review", human_review)
graph.add_node("detailed_answer", detailed_answer)

graph.add_edge(START, "planner")
graph.add_edge("planner", "human_review")
graph.add_conditional_edges(
    "human_review",
    lambda state: state["is_human_approved"],
    {
        True: "detailed_answer",
        False: "planner",
    }
)
graph.add_edge("detailed_answer", END)

app = graph.compile(checkpointer=InMemorySaver())


messages = [HumanMessage(content="can you give me road map of deep learning?")]
initial_state = {
    "messages" : messages,
    "is_human_approved": False
}
config1 = {"configurable": {"thread_id": "user-session-1"}}

response = app.invoke(initial_state, config=config1) # type: ignore

print(response["messages"][-1].content)


"""
   +-----------+     
   | __start__ |     
   +-----------+     
          *
          *
          *
    +---------+      
    | planner |      
    +---------+      
          *
          *
          *
  +--------------+   
  | human_review |   
  +--------------+   
          .
          .
          .
+-----------------+  
| detailed_answer |  
+-----------------+  
          *
          *
          *
    +---------+
    | __end__ |
    +---------+
"""