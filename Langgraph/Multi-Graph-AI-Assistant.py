from typing import Optional 
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

class MessageState(MessagesState):
	route: Optional[str]
	output: Optional[str]


def math_node(state: MessageState):
    text = state["messages"][-1].content.lower() # type: ignore

    numbers = [int(x) for x in text.replace("+", " ").split() if x.isdigit()]

    if numbers:
        return {"output": f"Result: {sum(numbers)}"}

    return {"output": "Math operation is not supported."}

math_graph = StateGraph(MessageState)
math_graph.add_node("math", math_node)
math_graph.add_edge(START, "math")
math_graph.add_edge("math", END)
math_agent = math_graph.compile()


def clean_text(state: MessageState):
    text = state["messages"][-1].content.lower() # type: ignore
    cleaned = text.strip()
    return {
		"output": f"Cleaned text : {cleaned}"
	}

text_graph = StateGraph(MessageState)
text_graph.add_node("clean_text", clean_text)
text_graph.add_edge(START, "clean_text")
text_graph.add_edge("clean_text", END)
text_agent = text_graph.compile()


def router_node(state: MessageState):
    text = state["messages"][-1].content.lower() # type: ignore
    
    if any(word in text for word in ["add", "+", "subtract", "multiply", "divide"]):
        return {"route": "math"}
    else:
        return {"route": "clean"}

def call_agent_node(state: MessageState):
    if state["route"] == "math":
        result = math_agent.invoke(state)
    else:
        result = text_agent.invoke(state)

    return {
		"messages": [AIMessage(content=result["output"])],
        "output": result["output"]
        }

call_agent_graph = StateGraph(MessageState)
call_agent_graph.add_node("router", router_node)
call_agent_graph.add_node("call_agent", call_agent_node)

call_agent_graph.add_edge(START, "router")
call_agent_graph.add_edge("router", "call_agent")
call_agent_graph.add_edge("call_agent", END)
app = call_agent_graph.compile()


print("=== ROUTER GRAPH ===")
print(app.get_graph().print_ascii())

print("=== MATH GRAPH ===")
print(math_agent.get_graph().print_ascii())

print("=== TEXT GRAPH ===")
print(text_agent.get_graph().print_ascii())


message = [HumanMessage(content="add 5 + 6")]
# message = [HumanMessage(content="              clen text")]
response = app.invoke({"messages": message}) # type: ignore

for m in response["messages"]:
    m.pretty_print()


"""
=== ROUTER GRAPH ===
+-----------+  
| __start__ |  
+-----------+  
       *       
       *       
       *       
  +--------+   
  | router |   
  +--------+   
       *       
       *       
       *       
+------------+ 
| call_agent |
+------------+
       *
       *
       *
  +---------+
  | __end__ |
  +---------+
None

=== MATH GRAPH ===
+-----------+
| __start__ |
+-----------+
      *
      *
      *
  +------+
  | math |
  +------+
      *
      *
      *
 +---------+
 | __end__ |
 +---------+
None

=== TEXT GRAPH ===
+-----------+
| __start__ |
+-----------+
       *
       *
       *
+------------+
| clean_text |
+------------+
       *
       *
       *
  +---------+
  | __end__ |
  +---------+
None

================================ Human Message =================================

add 5 + 6
================================== Ai Message ==================================

Result: 11
"""