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
        result =  {"output": f"Result: {sum(numbers)}"}
    else: 
        result =  {"output": "Math operation is not supported."}

    return {
        "messages" : [AIMessage(content=f"{result}")],
        "output" : result,
    }


def clean_text(state: MessageState):
    text = state["messages"][-1].content.lower() # type: ignore
    cleaned = text.strip()
    return {
        "messages": [AIMessage(content=f"Cleaned text : {cleaned}")],
		"output": cleaned
	}



def router_node(state: MessageState):
    text = state["messages"][-1].content.lower() # type: ignore
    
    if any(word in text for word in ["add", "+", "subtract", "multiply", "divide"]):
        return {"route": "math"}
    else:
        return {"route": "clean"}


graph = StateGraph(MessageState)

graph.add_node("router", router_node)
graph.add_node("math", math_node)
graph.add_node("clean", clean_text)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "math": "math",
        "clean": "clean",
    }
)
graph.add_edge("math", END)
graph.add_edge("clean", END)

app = graph.compile()

print(app.get_graph().print_ascii())

message = [HumanMessage(content="add 5 + 6")]
# message = [HumanMessage(content="              clen text")]
response = app.invoke({"messages": message}) # type: ignore

for m in response["messages"]:
    m.pretty_print()


"""
      +-----------+        
      | __start__ |        
      +-----------+        
             *
             *
             *
        +--------+
        | router |
        +--------+
        ..       .
       .          ..       
      .             .      
+-------+        +------+  
| clean |        | math |  
+-------+        +------+  
        **       *
          *    **
           *  *
       +---------+
       | __end__ |
       +---------+
None
================================ Human Message =================================

add 5 + 6
================================== Ai Message ==================================

{'output': 'Result: 11'}
"""