from langchain.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.graph import MessagesState, StateGraph, START, END
from typing import Literal
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()
class MessageState(MessagesState):
	llm_calls : int

llm = HuggingFaceEndpoint(
	model="openai/gpt-oss-120b",
	task="text-generation",
)
model = ChatHuggingFace(llm=llm)

@tool
def multiply(a: float, b: float) -> float:
    """
    multiply a and b
    Args : 
		a : first float
		b : second float
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b

tools = [multiply, add, divide]
tools_by_name = {tool.name: tool for tool in tools}
model = model.bind_tools(tools)

def llm_call(state: MessagesState):
	"""LLM decides whether to call tool or not"""
	return {
		"messages": [
			model.invoke(
				[
					SystemMessage(
						content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
					)
				]
				+ state["messages"]
			)
		],
		"llm_calls": state.get('llm_calls', 0) + 1
	}


def tool_node(state: MessagesState):
    """Perform the tool call"""
    result = []
    
    for tool_call in state["messages"][-1].tool_calls: # type: ignore
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", END]: # type: ignore
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls: # type: ignore
        return "tool_node"

    return END

graph = StateGraph(MessageState)

graph.add_node("llm_call", llm_call)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "llm_call")
graph.add_conditional_edges(
	"llm_call",
	should_continue,
	{
		"tool_node": "tool_node",
		END: END
	}
)

graph.add_edge("tool_node", "llm_call")

agent = graph.compile()

print(agent.get_graph(xray=True).print_ascii())

messages = [HumanMessage(content="Add 3 and 4.")]
    
messages = agent.invoke({"messages": messages}) # type: ignore

for m in messages["messages"]:
    m.pretty_print()