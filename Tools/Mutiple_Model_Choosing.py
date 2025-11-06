from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

from dotenv import load_dotenv


load_dotenv()

basic_model = ChatGoogleGenerativeAI(model='gemini-2.5-pro', temperature=0.5, timeout=10)
advanced_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.7, timeout=10)

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    request.model = model
    return handler(request)


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(
    model=basic_model,  # Default model
    tools=[search, get_weather],  # Default tools,
    middleware=[dynamic_model_selection],
	system_prompt="You are a helpful assistant. Use the avialable tolls when needed to answer user queries.",
)



# Run the agent
for chunk in agent.stream(
	{"messages": [{"role": "user", "content": "what is the weather in sf and also search for the capital of france"}
    ],"handle_tool_calls" : True,
    },  # type: ignore
	stream_mode="values"
    ):
    
	# # Each chunk contains the full state at that point
	latest_message = chunk["messages"][-1]
	if latest_message.content:
		print(f"Agent: {latest_message.content}")
	elif latest_message.tool_calls:
		print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")


for m in chunk["messages"]:
    print(m.type, ":", m.content or m.tool_calls)


"""
Agent: what is the weather in sf and also search for the capital of france
Calling tools: ['get_weather', 'search']
Agent: Results for: capital of france
Agent: The weather in San Francisco is sunny with a temperature of 72°F. The capital of France is Paris.


human : what is the weather in sf and also search for the capital of france
ai : [{'name': 'get_weather', 'args': {'location': 'sf'}, 'id': 'ef279694-aa46-4c79-9044-da928524ad47', 'type': 'tool_call'}, {'name': 'search', 'args': {'query': 'capital of france'}, 'id': 'da09f524-feb6-499a-8657-31efa1d45a53', 'type': 'tool_call'}]
tool : Weather in sf: Sunny, 72°F
tool : Results for: capital of france
ai : The weather in San Francisco is sunny with a temperature of 72°F. The capital of France is Paris. 
"""

