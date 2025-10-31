from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from dotenv import load_dotenv
# from vertexai import init

load_dotenv()
# init(project="gen-lang-client-0889101668", location="us-central1")

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0) 

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
res = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print(res) # It's always sunny in San Francisco! || tool_calls=[{'name': 'get_weather', 'args': {'city': 'San Francisco'},

"""
{'messages': [HumanMessage(content='what is the weather in sf', additional_kwargs={}, response_metadata={}, id='a3b4588d-4bc9-46c7-8c6c-035658ffb7fa'),
AIMessage(content='', additional_kwargs={'function_call': {'name': 'get_weather', 'arguments': '{"city": "San Francisco"}'}}, 
response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'grounding_metadata': {}, 'model_provider': 'google_genai'}, 
id='lc_run--c718e239-894d-4f92-a27f-c7b3bb84a394-0',

tool_calls=[{'name': 'get_weather', 'args': {'city': 'San Francisco'}, 'id': 'feaf12a2-99f9-49ba-91ad-fbbd0a009fe3', 'type': 'tool_call'}], 
usage_metadata={'input_tokens': 51, 'output_tokens': 73, 'total_tokens': 124, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 57}}), 

ToolMessage(content="It's always sunny in San Francisco!", name='get_weather', id='ca3bf917-b180-4f12-9202-540912d08439', tool_call_id='feaf12a2-99f9-49ba-91ad-fbbd0a009fe3'), 

AIMessage(content="It's always sunny in San Francisco!", additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'grounding_metadata': {}, 'model_provider': 'google_genai'}, id='lc_run--b07eec7d-1bbd-4871-97fd-a1ade8092317-0', 

usage_metadata={'input_tokens': 90, 'output_tokens': 9, 'total_tokens': 99, 'input_token_details': {'cache_read': 0}})]}
"""