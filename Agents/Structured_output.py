from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

class Contactinfo(BaseModel):
    name : str
    email : str
    contact : str

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro', temperature=0)

agent = create_agent(
	model=model,
	tools=[search],
    response_format=ToolStrategy(Contactinfo)
)

res = agent.invoke(
	{"messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]}
)

print(res["structured_response"]) # name='John Doe' email='john@example.com' contact='(555) 123-4567'



"""
{'messages': [HumanMessage(content='Extract contact info from: John Doe, john@example.com, (555) 123-4567', 
additional_kwargs={}, response_metadata={}, id='816bc8f1-79e9-4c7d-9a87-409b052af75f'), 
AIMessage(content='', additional_kwargs={'function_call': {'name': 'Contactinfo', 'arguments': '{"contact": "(555) 123-4567", "email": "john@example.com", "name": "John Doe"}'}}, 
response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-pro', 'safety_ratings': [], 'grounding_metadata': {}, 'model_provider': 'google_genai'}, id='lc_run--437bf80d-2861-4eba-93d5-b7ab8b1735a4-0', 
tool_calls=[{'name': 'Contactinfo', 'args': {'contact': '(555) 123-4567', 'email': 'john@example.com', 'name': 'John Doe'}, 'id': '5d800566-14f7-4842-8bb1-e192d1722c63', 'type': 'tool_call'}], usage_metadata={'input_tokens': 116, 'output_tokens': 173, 'total_tokens': 289, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 132}}), 
ToolMessage(content="Returning structured response: name='John Doe' email='john@example.com' contact='(555) 123-4567'", name='Contactinfo', id='73af76df-a931-4ec3-8aa4-7d71d046d667', tool_call_id='5d800566-14f7-4842-8bb1-e192d1722c63')], 

'structured_response': Contactinfo(name='John Doe', email='john@example.com', contact='(555) 123-4567')}
"""