from langchain.agents import create_agent
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
load_dotenv()


@tool
def flight_details(flight_number):
    """
	Returns the details of the flight
	"""
    return f"Found flight {flight_number} details..."

@tool
def book_flight(origin, destination, date):
	"""
	Books the flight from origin to destination
	"""
	return f"Flight Booked from {origin} to {destination} on {date}"

@tool
def cancel_flight(flight_number):
	"""
	Cancels the booked flight 
	"""
	return f"Flight {flight_number} cancelled"

@tool
def check_status(flight_number):
	"""
	Returns the status of the flight
	"""
	return f"Flight {flight_number} status..."


llm = HuggingFaceEndpoint(
	model="openai/gpt-oss-120b",
	task="text-generation",
	temperature=0
)
model = ChatHuggingFace(llm=llm)

system_prompt = """
You are a flight booking assistant. you must follow below rules strickely. 
1. You are not allowed to:
    - Give explanations
    - Give suggestions
    - Use any knowledge outside the tools

2. You are only allowed to respond by:
    - Call one of the provided tools, or
    - You can Ask a short clarification question if required information that is missing.
    - You can only return tool answers.

2. Available tools :
    - flight_details -> when the user wants to search or view available flights
    - book_flight -> when the user clearly wants to book a flight
    - cancel_flight -> when the user wants to cancel a booking
    - check_status -> when the user wants to check flight status


4. Required information:
    - To search or book a flight, you need:
        - origin
        - destination
        - travel date
    - If any of this is missing, ask only for the missing fields.

5. Date handling:
    - If the user uses relative dates like:
        [today, tomorrow, yesterday, day after tomorrow, next week, next month, this week, this month]
    - Convert them to the format DD-MM-YYYY
    - Use the current date as reference.

"""


memory = InMemorySaver()
config = {
	"thread_id": "1"
	}
agent = create_agent(
	model = model,
	tools = [flight_details, book_flight, cancel_flight, check_status],
	system_prompt = system_prompt,
	checkpointer=memory,
)

while True:
	query = input("User: ")
	
	if query.lower() in ('q', 'quit', 'exit'):
		break
	
	result = agent.invoke(
		{"messages": [{"role": "user", "content": query}]},
		{"configurable": config},
	)
	print(result["messages"][-1].content)