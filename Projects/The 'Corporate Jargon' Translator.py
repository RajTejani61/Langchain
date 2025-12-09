""" The "Corporate Jargon" Translator """

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest, ModelResponse, wrap_model_call
from pydantic import BaseModel, Field
from typing import Literal, Optional, Callable
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


print("Welcome to The Corporate Jargon Translator")

# Input Query
query = input("Enter the email : ")

# Select mode
print("Select mode : ")
print("1. Polite to Blunt")
print("2. Blunt to Polite")
choice = input("Enter your choice (1 or 2): ").strip()

try :
	if choice not in ("1", "2"):
		print("Invalid choice. Please enter 1 or 2.")
		raise ValueError
except ValueError as e:
	print(e)
	exit()


# Context schema
class choice_context(BaseModel):
	choice: Literal["1", "2"] = Field(description="1. Polite to Blunt\n2. Blunt to Polite")


# Dynamic prompt
@dynamic_prompt # type: ignore
def email_generate_prompt(request: ModelRequest) -> Optional[str]:
    # print("Dynamic prompt called")
    choice_value = request.runtime.context.choice # type: ignore
	
    if choice_value.startswith("1"):
        # print("1. Polite to Blunt called")
        prompt = """
			you are helpful assistant. 
			your job is to convert corporate language emails into blunt simple plain and easy english. 
			The email is written in Polite Corporate Language, use llm to generate simple and easy word to explain what the email is about so that the user can understand.
			Note :
				- Define the email subject to make it more clear for the user. (about 10 - 15 words)
				- Define the email body in more clear and simple manner for the user. (about 30 - 100 words)
				- At the end of the email define overview that shows the overall purpose of the email. (about 10 - 20 words)
			Example :
				Subject: Update Needed - API Integration Delay Affecting Mobile Release
				Email Body:
					...
				Overview:
					...
		"""

        return prompt
    
    elif choice_value.startswith("2"):
        # print("2. Blunt to Polite called")
        prompt = """
			you are helpful assistant. 
			your job is to convert blunt simple plain emails into polite corporate language. 
			The email is written in Blunt/Plain English, use llm to generate email with polite corporate and professional tone language that builds impression of professionalism.
			Note : 
				- Add Subject line in professional tone.
				- Add body with more professional and bussiness tone. use polite and professional language.
		"""
        
        return prompt


# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

try: 
	# Agent
	agent = create_agent(
		model = llm,
		middleware = [email_generate_prompt], # type: ignore
		context_schema = choice_context,
		system_prompt = "You are a helpful assistant.",
	)
	
	# Query
	ctx = choice_context(choice=choice)
	result = agent.invoke(
		{"messages": [{"role": "user", "content": query}]},
		context = ctx
	)
	
	
	# Output
	try:
		last_msg = result["messages"][-1].content
		print(last_msg)
	except Exception:
		print(result)
	
except Exception as e:
	print(e)
