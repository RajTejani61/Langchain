"""
Project 3: The Autonomous "Code Auditor" Agent
Problem Statement: Build an agentic workflow that acts as a Python code reviewer and fixer. 
The agent should take a snippet of "broken" Python code, attempt to fix it, and verify the fix. 
It should not stop until the code is syntactically correct or it hits a safety limit.
"""

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import ast
import re
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@tool
def check_code(code: str) -> bool:
	"""Check whether the code is syntactically correct.
		Returns True if the code is correct, or False if Invalid.
	"""

	try:
		ast.parse(code)
		return True
	except SyntaxError as e:
		return False


@tool
def extract_code(code: str):
    """If text contains a ```python``` or ```py``` block, return inner text, else return return text."""
    text = re.search(r"```(?:python|py)?\s*(.*?)\s*```", code, re.DOTALL | re.IGNORECASE)
    return text.group(1).strip() if text else code.strip()


@tool
def fix_code(broken_code: str, mac_attempts: int = 3):
	"""Function to fix broken code using LLM."""
	
	print("Fixing code...")
	code_check_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
	attempts = 0
	current_code = broken_code
	if check_code.invoke(current_code):
		print("Code is valid.")
		return {"fixed_code": current_code, "is_valid": True, "attempts": attempts-1}
	
	for attempt in range(1, mac_attempts+1):
		
		prompt = (
			"You are a Python assistant. Fix this broken python code so it is syntactically correct "
			"and only return the corrected code inside a ``` python ... ``` block (no extra commentary).\n\n"
			"Broken code:\n"
			f"{current_code}\n\n"
			"If you cannot fix fully, give the best attempt and include only the code block."
		)
		
		response = code_check_llm.invoke([HumanMessage(content=prompt)])
		fixed_code = extract_code.invoke(response.content) # type: ignore
		print(f"Attempt {attempt}: received {len(fixed_code.splitlines())} line(s) of code")
		
		if check_code.invoke(fixed_code):
			print("Code is fixed.")
			return {"fixed_code": fixed_code, "is_valid": True, "attempts": attempt}
		
		current_code = fixed_code

	print("Max attempts reached; returning best-effort code.")
	return {"fixed_code": current_code, "is_valid": False, "attempts": attempts}


code = """
print("Hello world")
def greet(name)
    print("Hello,", name)
greet("Alice"
"""

# code = """
# import json

# def load_data file_path):
# with open(file_path, "r") as f
#     data = json.load(f)
# return data

# def calculate_average(numbers):
#     total = 0
#     for n in numbers
#         total += n
#     avg = total / len(numbers
#     return avg

# class User:
#     def __init__(self, name age):
#         self.name = name
#         self.age = age
    
#     def greet(self)
#         print("Hello" self.name)

# def main():
#     file_path = "data.json"
#     data = load_data(file_path)
    
#     nums = data["values"]
#     print("Numbers:", nums)

#     avg = calculate_average(nums)
#     print("Average is" avg)

#     user = User("Alice", 25)
#     user.greet()

# if __name__ == "__main__
#     main()
# """

system_prompt = """
You are expert python code reviewer and fixer. You will be provided with a Python code snippet that may be broken. You should attempt to fix the code, and if you cannot fix it, you should return the best-effort fix.
you have tools like check_code, extract_code, and fix_code.
you can use the tools to fix the code.
At the end return the fixed code with proper indentation and formatting.
Tool usecase:
	- check_code: check whether the code is syntactically correct.
	- extract_code: if text contains a ```python``` or ```py``` block, return inner text, else return return text.
	- fix_code: function to fix broken code using LLM.
	- At the end return with indentation.
Workflow:
	1) Call the 'check_code' tool with the exact snippet.
	2) If the snippet is valid, return it and stop.
	3) If invalid, call 'fix_code' tool to generate a proposed fix.
	4) Use 'extract_code' to get the code from any returned text.
	5) Call 'check_code' on the proposed fix; repeat if necessary.
	Be concise and ensure the final output is a clean python snippet.
"""

agent = create_agent(
	model=llm,
	tools=[check_code, extract_code, fix_code],
	system_prompt=system_prompt,
)

result = agent.invoke({"messages": [{"role": "user", "content": f"Fix this python code: {code}"}]})
print(result["messages"][-1].content)
