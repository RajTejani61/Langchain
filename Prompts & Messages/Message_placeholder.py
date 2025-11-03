from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# chat template
chat_template = ChatPromptTemplate([
	('system', 'You are helpful customer support agent'),
	(MessagesPlaceholder(variable_name="chat_history")),
	('human', '{input}'),
])

# load history
chat_history = []
with open("MSG_PH_history.txt", "r") as f:
	chat_history.extend(f.readlines())


# create prompt
prompt = chat_template.invoke({'chat_history' : chat_history, 'input' : 'Where is my refund ?'})


"""
messages=[SystemMessage(content='You are helpful customer support agent', additional_kwargs={}, response_metadata={}), 
HumanMessage(content='HumanMessage(content="I want request a refund for my order #12345")\n', additional_kwargs={}, response_metadata={}), 
HumanMessage(content='AIMessage(content="Your refund request for order #12345 has ben initiated. It will be processed in 3-5 bussiness days.")', additional_kwargs={}, response_metadata={}), 
HumanMessage(content='Where is my refund ?', additional_kwargs={}, response_metadata={})]
"""


model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')
result = model.invoke(prompt)
print(result.content)
