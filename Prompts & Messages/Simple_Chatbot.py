from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

chat_history = [
	SystemMessage(content="You are a helpful assistant"),
]

while True:
    user_input = input('You : ')
    chat_history.append(HumanMessage(content=user_input))
    
    if user_input.lower() == 'exit':
        break
	
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI : ", result.content)

print("Chat history : ", chat_history)



# Output with Message history:
"""
You : Hi
AI :  Hello! How can I help you today?

You : which number is bigger 2 or 0
AI :  2 is bigger than 0.

You : multiply it with 10
AI :  2 multiplied by 10 is 20.

You : now add 5
AI :  20 + 5 = 25.

You : exit

Chat history :  [SystemMessage(content='You are a helpful assistant', additional_kwargs={}, response_metadata={}), 
				HumanMessage(content='Hi', additional_kwargs={}, response_metadata={}), 
				AIMessage(content='Hello! How can I help you today?', additional_kwargs={}, response_metadata={}), 
                HumanMessage(content='which number is bigger 2 or 0', additional_kwargs={}, response_metadata={}), 
                AIMessage(content='2 is bigger than 0.', additional_kwargs={}, response_metadata={}), 
                HumanMessage(content='multiply it with 10', additional_kwargs={}, response_metadata={}), 
                AIMessage(content='2 multiplied by 10 is 20.', additional_kwargs={}, response_metadata={}), 
                HumanMessage(content='now add 5', additional_kwargs={}, response_metadata={}), 
                AIMessage(content='20 + 5 = 25.', additional_kwargs={}, response_metadata={}), 
                HumanMessage(content='exit', additional_kwargs={}, response_metadata={})]
"""