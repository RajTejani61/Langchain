"""
A generic chatbot often fails to answer specialized questions correctly. 
Build a system that dynamically categorizes user queries (Billing, Technical, or General) and answer user's qquery for that specific topic.
"""

from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import Literal
import json
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
	model="openai/gpt-oss-120b",
	task="text-generation",
)
model = ChatHuggingFace(llm=llm)

class MessageState(MessagesState):
    category: Literal["billing", "technical", "general"]
    answer : str | None

def categorize_and_answer(state: MessageState):
    system_prompt = """
        You are a customer support assistant.

		Step 1: Categorize the user's query into EXACTLY one category:
		- billing
		- technical
		- general

		Step 2: Based on the category, answer using these rules:

		Billing:
		- Answer billing questions
		- Handle payments, subscriptions, invoices
		- Be policy-focused and precise

		Technical:
		- Answer technical questions
		- Debug, explain errors, give steps
		- Ask clarifying questions if needed

		General:
		- Answer general questions
		- Give high-level product explanations
		- Be friendly and simple

		Output instrcutions: 
        STRICT JSON:
		{
			"category": "billing | technical | general",
			"answer": "final answer"
		}
    """
    
    message = [
		SystemMessage(content=system_prompt),
		state["messages"][-1].content
	]
    
    response = model.invoke(message)
    
    raw_content = response.content
    
	# content=
	# '{\n  "category": "billing",\n  "answer": "I’m sorry to hear that your payment didn’t go through even though the amount was deducted. This usually happens when the transaction is flagged as pending or when the merchant’s system doesn’t register the payment correctly. Please try the following steps:\\n\\n1. **Check your bank/ card statement** – Verify whether the amount appears as a pending charge or a completed transaction.\\n2. **Wait 24‑48 hours** – Sometimes the payment will automatically reverse if the merchant cannot capture it.\\n3. **If the charge is still pending after 48\u202fhours**, contact your bank or card issuer to request a reversal.\\n4. **Reach out to our support team** with the transaction ID, the email associated with your account, and the approximate time of the attempted payment. We’ll investigate on our side and, if needed, issue a refund or re‑process the payment.\\n\\nWe aim to resolve billing issues quickly, so please let us know the details and we’ll take care of it for you."\n}' 
	# additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 290, 'prompt_tokens': 266, 'total_tokens': 556}, 'model_name': 'openai/gpt-oss-120b', 'system_fingerprint': 'fp_6b677c2caf', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019b3022-974c-7aa3-b643-917b0f8949bf-0' 
	# usage_metadata={'input_tokens': 266, 'output_tokens': 290, 'total_tokens': 556}
    
    data = json.loads(raw_content) # type: ignore
    category = data["category"]
    answer = data["answer"]
    return {
		"category" : category,
		"answer" : answer,
		"messages" : state["messages"] + [AIMessage(content=answer)]
	}

graph = StateGraph(MessageState)

graph.add_node("categorize_and_answer", categorize_and_answer)

graph.add_edge(START, "categorize_and_answer")
graph.add_edge("categorize_and_answer", END)

app = graph.compile()

# messages = [HumanMessage(content="I had issues with the LangGraph library")]
messages = [HumanMessage(content="My payment failed but money was deducted")]

response = app.invoke({"messages": messages}) # type: ignore

for m in response["messages"]:
	m.pretty_print()

