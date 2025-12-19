"""
A generic chatbot often fails to answer specialized questions correctly. 
Build a system that dynamically categorizes user queries (Billing, Technical, or General) and answer user's qquery for that specific topic.
"""

from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from typing import Literal
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()


class MessageState(MessagesState):
    category: Literal["billing", "technical", "general"]
    answer : str | None

class category_output(BaseModel):
    category : Literal["billing", "technical", "general"]


category_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct").with_structured_output(category_output)
billing_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
technical_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
general_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


def categorize(state: MessageState):
    system_prompt = """
        You are a classifier agent. Your job is to categorize the user's query into EXACTLY one category. 
		The categories are:
		- billing
		- technical
		- general
        """
    message = [
		SystemMessage(content=system_prompt),
		state["messages"][-1].content
	]
    response = category_model.invoke(message)
    category = response.category # type: ignore
    return {
		"category": category,
		"messages": state["messages"] + [AIMessage(content=f"Query : '{state['messages'][-1].content}' \nCategory : {category}")]
	}

def billing_node(state: MessageState):
    system_prompt = """
		You are a Billing Support Agent.

		Your role:
		- Handle questions related to billing, payments, invoices, refunds, subscriptions, pricing, and charges.
		- Explain billing issues clearly and politely.
		- Ask for clarification ONLY if required to resolve the issue.
		- Do NOT provide technical troubleshooting steps.
		- Do NOT speculate or invent charges.
		- If account-specific data is needed, clearly state that you cannot access personal account details.

		Response guidelines:
		- Be concise and professional.
		- Use simple language.
		- Provide step-by-step guidance only when necessary.
		- If the issue cannot be resolved, explain the next support action clearly.

		Tone: Calm, helpful, and customer-friendly.
    """
    message = [
		SystemMessage(content=system_prompt), 
		state["messages"][-1].content,
	]
    response = billing_model.invoke(message)
    return {
		"answer": response.content,
		"messages": state["messages"] + [AIMessage(content=response.content)]
	}

def technical_node(state: MessageState):
    system_prompt = """
		You are a Technical Support Engineer.

		Your role:
		- Handle technical issues, bugs, errors, configuration problems, APIs, libraries, and system behavior.
		- Analyze the problem logically and suggest actionable solutions.
		- Ask follow-up questions ONLY if critical details are missing.
		- Do NOT discuss billing or pricing.
		- Do NOT guess unsupported features.

		Response guidelines:
		- Use clear technical explanations.
		- Provide step-by-step fixes when applicable.
		- Mention common causes and best practices.
		- If unsure, explain what information is needed next.

		Tone: Precise, professional, and solution-oriented.
    """
    message = [
		SystemMessage(content=system_prompt),
		state["messages"][-1].content,
	]
    response = technical_model.invoke(message)
    return {
		"answer": response.content,
		"messages": state["messages"] + [AIMessage(content=response.content)]
	}

def general_node(state: MessageState):
    system_prompt = """
		You are a General Support Assistant.

		Your role:
		- Handle general questions about features, usage, capabilities, and documentation.
		- Provide clear explanations without going into deep technical or billing details.
		- Redirect the user politely if the question belongs to billing or technical support.
		- Do NOT provide speculative or unofficial information.

		Response guidelines:
		- Keep answers simple and easy to understand.
		- Use examples when helpful.
		- Avoid unnecessary technical jargon.

		Tone: Friendly, informative, and approachable.
    """
    message = [
		SystemMessage(content=system_prompt),
		state["messages"][-1].content,
	]
    response = general_model.invoke(message)
    return {
		"answer" : response.content,
		"messages": state["messages"] + [AIMessage(content=response.content)]
	}


graph = StateGraph(MessagesState)

graph.add_node("categorize", categorize)
graph.add_node("billing_node", billing_node)
graph.add_node("technical_node", technical_node)
graph.add_node("general_node", general_node)

graph.add_edge(START, "categorize")
graph.add_conditional_edges(
	"categorize",
	lambda state: state["category"],
	{
		"billing": "billing_node",
		"technical": "technical_node",
		"general": "general_node",
	}
)
graph.add_edge("billing_node", END)
graph.add_edge("technical_node", END)
graph.add_edge("general_node", END)

agent = graph.compile()

# messages = [HumanMessage(content="I had returned my order a week ago. When will I get a refund?")]
messages = [HumanMessage(content="I had issue in the LangGraph about Serializer in persistance. explain serializer?")]
response = agent.invoke({"messages": messages}) # type: ignore

for m in response["messages"]:
	m.pretty_print()


"""
                             +-----------+
                             | __start__ |
                             +-----------+
                                    *
                                    *
                                    *
                            +------------+
                            | categorize |.
                         ...+------------+ ....
                    .....          .           .....
                ....               .                ....
             ...                   .                    ...
+--------------+           +--------------+           +----------------+
| billing_node |           | general_node |           | technical_node |
+--------------+****       +--------------+         **+----------------+
                    *****          *           *****
                         ****      *       ****
                             ***   *    ***
                              +---------+
                              | __end__ |
                              +---------+
"""