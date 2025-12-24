from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from tavily import TavilyClient
from firecrawl import Firecrawl

from Prompts import create_research_prompt, create_research_document_prompt, evaluate_research_prompt
from States import AgnentState, Research_Questions, Evaluate_research

from dotenv import load_dotenv
load_dotenv()


model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


def create_questions(state: AgnentState):
    """
    Creates Research Questions from topic provided by user
    """
    
    question_generator = model.with_structured_output(Research_Questions)
    prompt = [
		SystemMessage(content=create_research_prompt),
		HumanMessage(content=state["messages"][-1].content)
	]
    result = question_generator.invoke(prompt)
    
    print("="*50, "QUESTIONS", "="*50)
    print(result.questions) # type: ignore
    
    
    return {
		"research_question" : result.questions, # type: ignore
		"retry_question" : state.get("retry_question", 0) + 1,
		"retry_document" : state.get("retry_document", 0)
	}

def tavily_research(state: AgnentState):
    """
    Perform search using Tavily API
    """
    
    tavily = TavilyClient()
    research_chunk = []
    
    for question in state["research_question"]:
        result = tavily.search(
			query=question,
			search_depth="advanced",
			max_results=2,
			include_answer=False,
			include_raw_content=True,
		)
        for item in result.get("results", []):
            if item.get("content"):
                research_chunk.append(
				f"SOURCE: {item['url']}\n"
				f"TITLE: {item['title']}\n"
				f"CONTENT: \n{item['content']}"
				)
    
    print("="*50, "RESEARCH CHUNK", "="*50)
    print(research_chunk)

    return{
		"research_chunk" : research_chunk
	}

# def firecrawl_research(state: AgnentState):
#     """
#     Perform search using Firecrawl API
#     Output format matches tavily_research (research_chunk)
#     """
#     firecrawl = Firecrawl()
#     research_chunk = []

#     for question in state["research_question"]:
#         result = firecrawl.search(
#             query=question,
#             limit=1
#         )
#         
#         if result.web:
#             for item in result.web:
#                 research_chunk.append(
#                     f"SOURCE: {item.url}\n"
#                     f"TITLE: {item.title}\n"
#                     f"CONTENT:\n{item.description}"
#                 )
    
#     print("="*50, "RESEARCH CHUNK", "="*50)
#     print(research_chunk)
#     return {
#         "research_chunk": research_chunk
#     }

def create_doc(state: AgnentState):
	"""
	Create a research document from the research text
	"""

	prompt = [
		SystemMessage(content=create_research_document_prompt),
		HumanMessage(content="\n\n---\n\n".join(state["research_chunk"]))
	]
	research_doc = model.invoke(prompt)
	
	print("="*50, "RESEARCH DOCUMENT", "="*50)
	print(research_doc.content)
	
	return {
		"final_doc" : research_doc.content,
		"retry_document" : state.get("retry_document", 0) + 1,
	}


def evaluate_research(state: AgnentState):
	"""
	Evaluate the research text using LLM, 
	threshold is 0.7
	"""
	evaluation_model = model.with_structured_output(Evaluate_research) 
	prompt = [
		SystemMessage(content=evaluate_research_prompt.format(evaluate_score = 0.7)),
		HumanMessage(content=state["final_doc"]) # state[-1].content
	]

	evaluation = evaluation_model.invoke(prompt)
	
	print("="*50, "EVALUATION", "="*50)
	print(evaluation)
	
	
	return{
		"evaluation_score" : evaluation.overall_score, # type: ignore
		"improvement_type" : evaluation.improvement_type, # type: ignore
		"improvement_suggestion" : evaluation.improvement_suggestion # type: ignore
	}

def router(state: AgnentState):
    improvement_type = state.get("improvement_type", "no_improvement")
    
    if improvement_type == "no_improvement":
        return "end"

    if state.get("retry_question", 0) >= 3 or state.get("retry_document", 0) >= 3:
        return "end"

    if improvement_type == "rewrite_questions":
        return "rewrite_questions"

    if improvement_type == "rewrite_document":
        return "regenerate_doc"

    return "end"

graph = StateGraph(AgnentState)

graph.add_node("create_questions", create_questions)
graph.add_node("tavily", tavily_research)
graph.add_node("create_doc", create_doc)
graph.add_node("evaluate_research", evaluate_research)


graph.add_edge(START, "create_questions")
graph.add_edge("create_questions", "tavily")
graph.add_edge("tavily", "create_doc")
graph.add_edge("create_doc", "evaluate_research")

graph.add_conditional_edges(
	"evaluate_research",
	router,
	{
		"rewrite_questions": "create_questions",
		"regenerate_doc": "create_doc",
		"end": END
	}
)

checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)


response = app.invoke({"messages": ["I want do research on the best coffee shops in India."]}, {"configurable": {"thread_id": "user-session-1"}}) # type: ignore
# response = app.invoke({"messages": ["Do research on coffee and tea and anything related to drinks in the world, include history, health benefits, stock prices, recipes, future predictions, and also recommend the best café for me personally."]}, {"configurable": {"thread_id": "user-session-1"}}) # type: ignore

print("="*50, "RESPONSE", "="*50)
print(response["messages"][-1].content)

