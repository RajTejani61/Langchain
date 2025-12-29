from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import CachePolicy
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from tavily import TavilyClient
from firecrawl import Firecrawl

from Prompts import create_research_prompt, create_research_document_prompt, evaluate_research_prompt
from States import AgnentState, Research_Questions, Evaluate_research
import asyncio
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.5)


async def stream_text(text, delay=0.05):
    text = str(text)
    for word in text.split():
        print(word, end=" ", flush=True)
        await asyncio.sleep(delay)
    print()


async def create_questions(state: AgnentState):
    """
    Creates Research Questions from topic provided by user
    """
    
    question_generator = model.with_structured_output(Research_Questions)
    prompt = [
		SystemMessage(content=create_research_prompt),
		HumanMessage(content=state["messages"][-1].content)
	]
    result = await question_generator.ainvoke(prompt)
    
    print("="*50, "QUESTIONS", "="*50)
    await stream_text(result.questions) # type: ignore
    
    return {
		"research_question" : result.questions, # type: ignore
		"retry_question" : state.get("retry_question", 0) + 1,
		"retry_document" : state.get("retry_document", 0)
	}

async def tavily_research(state: AgnentState):
    """
    Perform search using Tavily API
    """
    
    tavily = TavilyClient()
    research_chunk = []
    
    print("="*50, "RESEARCH CHUNK", "="*50)
    await stream_text("fetching results from web...")
    
    for question in state["research_question"]:
        result = tavily.search(
			query=question,
			search_depth="advanced",
			max_results=1,
			include_answer=False,
			include_raw_content=True,
		)
        for i, item in enumerate(result.get("results", [])):
            if item.get("content"):
                research_chunk.append(
				f"SOURCE {i}: {item['url']}\n"
				f"TITLE: {item['title']}\n"
				f"CONTENT: \n{item['content']}"
				)
                
                await stream_text(f"SOURCE: {item['url']}\n"
				f"TITLE: {item['title']}\n"
				f"CONTENT: \n{item['content'][:100]}.....\n"
				)
    
    return{
		"research_chunk" : research_chunk
	}

async def create_doc(state: AgnentState):
	"""
	Create a research document from the research text
	"""

	await stream_text("Creating research document...")
	prompt = [
		SystemMessage(content=create_research_document_prompt),
		HumanMessage(content="\n\n---\n\n".join(state["research_chunk"]))
	]
	research_doc = await model.ainvoke(prompt)
	
	print("="*50, "RESEARCH DOCUMENT", "="*50)
	await stream_text(research_doc.content)
	
	return {
		"final_doc" : research_doc.content,
		"retry_document" : state.get("retry_document", 0) + 1,
	}


async def evaluate_research(state: AgnentState):
	"""
	Evaluate the research text using LLM, 
	threshold is 0.7
	"""
	await stream_text("Evaluating research document...")
	evaluation_model = model.with_structured_output(Evaluate_research).bind(temperature=0)
	prompt = [
		SystemMessage(content=evaluate_research_prompt.replace("{evaluate_score}", "0.7")),
		HumanMessage(content=state["final_doc"])
	]

	evaluation = await evaluation_model.ainvoke(prompt)
	
	print("="*50, "EVALUATION", "="*50)
	await stream_text(evaluation)
	
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
graph.add_node("tavily", tavily_research, cache_policy=CachePolicy(ttl=2000))
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


async def main():
	response = await app.ainvoke({"messages": ["I want do research on transformers in deep learning"]}, {"configurable": {"thread_id": "user-session-1"}}) # type: ignore


asyncio.run(main())