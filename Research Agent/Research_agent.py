from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from tavily import TavilyClient

from Prompts import create_research_prompt, create_research_document_prompt, evaluate_research_prompt
from States import AgnentState, Research_Questions, Evaluate_research

from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.5)

def create_questions(state: AgnentState):
    """
    Creates Research Questions from topic provided by user
    """
    
    question_generator = model.with_structured_output(Research_Questions)
    prompt = [
		SystemMessage(content=create_research_prompt),
		HumanMessage(content=state["messages"][-1].content)
	]
    
    print("="*50, "QUESTIONS", "="*50, flush=True)
    for result in question_generator.stream(prompt):
        print("Generating questions...")
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
    
    print("\n\n", "="*50, "RESEARCH CHUNK", "="*50)
    print("fetching results from web...")
    
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
                
                print(f"SOURCE: {item['url']}\n"
				f"TITLE: {item['title']}\n"
				f"CONTENT: \n{item['content'][:100]}.....\n")
	
    return{
		"research_chunk" : research_chunk
	}

def create_doc(state: AgnentState):
	"""
	Create a research document from the research text
	"""

	print("Creating research document...")
	prompt = [
		SystemMessage(content=create_research_document_prompt),
		HumanMessage(content="\n\n---\n\n".join(state["research_chunk"]))
	]
	
	print("="*50, "RESEARCH DOCUMENT", "="*50)
	for chunk in model.stream(prompt):
		pass
	
	return {
		"final_doc" : chunk.content,
		"retry_document" : state.get("retry_document", 0) + 1,
	}


def evaluate_research(state: AgnentState):
    """
    Evaluate the research text using LLM, 
    threshold is 0.7
    """
    print("\n\n" +"="*50, "EVALUATION", "="*50)
    print("Evaluating research document...")
    evaluation_model = model.with_structured_output(Evaluate_research).bind(temperature=0.3)
    prompt = [
        SystemMessage(content=evaluate_research_prompt),
        HumanMessage(content=state["final_doc"]),
    ]
    
    result = evaluation_model.invoke(prompt)
    print(result.overall_score) # type: ignore
    print(result.improvement_type) # type: ignore
    print(result.improvement_suggestion) # type: ignore
    return {
        "evaluation_score": result.overall_score, # type: ignore
        "improvement_type": result.improvement_type, # type: ignore
        "improvement_suggestion": result.improvement_suggestion, # type: ignore
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


for message_chunk, metadata in app.stream(
    {"messages": ["I want do research best coffee shops in india"]}, # type: ignore
	config={"configurable": {"thread_id": "user-session-1"}},
    stream_mode="messages",  
):
    
	print(message_chunk.content, end="", flush=True) # type: ignore