from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_pinecone import PineconeVectorStore

from tavily import TavilyClient
import json

from Prompts import create_research_prompt, create_research_document_prompt, evaluate_research_prompt
from States import AgnentState, Research_Questions, Evaluate_research

from dotenv import load_dotenv
load_dotenv()


model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.5)
embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')

VECTORESTORE = PineconeVectorStore(
    index_name="research-agent-semantic-cache",
    embedding=embeddings,
)
DISTANCE = 0.8


def get_cache(query: str):
    
    docs = VECTORESTORE.similarity_search_with_score(query, k=1)
    # print("Cache found...")
    # print(docs)
    
    if not docs:
        return None
    
    doc, distance = docs[0]
    if distance < DISTANCE:
        print("Distance too short", distance)
        return None
    
    if "response" not in doc.metadata:
        print("No response in metadata", doc.metadata)
        return None
    
    print("Distance : ", distance)
    # print("Cache hit", doc.metadata["response"])
    return json.loads(doc.metadata["response"])


def set_cache(query: str, value: dict | list):
    
    VECTORESTORE.add_texts(
        texts=[query],
        metadatas=[{
            "response": json.dumps(value),
        }]
    )


def check_cache(state: AgnentState):
    
    query = str(state["messages"][-1].content).strip()
    
    print("Checking Cache query : ", query)
    cached = get_cache(query)
    
    print("="*50, "CACHE RESULT", "="*50)
    print(cached)
    
    if cached:
        return {
            "research_chunk": cached["research_chunk"],
            "cache_hit": True,
            "user_query": query,
        }
    
    return {
        "cache_hit": False,
        "user_query": query,
    }

def cache_router(state: AgnentState):
    if state.get("cache_hit"):
        return "create_doc"
    
    return "create_questions"


def create_questions(state: AgnentState):
    """
    Creates Research Questions from topic provided by user
    """
    
    prompt = [
        SystemMessage(content=create_research_prompt),
        HumanMessage(content=state["user_query"])
    ]
    
    # question_generator = model
    # print("="*50, "QUESTIONS", "="*50, flush=True)
    # for question in question_generator.stream(prompt):
    #     if question.content:
    #         yield {"question" : question}
    
    question_generator = model.with_structured_output(Research_Questions)
    print("="*50, "QUESTIONS", "="*50, flush=True)
    result = question_generator.invoke(prompt)
    print(result.questions) # type: ignore
    
    return {
        "research_question" : result.questions, # type: ignore
        "retry_question" : state.get("retry_question", 0) + 1,
        "retry_document" : state.get("retry_document", 0),
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
    
    set_cache(
        query=state["user_query"], # type: ignore
        value={
            "research_chunk": research_chunk,
        },
    )

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
        if chunk.content:
            yield {"final_doc" : chunk.content}
    
    yield {
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
    
    final_doc = state["final_doc"]
    prompt = [
        SystemMessage(content=evaluate_research_prompt),
        HumanMessage(content=f"""
            RESEARCH DOCUMENT:
            {final_doc}
            """)
            ]

    evaluation_model = model.with_structured_output(Evaluate_research)
    try:
        result = evaluation_model.invoke(prompt)
    except Exception as e:
        print(e)
        return{
            "evaluation_score": 0,
            "improvement_type": "no_improvement",
            "improvement_suggestion": "",
        }
    
    print("Overall Score:", result.overall_score) # type: ignore
    print("Improvement Type:", result.improvement_type) # type: ignore
    print("Improvement Suggestion:", result.improvement_suggestion) # type: ignore
    
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

graph.add_node("check_cache", check_cache)
graph.add_node("create_questions", create_questions)
graph.add_node("tavily", tavily_research)
graph.add_node("create_doc", create_doc)
graph.add_node("evaluate_research", evaluate_research)


graph.add_edge(START, "check_cache")

graph.add_conditional_edges(
    "check_cache",
    cache_router,
    {
        "create_doc": "create_doc",
        "create_questions": "create_questions",
    }
)

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
    {"messages": ["I want do research best black coffee in india"]}, # type: ignore
    config={"configurable": {"thread_id": "user-session-1"}},
    stream_mode="messages",  
):
    print(message_chunk.content, end="", flush=True) # type: ignore