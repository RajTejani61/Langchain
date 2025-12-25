from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
load_dotenv()

class AgentState(MessagesState):
	question : str
	if_doc_related : bool
	retrival_questions : List[str]
	context : List[str]
	answer : str

class Classify_Question(BaseModel):
	if_doc_related : bool = Field(default=False, description="True if the question requires the uploaded document to answer.")

class RetrievalQuestions(BaseModel):
    questions : List[str] = Field(description="3 to 5 short search queries for retrieving document context.")


model = ChatGroq(model="openai/gpt-oss-120b")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


def load_pdf(path):
    print("Adding documents...")
    loader = PyPDFLoader(path)
    document = loader.load()
    chunk = splitter.split_documents(document)
    
    len_chunk = len(chunk)
    print("Chunk length : ",len_chunk)
    
    INDEX_NAME = "simple-qa-rag"
    vectorstore = PineconeVectorStore(
		index_name=INDEX_NAME,
		embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
	)
    vectorstore.add_documents(chunk)
    print("Documents added to vectorstore")
    return vectorstore


path = "D:\\Langchain\\Projects\\AI_internship_Update.pdf"
VECTORSTORE = load_pdf(path)
# VECTORSTORE = PineconeVectorStore(
# 	index_name="simple-qa-rag",
# 	embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
# )

def clarify_question(state : AgentState):
    
    system_prompt = f"""
	You are a classifier.
    
	Your job:
	1. Rewrite the user question to be clear and specific.
	2. Decide if the question can be answered ONLY using the uploaded document.
    
	Rules:
	- If document knowledge is required → if_doc_related = true
	- If general knowledge is enough → false
	- Do NOT answer the question.
	"""
    
    llm = model.with_structured_output(Classify_Question)
    response = llm.invoke(
        [SystemMessage(content=system_prompt), 
        HumanMessage(content=state["question"])]
        )
    
    print("="*100, "CLASSIFICATION", "="*100)
    print(response)
	
    return {
		"if_doc_related" : response.if_doc_related # type: ignore
	}

def routing(state: AgentState):
    if state["if_doc_related"]:
        return "get_context"
    return "answer_directly"

def retrival_questions(state : AgentState):
    system_prompt = """
		You generate search queries for document retrieval.

		Rules:
		- Generate 3 to 5 short, precise questions
		- Questions must help retrieve relevant document sections
		- Do not repeat the original question
	"""
    
    llm = model.with_structured_output(RetrievalQuestions)
	
    response = llm.invoke(
        [SystemMessage(content=system_prompt), 
        HumanMessage(content=state["question"])]
        )
    
    print("="*100, "RETRIVAL QUESTIONS", "="*100)
    print(response)
    
    return {
		"retrival_questions" : response.questions # type: ignore
	}

def retrieve_context(state : AgentState):
    
    retriver = VECTORSTORE.as_retriever(
		search_type="similarity",
		search_kwargs={"k": 5},
	)
    
    docs = []
    for q in state["retrival_questions"]:
        response = retriver.invoke(q)
        docs.extend(response)
    
    context = "\n\n ----- \n\n".join(d.page_content for d in docs)
    print("="*100, "CONTEXT", "="*100)
    print(context)
    
    return {
		"context" : context
	}

def answer_with_context(state: AgentState):
    system_prompt = """
		You are a factual assistant.
        
		Rules:
		- Answer ONLY from the provided context
		- If the answer is missing, say: "Not found in document"
		- Be concise and precise
	"""
    
    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
		Context:
		{state["context"]}
    
		Question:
		{state["question"]}
		""")
    ])
    
    return {"answer": response.content}

def direct_answer(state: AgentState):
    response = model.invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=state["question"])
    ])

    return {"answer": response.content}

graph = StateGraph(AgentState)

graph.add_node("clarify_question", clarify_question)
graph.add_node("generate_retrival_questions", retrival_questions)
graph.add_node("retrvie_context", retrieve_context)
graph.add_node("answer_with_context", answer_with_context)
graph.add_node("direct_answer", direct_answer)

graph.add_edge(START, "clarify_question")
graph.add_conditional_edges(
	"clarify_question",
	routing,
	{
		"get_context": "generate_retrival_questions",
		"answer_directly": "direct_answer"
	}
)
graph.add_edge("generate_retrival_questions", "retrvie_context")
graph.add_edge("retrvie_context", "answer_with_context")
graph.add_edge("answer_with_context", END)
graph.add_edge("direct_answer", END)

app = graph.compile(checkpointer=InMemorySaver())


# response = app.invoke({"question": "explain the machine learning from uploaded document?"}, {"configurable": {"thread_id": "user-session-1"}}) # type: ignore
response = app.invoke({"question": "What topics are listed in the AI internship document?"}, {"configurable": {"thread_id": "user-session-1"}}) # type: ignore

print('='*50, "FINAL ANSWER", '='*50)
print(response)