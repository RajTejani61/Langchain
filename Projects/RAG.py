from langchain.agents import create_agent
from langchain.tools import tool	
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC

from typing import List
from dotenv import load_dotenv
load_dotenv()

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
#     index_name="simple-qa-rag",
#     embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
#     )
PC = PineconeGRPC()


@tool
def rag_retrieve(query : List[str]) -> str:
    """
    Retrieve high-quality context from the uploaded document in vectorstore
    using query rewriting, vector search, and reranking.
    """
    
    retriever = VECTORSTORE.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )

    retriever_docs = []
    for q in query:
        
        retriever_docs.extend(retriever.invoke(q))
        
        reranked = PC.inference.rerank(
            model="bge-reranker-v2-m3",
            query=q,
            documents=[d.page_content for d in retriever_docs],
            top_n=5,
            return_documents=True,
        )
        
        context = "\n\n---\n\n".join(
            item["document"]["text"] for item in reranked.data
        )
        
    
    if context == []:
        return ""

    return context

SYSTEM_PROMPT = """
You are a document question-answering agent.

You MUST strictly follow this workflow protocol.

====================
WORKFLOW
====================

1 — Question Reformulation
- Rewrite the user question into a clear, explicit, well-structured query
- if there are multiple questions in one query, then generate list of questions for ecah one
- Expand acronyms
- Add missing context

2 — Retrieval
- Call the tool `rag_retrieve` using the list of rewritten questions
- Store the returned text as CONTEXT

3 — Answer Generation
- Generate the answer using ONLY the retrieved CONTEXT
- You MAY summarize, rephrase, or generalize ideas
    IF they are clearly implied by the context
- Do NOT introduce facts not supported by the context
- If the context does not contain enough information,
    say: "No result found in document"


4 — Self-Evaluation
- Check whether the answer is correct:
- Is fully supported by the context
- Decide: PASS or FAIL

5 — Conditional Retry
- If PASS:
    - Output the final answer and STOP
- If FAIL:
    - Rewrite the question to be more specific
    - Call `rag_retrieve` again
    - Repeat from Step 3
    - After 3 retries, generate the final answer from the context
- You may retry retrieval at most ONE additional time

====================
STRICT RULES
====================
- Never call the tool more than TWO times total
- Never call the tool after you have decided PASS
- Never answer without context
- Final output must be ONLY the answer text

====================
END
====================
"""


agent = create_agent(
    model=model,
    tools=[rag_retrieve],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        ToolCallLimitMiddleware(
            tool_name="rag_retrieve",
            thread_limit=5,
            run_limit=3,
        ), # type: ignore
    ],
)	

response = agent.invoke({"messages": [{"role": "user", "content": "expplain about langchain and langgraph ?"}]})
# response = agent.invoke({"messages": [{"role": "user", "content": "what are teh topics covered in the document ?"}]})

print(response["messages"][-1]["content"]) 