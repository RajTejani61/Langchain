"""
Project 2: The "Unbiased" News Analyst
Problem Statement: We want to analyze conflicting reports of the same event. Build a RAG (Retrieval Augmented Generation) system that can ingest two distinct text files containing conflicting viewpoints. The system should answer user questions strictly based on the context provided in those files.
"""

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents import create_agent
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
load_dotenv()


llm = HuggingFaceEndpoint(
    model="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# path1 = "D:\\Langchain\\Projects\\AI_internship_Update.pdf" 
# path2 = "D:\\Langchain\\Projects\\Machine learning.pdf"

path1 = input("Enter the path of the first file : ").strip()
path2 = input("Enter the path of the second file : ").strip()

loader1 = PyPDFLoader(path1)
loader2 = PyPDFLoader(path2)

documents1 = loader1.load()
documents2 = loader2.load()

for d in documents1:
    if not d.metadata:
        d.metadata = {}
    d.metadata['source'] = 'FileA'
    d.metadata['orig_path'] = path1

for d in documents2:
    if not d.metadata:
        d.metadata = {}
    d.metadata['source'] = 'FileB'
    d.metadata['orig_path'] = path2

chunk1 = splitter.split_documents(documents1)
chunk2 = splitter.split_documents(documents2)

len_chunk1 = len(chunk1)
len_chunk2 = len(chunk2)

print("Chunk 1 length : ",len_chunk1)
print("Chunk 2 length : ", len_chunk2)

INDEX_NAME = os.environ.get("PINECONE_INDEX", "the-unbiased-news-analyst")


vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME, 
    embedding=embeddings,
)

vectorstore.add_documents(chunk1)
vectorstore.add_documents(chunk2)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool is used to fetch supporting context from vectorstore by performing similarity search.
    Use this tool to fetch supporting context from the two user-supplied files. 
    Input should be a natural-language query. The tool returns retrieved text.
    Chunks annotated with source tags like [FileA] or [FileB].
    """
    
    retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5},
            )
    
    top_docs = retriever.invoke(query)
    
    if not top_docs:
        print("No relevant documents found in the provided files.")
        exit()
    
    context = ""
    
    for i, doc in enumerate(top_docs, start=1):
        source = doc.metadata.get("source", "UnknownSource")
        content = doc.page_content.strip()
        
        context += f"\n Chunk {i} (Source ===>: {source})  \nContent ===>: {content}\n"
    
    return context

prompt = """
You are an assistant that answer the question using only the retrieved text (which comes from retriver tool that performs similarity search on two files supplied by the user). 
Do not use any outside knowledge to asnwer the question and only use the retrieved text.
Instructions : 
    - If the Retriever doesn't contain the answer, say: 'I don't know the answer based on the provided documents.'
    - Rewrite the user's question into a short, concise question that is a suitable input for the Retriever tool.\n
        Rules : 
        1. Do not change the meaning/intent of the question.\n
        2. Keep it short (5-20 words).\n
        3. Remove vague filler words (e.g., 'please', 'could you') and turn questions into keyphrases.\n
        4. If the user mentions specific sources or files, add those source tokens (e.g., FileA, FileB) so retriever can filter by metadata if available.
    - At the end Keep answers length at most 5 to 10 sentences. Write answer with proper formatting and indentation. After the answer, list each source you used in square brackets (ex.- [File A], [File B]). 
    - If the retrieved content contains conflicts, explicitly say both claims and indicate their sources.\n
"""

query = input("Enter the query : ").strip()

agent = create_agent(
    model=model,
    tools=[retriever_tool],
    system_prompt=prompt,
)

res = agent.invoke({"messages": [{"role": "user", "content": query}]})

print(res["messages"][-1].content)
