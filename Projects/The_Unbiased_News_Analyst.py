"""
Project 2: The "Unbiased" News Analyst
Problem Statement: We want to analyze conflicting reports of the same event. Build a RAG (Retrieval Augmented Generation) system that can ingest two distinct text files containing conflicting viewpoints. The system should answer user questions strictly based on the context provided in those files.
"""

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

vectorstore = FAISS.from_documents(documents=chunk1, embedding=embeddings, ids=[str(i) for i in range(len_chunk1)])
vectorstore.add_documents(documents=chunk2, ids=[str(len_chunk1 + i) for i in range(len_chunk2)])
vectorstore.save_local("The Unbiased News Analyst vectorstore")

query = input("Enter the query : ").strip()

retriver = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

top_docs = retriver.invoke(query)

if not top_docs:
    print("No relevant documents found in the provided files.")
    exit()

context = ""

for i, doc in enumerate(top_docs, start=1):
    source = doc.metadata.get("source", "UnknownSource")
    content = doc.page_content.strip()
    
    context += f"\n Chunk {i} (Source ===>: {source})  \nContent ===>: {content}\n"

prompt = f"""
You are an assistant that must answer the question using only the retrieved context below (which comes from two files supplied by the user). Do not use any outside knowledge. If the answer is not contained in the provided context, reply: "I don't know the answer based on the provided documents." 
Keep answers length at most 5 to 20 sentences. After the answer, list each source you used in square brackets (ex.- [File A], [File B]). If the retrieved content contains conflicts, explicitly say both claims and indicate their sources.
Question: \n{query}\n\n 
Context: \n{context} 
Answer:
"""

res = model.invoke(prompt)

print(res.content)
