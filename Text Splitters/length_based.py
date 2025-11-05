from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("D:/Langchain/Data Loaders/AI_internship_Update.pdf")
docs = loader.load()

splitter = CharacterTextSplitter(
	chunk_size=100,
	chunk_overlap=10,
	separator=" "
)

splitted_text = splitter.split_documents(docs)

print(len(splitted_text))
print(splitted_text[1].page_content)
print(splitted_text[1].metadata)

"""
175
Introduction to AI ● Definition and History of AI ● Types of AI: Narrow AI, General AI, and Super AI
{'producer': 'Skia/PDF m140 Google Docs Renderer', 'creator': 'PyPDF', 'creationdate': '', 'title': 'AI internship', 'moddate': '2025-08-14T11:20:03+05:30', 'source': 'D:/Langchain/Data Loaders/AI_internship_Update.pdf', 'total_pages': 23, 'page': 1, 'page_label': '2'}
"""