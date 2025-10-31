from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "who is rohit sharma"

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity(np.array([query_embedding]), np.array(doc_embedding))[0] # [[0.89709465 0.80898878 0.81865704 0.81300512 0.78034125]]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("Similarity score is : ", score)


"""
tell me about virat kohli
Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.
Similarity score is :  0.9022800342666397

tell me about bumrah
Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers.
Similarity score is :  0.8702866495498517

who is rohit sharma
Rohit Sharma is known for his elegant batting and record-breaking double centuries.
Similarity score is :  0.8774033260602424
"""