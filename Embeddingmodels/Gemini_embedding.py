from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')

text = "Delhi is the capital of India."

vector = embedding.embed_query(text)

print(str(vector[:10]))

"""
[-0.023913294076919556, -0.0025249493774026632, 0.0032722405157983303, -0.05829067528247833, -0.014652526937425137,
    0.0067774392664432526, 0.01657681167125702, 0.0019791540689766407, -0.00727713480591774, -0.009348350577056408]
"""