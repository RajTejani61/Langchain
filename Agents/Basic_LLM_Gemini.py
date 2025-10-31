from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)

res = model.invoke("What is the capital of India?")

print(res) # The capital of India is **New Delhi**