from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
	repo_id="meta-llama/Llama-3.1-8B-Instruct",
	task="text-generation",
)
model = ChatHuggingFace(llm=llm)

# 1 prompt :- Detailed Report
template1 = PromptTemplate(
	template="""
	Please provide a detailed report on following topic : {topic}
	""",
	input_variables=['topic']
)

# 2 Prompt :- Summary
template2 = PromptTemplate(
	template="Write a 5 line summary of the following text : {text}",
	input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

res = chain.invoke("Artificial Intelligence")

print(res)