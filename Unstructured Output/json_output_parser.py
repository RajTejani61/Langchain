from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
	repo_id="meta-llama/Llama-3.1-8B-Instruct",
	task="text-generation",
)
model = ChatHuggingFace(llm=llm)

json_parser = JsonOutputParser()

template = PromptTemplate(
	template="Give me name, age and city of fictional person \n {format_instructions}", # Give me name, age and city of fictional person \n Return a JSON object
	input_variables=[],
	partial_variables={'format_instructions' : json_parser.get_format_instructions()}
)


# prompt = template.format()
# model_result = model.invoke(prompt)
# parsed_result = json_parser.parse(model_result.content)

chain = template | model | json_parser
result = chain.invoke({})
print(result) # {'name': 'Emily Wilson', 'age': 28, 'city': 'Seattle'}
