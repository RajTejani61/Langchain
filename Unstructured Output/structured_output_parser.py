from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers.structured import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
	repo_id="meta-llama/Llama-3.1-8B-Instruct",
	task="text-generation",
)
model = ChatHuggingFace(llm=llm)

schema = [
	ResponseSchema(name="fact 1", description="fact 1 about the topic"),
	ResponseSchema(name="fact 2", description="fact 2 about the topic"),
	ResponseSchema(name="fact 3", description="fact 3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas=schema)

template = PromptTemplate(
	template="Give 3 facts about following topic : {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | model | parser
result = chain.invoke({"topic" : "Artificial Intelligence"})
print(result)
