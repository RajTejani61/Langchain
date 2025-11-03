from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
	repo_id="google/gemma-2-2b-it",
	task="text-generation",
)
model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name : str = Field(description="Name of the person")
    age : int = Field(gt=18, description="Age of the person")
    city : str = Field(description="name of the city that person is from")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
	template="Give me name, age and city of a fictional {place} person \n {format_instructions}",
	input_variables=["place"],
	partial_variables={'format_instructions' : parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({"place" : "Indian"})
print(result) # name='Rohan Kumar' age=25 city='Mumbai'


"""
Prompt : 
Give me name, age and city of a fictional Indian person 
\n The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, 
for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\n
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n
{"properties": {"name": {"description": "Name of the person", "title": "Name", "type": "string"}, "age": {"description": "Age of the person", "exclusiveMinimum": 18, "title": "Age", "type": "integer"}, "city": {"description": "name of the city that person is from", "title": "City", "type": "string"}}, "required": ["name", "age", "city"]}\n```'
"""