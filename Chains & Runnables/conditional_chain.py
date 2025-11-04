from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model1 = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

model2 = ChatHuggingFace(
	llm=HuggingFaceEndpoint(
		model="google/gemma-2-2b-it",
		task="text-generation",
	)
)

class feedback(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(description="Return sentiment of the review wither positive or negative.")


str_parser = StrOutputParser()
pydantic_parser = PydanticOutputParser(pydantic_object=feedback)


prompt1 = PromptTemplate(
	template="Classy the dentiment of the following feedback text into positive or negative \n {feedback} \n {format_instructions}",
	input_variables=['feedback'],
	partial_variables={'format_instructions' : pydantic_parser.get_format_instructions()}
)

postive_prompt = PromptTemplate(
    template=(
        "The following customer feedback is positive:\n\n"
        "{feedback}\n\n"
        "Write a warm and appreciative response from a company representative. "
        "Thank them sincerely and encourage them to return. "
        "Keep it friendly and brief (2–3 sentences)."
    ),
	input_variables=['feedback']
)

negative_prompt = PromptTemplate(
	template=(
        "The following customer feedback is negative:\n\n"
        "{feedback}\n\n"
        "Write a short, polite, and empathetic response from a company representative. "
        "Acknowledge the concern, apologize briefly, and assure improvement. "
        "Keep the tone professional and concise (2–3 sentences)."
	),
	input_variables=['feedback']
)


classfier_chain = prompt1 | model1 | pydantic_parser # RunnableSequence([prompt1, model1, pydantic_parser])
positive_chain = postive_prompt | model2 | str_parser 
negative_chain = negative_prompt | model2 | str_parser 

# Branch
branch_chain = RunnableBranch(
	(
        lambda x : x.sentiment == "Positive",  # type: ignore
        positive_chain
	),
	(
        lambda x : x.sentiment == "Negative",  # type: ignore
        negative_chain
	),
	RunnableLambda(lambda x : "Cound not find sentiment in the feedback"),	
)

chain = classfier_chain | branch_chain

res = chain.invoke({"feedback" : "This is a terrible phone"})
print(res)

# print(chain.get_graph().draw_ascii())


"""
      +-------------+      
      | PromptInput |      
      +-------------+
             *
             *
             *
    +----------------+
    | PromptTemplate |
    +----------------+
             *
             *
             *
+------------------------+
| ChatGoogleGenerativeAI |
+------------------------+
             *
             *
             *
 +----------------------+
 | PydanticOutputParser |
 +----------------------+
             *
             *
             *
        +--------+
        | Branch |
        +--------+
             *
             *
             *
     +--------------+
     | BranchOutput |
     +--------------+
"""