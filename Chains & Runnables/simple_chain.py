from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
	template="Give me a report on following topic : {topic}",
	input_variables=['topic']
)

prompt2 = PromptTemplate(
	template="Write a 5 line summary of the following text : {text}",
	input_variables=['text']
)

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
# chain = RunnableSequence([prompt1, model, parser, prompt2, model, parser])
# res = chain.invoke({"topic" : "Artificial Intelligence"})

result = chain.invoke({"topic" : "Artificial Intelligence"})
print(result)

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
    +-----------------+
    | StrOutputParser |
    +-----------------+
             *
             *
             *
+-----------------------+
| StrOutputParserOutput |
+-----------------------+
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
    +-----------------+
    | StrOutputParser |
    +-----------------+
             *
             *
             *
+-----------------------+
| StrOutputParserOutput |
+-----------------------+

"""