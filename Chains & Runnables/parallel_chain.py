from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence

from dotenv import load_dotenv

load_dotenv()
model1 = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

model2 = ChatHuggingFace(
	llm=HuggingFaceEndpoint(
		model="google/gemma-2-2b-it",
		task="text-generation",
	)
)

prompt1 = PromptTemplate(
	template="Generate a short and simple report from follwing text \n {text}",
	input_variables=['text']
)

prompt2 = PromptTemplate(
	template="Generate 5 short question answers from following text \n {text}",
	input_variables=['text']
)

prompt3 = PromptTemplate(
	template="merge the provided report and quiz into single document \n report notes : {report} \n quiz : {quiz}",
	input_variables=['report', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
	"report" : prompt1 | model1 | parser, # RunnableSequence([prompt1, model1, parser]),
	"quiz" : prompt2 | model2 | parser, # RunnableSequence([prompt2, model2, parser])
})

mrege_chain = parallel_chain | prompt3 | model1 | parser


text = """
In statistics, linear regression is a model that estimates the relationship between a scalar response (dependent variable) and one or more explanatory variables (regressor or independent variable). A model with exactly one explanatory variable is a simple linear regression; a model with two or more explanatory variables is a multiple linear regression.[1] This term is distinct from multivariate linear regression, which predicts multiple correlated dependent variables rather than a single dependent variable.[2]
In linear regression, the relationships are modeled using linear predictor functions whose unknown model parameters are estimated from the data. Most commonly, the conditional mean of the response given the values of the explanatory variables (or predictors) is assumed to be an affine function of those values; less commonly, the conditional median or some other quantile is used. Like all forms of regression analysis, linear regression focuses on the conditional probability distribution of the response given the values of the predictors, rather than on the joint probability distribution of all of these variables, which is the domain of multivariate analysis.
Linear regression is also a type of machine learning algorithm, more specifically a supervised algorithm, that learns from the labelled datasets and maps the data points to the most optimized linear functions that can be used for prediction on new datasets.[3]
Linear regression was the first type of regression analysis to be studied rigorously, and to be used extensively in practical applications.[4] This is because models which depend linearly on their unknown parameters are easier to fit than models which are non-linearly related to their parameters and because the statistical properties of the resulting estimators are easier to determine.
"""

result = mrege_chain.invoke({"text" : text})
print(result)

# mrege_chain.get_graph().print_ascii()



"""
                  +----------------------------+
                  | Parallel<report,quiz>Input |
                  +----------------------------+
                      ***                 ****
                  ****                        ***
                **                               **
    +----------------+                      +----------------+
    | PromptTemplate |                      | PromptTemplate |
    +----------------+                      +----------------+
             *                                        *
             *                                        *
             *                                        *
+------------------------+                  +-----------------+
| ChatGoogleGenerativeAI |                  | ChatHuggingFace |
+------------------------+                  +-----------------+
             *                                        *
             *                                        *
             *                                        *
    +-----------------+                     +-----------------+
    | StrOutputParser |                     | StrOutputParser |
    +-----------------+                     +-----------------+
                      ***                 ****
                         ****          ***
                             **      **
                  +-----------------------------+
                  | Parallel<report,quiz>Output |
                  +-----------------------------+
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