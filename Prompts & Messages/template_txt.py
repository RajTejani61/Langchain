from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
	template="""
Please Summarize the research paper titled {paper} with following specifications :
Explanation Style : {style}
Explanation Length : {length}

1. Mathamatical Details : 
	- Include relavant mathamatical equations if present in tha paper
	- Explain the mathamatical concepts using simple, intuitive language where applicable

2. Analogies :
	- Use relatable analogies to explain complex concepts

If certain details are not present in the paper, respond with "Insufficient information is not available" insted of guessing.
Ensure summary is complete and accurate and aligned with provided specifications.
""",
input_variables=["paper", "style", "length"],
validate_template=True
)

template.save("prompt_template_for_summarization.json")