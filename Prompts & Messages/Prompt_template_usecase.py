from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')


st.header("Research Tool")

paper_input = st.selectbox("Select Research Paper Name : ", ["Attention is all you need", "The Generative AI Revolution", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners"])

style_input = st.selectbox("Select Explanation Style : ", ["Beginner friendly", "Technical", "Mathematical", "Code-Oriented"])

length_input = st.selectbox("Select Explanation Length : ", ["Short [1-2 paragraphs]", "Medium [3-4 paragraphs]", "Long [5-6 paragraphs]"])


template = load_prompt("prompt_template_for_summarization.json")
prompt = template.invoke({
	"paper" : paper_input,
	"style" : style_input,
	"length" : length_input
})


if st.button("Summarize"):
	res = model.invoke(prompt)
	st.write(res.content)