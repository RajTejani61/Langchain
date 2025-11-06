import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()

# File Upload
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
        
    st.success("✅ PDF uploaded successfully!")
    
    # Load the PDF
    loader = PyPDFLoader(temp_file_path)
    pages = loader.load()

    text= "\n".join([page.page_content for page in pages])

    st.subheader("📜 Extracted Text (Preview)")
    st.text_area("Extracted Text", text[:1500] + "...", height=300)
    
    llm = ChatGoogleGenerativeAI(
		model="gemini-2.0-flash",
		temperature=0.7
	)
    parser = StrOutputParser()
    
    task = st.selectbox("Choose what to do with the extracted text:", [
        "Summarize the PDF",
        "Ask a Question about PDF"
    ])
    
    # Ask a question
    user_query = ""
    if task == "Ask a Question about PDF":
        user_query = st.text_input("Ask your question:")
        
    
    summarize_prompt = PromptTemplate(
        template="""
        Summarize this PDF text : \n\n {text}
        """,
        input_variables=["text"]
    )
    
    qa_prompt = PromptTemplate(
        template="""
        Answer the following question using this PDF text:\n\n{text}\n\nQuestion: {question}
        """,
        input_variables=["text", "question"]
    )
    
    summarize_chain = summarize_prompt | llm | parser
    qa_chain = qa_prompt | llm | parser
    
    branch_chain = RunnableBranch(
        (lambda x : x["task"] == 'Summarize the PDF', summarize_chain), # type: ignore
        (lambda x : x["task"] == 'Ask a Question about PDF', qa_chain), # type: ignore
        RunnablePassthrough()
    )
    
    # Run
    if st.button("Run Task"):
        with st.spinner("Processing..."):
            inputs = {"task": task, "text": text, "question": user_query}
            result = branch_chain.invoke(inputs)
            st.subheader("✨ Result")
            st.write(result)

    # Cleanup
    os.remove(temp_file_path)