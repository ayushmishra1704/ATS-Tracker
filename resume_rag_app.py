import streamlit as st
import pdfplumber  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# LLM and Embedding setup
llm = Ollama(model="gemma:2b")
embedding = OllamaEmbeddings(model="nomic-embed-text")

st.title(" Resume Analyzer with RAG (Free LLM)")

# File Upload
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

# Optional JD Upload
jd_file = st.file_uploader("Upload Job Description (Optional, PDF)", type="pdf")
jd_text = ""
if jd_file:
    with open("temp_jd.pdf", "wb") as f:
        f.write(jd_file.read())
    jd_doc = pdfplumber.open("temp_jd.pdf")
    for page in jd_doc:
        jd_text += page.get_text()

if uploaded_file:
    # Save PDF temporarily
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split
    loader = PyMuPDFLoader("temp_resume.pdf")
    pages = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    # Create FAISS Vector Store
    vectordb = FAISS.from_documents(docs, embedding)
    retriever = vectordb.as_retriever()

    # Custom RAG Prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an ATS (Applicant Tracking System) resume reviewer and career coach.

Using the following context from the user's resume, answer the question. 
Be specific, helpful, and tailor your feedback based on best resume practices.

---------------------
Context:
{context}
---------------------

Question:
{question}

Answer in a helpful tone:
"""
    )

    # Retrieval QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

    st.success(" Resume processed and indexed.")

    query = st.text_input("Ask something about your resume:", 
                          placeholder="e.g., How can I improve this for a data scientist role?")

    # Append JD text if provided
    if jd_text:
        query += f"\n\nJob Description:\n{jd_text}"

    if query:
        with st.spinner("Analyzing..."):
            answer = qa_chain.run(query)
        st.subheader(" LLM Feedback")
        st.write(answer)
