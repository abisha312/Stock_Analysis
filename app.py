import streamlit as st
import os
import requests
import json
import time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# ----------------- Streamlit Page Config -----------------
st.set_page_config(page_title="News Research Tool", layout="wide")
st.title("📰 News Research Tool")

# Gemini API key
# Note: This reads the key from the secrets.toml file for secure deployment.
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("Please set your Gemini API key in Streamlit secrets.")
    st.stop()

# ----------------- Sidebar Inputs -----------------
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"News URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

# FAISS index file path
file_path = "faiss_index"

# ----------------- LLM API Calls -----------------
def generate_summary_with_gemini(text):
    """
    Generates a summary of the provided text using the Gemini API.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"
    
    # Construct the payload for the API call
    payload = {
        "contents": [{
            "parts": [
                {"text": f"Summarize the following article text concisely: {text}"}
            ]
        }]
    }

    headers = {
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for bad status codes
        
        result = response.json()
        summary = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No summary available.')
        return summary
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Gemini API: {e}")
        return "Failed to get summary due to an API error."

def answer_query_with_gemini(question, context):
    """
    Answers a question based on the provided context using the Gemini API.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"
    
    # Construct the payload for the API call
    payload = {
        "contents": [{
            "parts": [
                {"text": f"Answer the following question based on the provided articles. If the information is not in the articles, say so. Question: {question}. Articles: {context}"}
            ]
        }]
    }

    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        
        result = response.json()
        answer = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No answer available.')
        return answer
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Gemini API: {e}")
        return "Failed to get an answer due to an API error."

# ----------------- Vectorstore Loader/Creator -----------------
def get_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": "cpu"}
    )
    if os.path.exists(file_path):
        st.sidebar.info("Loading existing knowledge base...")
        vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
    else:
        st.sidebar.info("Creating FAISS index (first-time run)...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(file_path)
    return vectorstore

# ----------------- Process URLs (Cloud-based Summarization) -----------------
if process_url_clicked:
    if not urls:
        st.sidebar.error("Please enter at least one URL.")
    else:
        st.sidebar.info("Loading articles...")
        loader = UnstructuredURLLoader(urls=urls)
        
        # Add a custom user-agent to the requests to avoid being blocked by some sites.
        # This is a common practice to mimic a web browser.
        try:
            loader.requests_kwargs = {'headers': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}}
            data = loader.load()
        except Exception as e:
            st.error(f"Failed to load URLs. The website might be blocking the request. Error: {e}")
            data = []

        if not data:
            st.stop()
        
        st.sidebar.info("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=2000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(data)

        st.sidebar.info("Summarizing chunks using Gemini API...")
        summarized_docs = []
        progress_bar = st.sidebar.progress(0)
        
        for i, doc in enumerate(docs):
            summary = generate_summary_with_gemini(doc.page_content)
            doc.page_content = summary
            summarized_docs.append(doc)
            progress_bar.progress((i + 1) / len(docs))
            time.sleep(0.5) # small delay to prevent API rate limiting
        
        st.sidebar.info("Creating vectorstore from summarized documents...")
        vectorstore = get_vectorstore(summarized_docs)
        st.sidebar.success("Processing completed! Your knowledge base is ready.")

# ----------------- User Query (Cloud-based QA) -----------------
query = st.text_input("Ask a question about the articles:")

if query:
    if not os.path.exists(file_path):
        st.error("No knowledge base found. Please process URLs first.")
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name="hkunlp/instructor-large",
            model_kwargs={"device": "cpu"}
        )
        vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)

        st.info("Searching for relevant information...")
        
        # Limit retrieved chunks to top 2 for efficiency
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        relevant_docs = retriever.get_relevant_documents(query)
        
        context = " ".join([doc.page_content for doc in relevant_docs])

        st.info("Answering your question using the Gemini API...")
        answer = answer_query_with_gemini(query, context)
        
        st.subheader("Answer")
        st.write(answer)

        if relevant_docs:
            st.subheader("Sources")
            for doc in relevant_docs:
                st.write(doc.metadata.get('source', 'Unknown source'))
