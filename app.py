import streamlit as st
import os
import requests
import json
import time
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor

# ----------------- Streamlit Page Config -----------------
st.set_page_config(page_title="News Research Tool", layout="wide")
st.title("📰 News Research Tool")

# Gemini API key
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
def get_llm_response(prompt, max_retries=5, base_delay=5):
    """
    Makes a single call to the Gemini API with a given prompt and handles rate limits with exponential backoff.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt}
            ]
        }]
    }

    headers = {
        'Content-Type': 'application/json'
    }

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            
            result = response.json()
            text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No response available.')
            return text
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                st.warning(f"Rate limit hit. Retrying in {base_delay} seconds...")
                time.sleep(base_delay)
                retries += 1
                base_delay *= 2  # Exponential backoff
            else:
                st.error(f"Error calling Gemini API: {e}")
                return "Failed to get a response due to an API error."
        except requests.exceptions.RequestException as e:
            st.error(f"Error calling Gemini API: {e}")
            return "Failed to get a response due to an API error."
    
    st.error("Maximum retries reached for API call. Please try again later.")
    return "Failed to get a response after multiple retries."

# ----------------- Custom URL Loader -----------------
def load_and_parse_urls(urls):
    """
    Manually loads and parses content from URLs with a simple approach.
    """
    data = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for url in urls:
        st.sidebar.info(f"Attempting to load URL: {url}")
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Use a more generic approach to find content
            paragraphs = soup.find_all('p')
            text_content = "\n".join([p.get_text() for p in paragraphs])
                
            if text_content:
                data.append(Document(page_content=text_content, metadata={'source': url}))
            else:
                st.warning(f"Could not find main content for URL: {url}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to load URL {url}. Error: {e}")
            st.warning("This could be due to website security blocking automated requests.")
    return data

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
        st.sidebar.info("Loading and parsing articles...")
        data = load_and_parse_urls(urls)

        if not data:
            st.stop()
        
        st.sidebar.info("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ","],
            chunk_size=2000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(data)

        st.sidebar.info("Summarizing chunks using Gemini API...")
        summarized_docs = []
        progress_bar = st.sidebar.progress(0)
        
        # Sequentially process each document
        for i, doc in enumerate(docs):
            prompt = f"Summarize the following article text concisely: {doc.page_content}"
            summary = get_llm_response(prompt)
            doc.page_content = summary
            summarized_docs.append(doc)
            progress_bar.progress((i + 1) / len(docs))
            
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
        
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        relevant_docs = retriever.get_relevant_documents(query)
        
        context = " ".join([doc.page_content for doc in relevant_docs])

        st.info("Answering your question using the Gemini API...")
        prompt = f"Answer the following question based on the provided articles. If the information is not in the articles, say so. Question: {query}. Articles: {context}"
        answer = get_llm_response(prompt)
        
        st.subheader("Answer")
        st.write(answer)

        if relevant_docs:
            st.subheader("Sources")
            for doc in relevant_docs:
                st.write(doc.metadata.get('source', 'Unknown source'))
