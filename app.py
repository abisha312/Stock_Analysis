import streamlit as st
import os
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

# ----------------- Sidebar Inputs -----------------
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"News URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

# FAISS index file path
file_path = "faiss_index"

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

# ----------------- Process URLs -----------------
if process_url_clicked:
    if not urls:
        st.sidebar.error("Please enter at least one URL.")
    else:
        st.sidebar.info("Loading articles...")
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        st.sidebar.info("Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=500,       # smaller chunks
            chunk_overlap=50      # less overlap
        )
        docs = text_splitter.split_documents(data)

        vectorstore = get_vectorstore(docs)
        st.sidebar.success("Processing completed!")

# ----------------- User Query -----------------
query = st.text_input("Ask a question about the articles:")

if query:
    if not os.path.exists(file_path):
        st.error("No knowledge base found. Please process URLs first.")
    else:
        # Load embeddings & vectorstore
        embeddings = HuggingFaceEmbeddings(
            model_name="hkunlp/instructor-large",
            model_kwargs={"device": "cpu"}
        )
        vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)

        # Use free BLOOM model for text generation (CPU-friendly)
        pipe = pipeline(
            "text-generation",
            model="bigscience/bloom-560m",
            max_new_tokens=256,
            temperature=0.7
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        # Build retrieval chain
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        st.info("Processing your query...")
        result = chain({"question": query}, return_only_outputs=True)

        st.subheader("Answer")
        st.write(result["answer"])

        if result.get("sources"):
            st.subheader("Sources")
            for source in result["sources"].split("\n"):
                st.write(source)
