import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
import os
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="News Research Tool", layout="wide")
st.title("📰 News Research Tool")
st.sidebar.header("Configuration")

# Sidebar inputs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"News URL {i+1}")

    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

file_path = "faiss_index"

# Process URLs
if process_url_clicked:
    if not urls:
        st.sidebar.error("Please enter at least one URL.")
    else:
        st.sidebar.write("Loading articles...")

        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        st.sidebar.write("Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=1000,
            chunk_overlap=100,
        )
        docs = text_splitter.split_documents(data)

        st.sidebar.write("Creating embeddings...")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Save locally
        vectorstore.save_local(file_path)

        st.sidebar.success("Processing completed!")

# User Query
query = st.text_input("Ask a question about the articles:")

if query:
    if not os.path.exists(file_path):
        st.error("No knowledge base found. Please process URLs first.")
    else:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            max_completion_tokens=500  # updated param
        )

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        st.write("Processing your query...")
        result = chain({"question": query}, return_only_outputs=True)

        st.subheader("Answer")
        st.write(result["answer"])

        if result.get("sources"):
            st.subheader("Sources")
            sources = result["sources"].split("\n")
            for source in sources:
                st.write(source)
