import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
import os

# ✅ Must be first Streamlit command
st.set_page_config(page_title="News Research Tool", layout="wide")

st.title("📰 News Research Tool")
st.sidebar.header("Configuration")

# Load OpenAI API key safely from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")
api_key = os.environ.get("OPENAI_API_KEY")

# Debug: check if key is loaded
if not api_key:
    st.sidebar.error("❌ OpenAI API key not found. Please check Streamlit secrets.")
    st.stop()
else:
    st.sidebar.success("✅ OpenAI API key loaded successfully.")
    st.sidebar.write("API Key preview:", api_key[:4] + "...")  # safe preview

# Sidebar input for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"News URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

# FAISS index file path
file_path = "faiss_index"

# Function to create/load vectorstore
def get_vectorstore(docs):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    except Exception as e:
        st.error(f"❌ Failed to create embeddings: {e}")
        st.stop()

    if os.path.exists(file_path):
        st.sidebar.info("Loading existing knowledge base...")
        try:
            vectorstore = FAISS.load_local(
                file_path, embeddings, allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"❌ Failed to load FAISS index: {e}")
            st.stop()
    else:
        st.sidebar.info("Creating embeddings (first-time run)...")
        try:
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(file_path)
        except Exception as e:
            st.error(f"❌ Failed to create FAISS index: {e}")
            st.stop()

    return vectorstore

# Process URLs
if process_url_clicked:
    if not urls:
        st.sidebar.error("Please enter at least one URL.")
    else:
        st.sidebar.info("Loading articles...")
        try:
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()
        except Exception as e:
            st.error(f"❌ Failed to load articles: {e}")
            st.stop()

        st.sidebar.info("Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=1000,
            chunk_overlap=100,
        )
        docs = text_splitter.split_documents(data)

        # Create/load FAISS vectorstore
        vectorstore = get_vectorstore(docs)
        st.sidebar.success("✅ Processing completed!")

# User query input
query = st.text_input("Ask a question about the articles:")

if query:
    if not os.path.exists(file_path):
        st.error("No knowledge base found. Please process URLs first.")
        st.stop()

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.load_local(
            file_path, embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"❌ Failed to load FAISS index: {e}")
        st.stop()

    try:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            max_tokens=500
        )
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize LLM chain: {e}")
        st.stop()

    st.info("Processing your query...")
    try:
        result = chain({"question": query}, return_only_outputs=True)
        st.subheader("Answer")
        st.write(result["answer"])

        if result.get("sources"):
            st.subheader("Sources")
            sources = result["sources"].split("\n")
            for source in sources:
                st.write(source)
    except Exception as e:
        st.error(f"❌ Failed to process query: {e}")
