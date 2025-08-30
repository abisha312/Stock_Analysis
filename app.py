import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# ------------------------------
# Streamlit page config
# ------------------------------
st.set_page_config(page_title="News Research Tool", layout="wide")
st.title("📰 News Research Tool")
st.sidebar.header("Configuration")

# ------------------------------
# Sidebar input: URLs
# ------------------------------
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"News URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

# ------------------------------
# FAISS index path
# ------------------------------
file_path = "faiss_index"

# ------------------------------
# Function to create/load vectorstore
# ------------------------------
def get_vectorstore(docs):
    try:
        embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large",
            model_kwargs={"device": "cpu"}  # change to "cuda" if GPU is available
        )
        if os.path.exists(file_path):
            st.sidebar.info("Loading existing knowledge base...")
            vectorstore = FAISS.load_local(file_path, embeddings)
        else:
            st.sidebar.write("Creating embeddings (first-time run)...")
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(file_path)
        return vectorstore
    except Exception as e:
        st.sidebar.error(f"❌ Failed to create FAISS index: {e}")
        return None

# ------------------------------
# Process URLs
# ------------------------------
if process_url_clicked:
    if not urls:
        st.sidebar.error("Please enter at least one URL.")
    else:
        st.sidebar.write("Loading articles...")
        loader = UnstructuredURLLoader(urls=urls)
        try:
            data = loader.load()
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load URLs: {e}")
            st.stop()

        st.sidebar.write("Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=1000,
            chunk_overlap=100,
        )
        docs = text_splitter.split_documents(data)

        vectorstore = get_vectorstore(docs)
        if vectorstore:
            st.sidebar.success("✅ Processing completed!")

# ------------------------------
# User query input
# ------------------------------
query = st.text_input("Ask a question about the articles:")

if query:
    if not os.path.exists(file_path):
        st.error("No knowledge base found. Please process URLs first.")
    else:
        embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large",
            model_kwargs={"device": "cpu"}
        )
        vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            max_tokens=500
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
