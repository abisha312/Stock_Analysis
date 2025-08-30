import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.llms import Cohere
from langchain.embeddings.base import Embeddings
import cohere
import os

# ✅ Load Cohere API key from Streamlit secrets
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
co = cohere.Client(COHERE_API_KEY)

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

# ✅ Custom Cohere Embeddings for LangChain
class CohereEmbeddings(Embeddings):
    def embed_documents(self, texts):
        response = co.embed(model="large", texts=texts)
        return response.embeddings

    def embed_query(self, text):
        response = co.embed(model="large", texts=[text])
        return response.embeddings[0]

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
        embeddings = CohereEmbeddings()
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
        embeddings = CohereEmbeddings()
        vectorstore = FAISS.load_local(file_path, embeddings)

        llm = CohereChat(
            cohere_api_key=COHERE_API_KEY,
            model="command-nightly",
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
