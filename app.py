from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import streamlit as st
import os

st.set_page_config(page_title="News Research Tool", layout="wide")
st.title("📰 News Research Tool")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"News URL {i+1}")
    if url:
        urls.append(url)
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_index"

def get_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": "cpu"}
    )
    if os.path.exists(file_path):
        vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(file_path)
    return vectorstore

if process_url_clicked:
    if not urls:
        st.sidebar.error("Please enter at least one URL.")
    else:
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","], chunk_size=1000, chunk_overlap=100
        )
        docs = text_splitter.split_documents(data)
        vectorstore = get_vectorstore(docs)
        st.sidebar.success("Processing completed!")

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

        # Use a smaller model for CPU
        pipe = pipeline("text-generation", model="tiiuae/falcon-1b-instruct", max_new_tokens=256, temperature=0)
        llm = HuggingFacePipeline(pipeline=pipe)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

        st.info("Processing your query...")
        result = chain({"question": query}, return_only_outputs=True)
        st.subheader("Answer")
        st.write(result["answer"])
        if result.get("sources"):
            st.subheader("Sources")
            for source in result["sources"].split("\n"):
                st.write(source)
