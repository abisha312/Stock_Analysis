import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, logging

# Suppress warnings from transformers
logging.set_verbosity_error()

# ---------------- Streamlit Page ----------------
st.set_page_config(page_title="News Research Tool", layout="wide")
st.title("📰 News Research Tool")

# ---------------- Sidebar Inputs ----------------
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"News URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_index"

# ---------------- FAISS Loader/Creator ----------------
def get_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="hkunlp/instructor-small",  # smaller for CPU
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

# ---------------- Process URLs ----------------
if process_url_clicked:
    if not urls:
        st.sidebar.error("Please enter at least one URL.")
    else:
        try:
            st.sidebar.info("Loading articles...")
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            st.sidebar.info("Splitting text...")
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", ","],
                chunk_size=3000,  # larger chunks reduce number of embeddings
                chunk_overlap=200
            )
            docs = text_splitter.split_documents(data)

            vectorstore = get_vectorstore(docs)
            st.sidebar.success("Processing completed!")
        except Exception as e:
            st.sidebar.error(f"Error processing URLs: {e}")

# ---------------- User Query ----------------
query = st.text_input("Ask a question about the articles:")

if query:
    if not os.path.exists(file_path):
        st.error("No knowledge base found. Please process URLs first.")
    else:
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="hkunlp/instructor-small",
                model_kwargs={"device": "cpu"}
            )
            vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)

            # Use a small CPU-friendly model
            pipe = pipeline(
                "text-generation",
                model="bigscience/bloom-560m",  # free, CPU-friendly
                max_new_tokens=256,             # limit output
                temperature=0
            )
            llm = HuggingFacePipeline(pipeline=pipe)

            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )

            st.info("Processing your query...")
            result = chain({"question": query}, return_only_outputs=True)

            # Limit answer to ~500 words
            answer_text = result["answer"]
            words = answer_text.split()
            if len(words) > 500:
                answer_text = " ".join(words[:500]) + "..."

            st.subheader("Answer")
            st.write(answer_text)

            if result.get("sources"):
                st.subheader("Sources")
                for source in result["sources"].split("\n"):
                    st.write(source)

        except Exception as e:
            st.error(f"Error generating answer: {e}")
