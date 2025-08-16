import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings   # ✅ updated imports
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.title("Stock Analysis Chatbot 📈")
st.sidebar.title("Settings")

# URLs input
urls = st.sidebar.text_area("Enter article URLs (comma separated)")
process_url_clicked = st.sidebar.button("Process URLs")

# Vector store filename
VECTOR_STORE_PATH = "vectorstore.pkl"

if process_url_clicked:
    with st.spinner("Loading and processing documents..."):
        try:
            url_list = [url.strip() for url in urls.split(",") if url.strip()]
            loader = UnstructuredURLLoader(urls=url_list)
            data = loader.load()

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", "?", "!"],
                chunk_size=1000,
                chunk_overlap=150
            )
            docs = text_splitter.split_documents(data)

            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectorstore = FAISS.from_documents(docs, embeddings)

            # Save vectorstore
            with open(VECTOR_STORE_PATH, "wb") as f:
                pickle.dump(vectorstore, f)

            st.success("Documents processed and saved!")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Ask questions
query = st.text_input("Ask a question about the documents")
if query:
    if os.path.exists(VECTOR_STORE_PATH):
        with st.spinner("Searching for an answer..."):
            try:
                with open(VECTOR_STORE_PATH, "rb") as f:
                    vectorstore = pickle.load(f)

                llm = ChatOpenAI(
                    temperature=0.7,
                    model="gpt-3.5-turbo",
                    openai_api_key=openai_api_key,
                    max_tokens=500
                )

                chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=llm,
                    retriever=vectorstore.as_retriever()
                )

                result = chain({"question": query}, return_only_outputs=True)

                st.subheader("Answer")
                st.write(result["answer"])

                if result.get("sources"):
                    st.subheader("Sources")
                    st.write(result["sources"])

            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please process URLs first.")
