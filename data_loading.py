import os
from typing import List, Optional, Any
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
# from config import GOOGLE_API_KEY, USER_AGENT
google_api_key="AIzaSyAw4gUH6WEbGZaBg6KSLoYfU_djx-mEjxY"
# Set default USER_AGENT
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"



embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
    )

headers = {
    "User-Agent": os.environ.get("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
}

def doc_list():
    docs_list = []
    urls = [
            "https://github.com/spheronFdn/docs/blob/main/examples/protocol/icl-example.yaml",
            "https://docs.spheron.network/user-guide/icl?utm_source=chatgpt.com",
            "https://github.com/spheronFdn/docs/blob/main/examples/protocol/icl-multiservice-example.yaml",
        ]
    for url in urls:
        try:
            loader = WebBaseLoader(url,header_template=headers)
            docs = loader.load()
            docs_list.extend(docs)
        except Exception as e:
            print(f"Error loading URL {url}: {str(e)}")
            continue
    return docs_list

def vectordb(docs_list,embedding):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500,  # Larger chunks for better YAML context
            chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Force local mode by specifying persist_directory
        return Chroma.from_documents(
            documents=doc_splits,
            embedding=embeddings,
            collection_name="icl-docs",
            persist_directory="./chroma_db"  # Ensure Chroma is stored locally
        )