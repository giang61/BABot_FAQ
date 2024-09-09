import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone as pcvs
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv(find_dotenv('.env'))

# Get necessary API keys and configurations
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set OpenAI API key as environment variable
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

index_name = os.getenv('INDEX_NAME')
namespace = os.getenv('NAMESPACE')
folder_path = os.getenv('FOLDER_PATH')

# configure client
pc = Pinecone(api_key=PINECONE_API_KEY)
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)

# check if index already exists (it shouldn't if this is first time)
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embedding-ada-002
        metric='cosine',
        spec=spec
    )

# Map file types to their respective loader classes
file_loader_map = {
    '**/*.csv': CSVLoader,
    '**/*.xlsx': UnstructuredExcelLoader,
    '**/*.txt': TextLoader,
    '**/*.pdf': PyPDFLoader,
}



def process_files_from_folder(folder_path, file_loader_map, text_splitter, embeddings, index_name, namespace):
    processed_docs = []

    for file_glob, loader_class in file_loader_map.items():
        try:
            # Use DirectoryLoader with the appropriate loader class for each file type
            loader = DirectoryLoader(path=folder_path, glob=file_glob, loader_cls=loader_class, show_progress=True)
            docs = loader.load()
            docs_split = text_splitter.split_documents(docs)
            pcvs.from_documents(docs_split, embeddings, index_name=index_name, namespace=namespace)
            processed_docs.extend(docs_split)
        except Exception as e:
            print(f"Error processing files matching {file_glob}: {e}")
            continue

    return processed_docs

@st.cache_resource  # (optional, based on use case)
def load_split_files(index_name, folder_path):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    # Process all supported files in the folder
    doc_db = process_files_from_folder(folder_path, file_loader_map, text_splitter, embeddings, index_name, namespace)

    return doc_db

# Load and process documents from the folder
doc_db = load_split_files(index_name, folder_path)
