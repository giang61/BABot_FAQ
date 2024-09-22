import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone as pcvs
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set up API keys and configurations
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
cloud = os.getenv('PINECONE_CLOUD', 'aws')
region = os.getenv('PINECONE_REGION', 'us-east-1')
index_name = os.getenv('INDEX_NAME')
namespace = os.getenv('NAMESPACE')

# Initialize Pinecone serverless specification
spec = ServerlessSpec(cloud=cloud, region=region)

# Check if namespace exists and delete it
def clear_namespace_if_exists(pinecone_client, index_name, namespace):
    try:
        # Check if the index exists
        if index_name in pinecone_client.list_indexes().names():
            # Delete all vectors in the namespace if it exists
            pinecone_client.Index(index_name).delete(delete_all=True, namespace=namespace)
            print(f"Namespace '{namespace}' cleared.")
        else:
            print(f"Index '{index_name}' does not exist yet.")
    except Exception as e:
        print(f"Error while clearing namespace '{namespace}': {e}")

# Create index if it does not exist
if index_name not in pc.list_indexes().names():
    pc.create_index(index_name, dimension=1536, metric='cosine', spec=spec)

# Clear the namespace if it exists
clear_namespace_if_exists(pc, index_name, namespace)

# Map file types to loaders
file_loader_map = {
    'csv': CSVLoader,
    'xlsx': UnstructuredExcelLoader,
    'txt': TextLoader,
    'pdf': PyPDFLoader,
}

# Streamlit file uploader
st.title("Document Embedding Processor")
uploaded_files = st.file_uploader("Upload your files", type=["csv", "xlsx", "txt", "pdf"], accept_multiple_files=True)

def process_uploaded_files(uploaded_files, file_loader_map, text_splitter, embeddings, index_name, namespace):
    processed_docs = []

    for uploaded_file in uploaded_files:
        try:
            file_extension = uploaded_file.name.split(".")[-1]
            loader_class = file_loader_map.get(file_extension)

            if loader_class:
                # Save the uploaded file temporarily
                temp_file_path = f"./temp/{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Use appropriate loader for the file
                loader = loader_class(file_path=temp_file_path)
                docs = loader.load()
                docs_split = text_splitter.split_documents(docs)

                # Store the document embeddings in Pinecone
                pcvs.from_documents(docs_split, embeddings, index_name=index_name, namespace=namespace)
                processed_docs.extend(docs_split)

                # Remove the temporary file after processing
                os.remove(temp_file_path)
            else:
                st.error(f"Unsupported file type: {file_extension}")

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            continue

    return processed_docs


# Cache the file processing
@st.cache_resource
def load_split_files(uploaded_files):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')  # Ensure the model is correct

    # Process the uploaded files
    doc_db = process_uploaded_files(uploaded_files, file_loader_map, text_splitter, embeddings, index_name, namespace)

    return doc_db


# If files are uploaded, process them
if uploaded_files:
    with st.spinner('Processing uploaded files...'):
        doc_db = load_split_files(uploaded_files)
        if doc_db:
            st.success(f"Processed {len(doc_db)} documents successfully!")

