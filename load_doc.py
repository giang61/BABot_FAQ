import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone as pcvs
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from fpdf import FPDF  # Import FPDF for text-to-PDF conversion

# Load environment variables
load_dotenv(find_dotenv())

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
INDEX_NAME = os.getenv('INDEX_NAME')
NAMESPACE = os.getenv('NAMESPACE')

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
cloud = os.getenv('PINECONE_CLOUD', 'aws')
region = os.getenv('PINECONE_REGION', 'us-east-1')
index_name = os.getenv('INDEX_NAME')
namespace = os.getenv('NAMESPACE')

# Initialize Pinecone serverless specification
spec = ServerlessSpec(cloud=cloud, region=region)

# Function to clear namespace if it exists
def clear_namespace_if_exists(pinecone_client, index_name, namespace):
    try:
        if index_name in pinecone_client.list_indexes():
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

# Streamlit file uploader
st.title("Document Embedding Processor")
uploaded_files = st.file_uploader("Upload your documents (.txt or .pdf)", type=["pdf", "txt"], accept_multiple_files=True)

@st.cache_resource
def convert_txt_to_pdf(txt_file_path, pdf_file_path):
    """Converts a .txt file to a .pdf file."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Set basic font
    pdf.set_font("Arial", size=12)

    # Open the text file in read mode
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Handle non-ASCII characters by replacing them with empty or compatible chars
            safe_line = line.encode('ascii', 'ignore').decode('ascii')
            pdf.multi_cell(0, 10, safe_line)

    # Save the converted PDF to the specified path
    pdf.output(pdf_file_path)

def load_split_files(uploaded_files):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    if uploaded_files:
        temp_dir = './temp'
        os.makedirs(temp_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            try:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)

                # Save uploaded file locally to temp directory
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Process PDF or text files
                if uploaded_file.type == "application/pdf":
                    loader = PyPDFLoader(temp_file_path)  # File path instead of file object
                    doc_file = loader.load()
                elif uploaded_file.type == "text/plain":
                    # Convert txt file to PDF
                    pdf_file_path = os.path.join(temp_dir, f"{os.path.splitext(uploaded_file.name)[0]}.pdf")
                    convert_txt_to_pdf(temp_file_path, pdf_file_path)
                    # Now process the converted PDF
                    loader = PyPDFLoader(pdf_file_path)
                    doc_file = loader.load()
                else:
                    st.error(f"Unsupported file type: {uploaded_file.type}")
                    continue

                # Split the documents and store embeddings in Pinecone
                doc_file_split = text_splitter.split_documents(doc_file)
                pcvs.from_documents(doc_file_split, embeddings, index_name=INDEX_NAME, namespace=NAMESPACE)

                st.success(f"Processed {uploaded_file.name} successfully.")

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

    return  # Optionally return something if needed

# If files are uploaded, process them
if uploaded_files:
    with st.spinner('Processing uploaded files...'):
        load_split_files(uploaded_files)
