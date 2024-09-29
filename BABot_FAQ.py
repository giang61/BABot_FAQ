import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Pinecone as pcvs
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from fpdf import FPDF  # Import FPDF for text-to-PDF conversion
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv(find_dotenv('.env'))
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
            print(f"Namespace '{namespace}' does not exist yet.")
    except Exception as e:
        print(f"Error while clearing namespace '{namespace}': {e}")

# Create index if it does not exist
if index_name not in pc.list_indexes().names():
    pc.create_index(index_name, dimension=1536, metric='cosine', spec=spec)

# Clear the namespace if it exists
clear_namespace_if_exists(pc, index_name, namespace)

# Streamlit page configuration
st.set_page_config(page_title="BABot_FAQ", page_icon=":white_check_mark:", layout="wide")

# Constants
CONVERSATION_FILE_PATH = "conversation_history.txt"

# Initialize models
llm = ChatOpenAI(model_name='gpt-4o', temperature=0.5, max_tokens=1024)
embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
doc_db = pcvs.from_documents('', embeddings, index_name=INDEX_NAME, namespace=NAMESPACE)

def retrieval_answer(query):
    """Retrieve answer from the model based on the query."""
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=doc_db.as_retriever(),
    )
    return qa.run(query)

def chatbot_response(input_text):
    """Return the chatbot's response."""
    return input_text  # Placeholder for actual response logic

def load_conversation_history():
    """Load conversation history from a file."""
    try:
        with open(CONVERSATION_FILE_PATH, "r") as file:
            return file.readlines()
    except FileNotFoundError:
        return []

def save_conversation_history(user_input, bot_response):
    """Save conversation history to a file."""
    with open(CONVERSATION_FILE_PATH, "a") as file:
        file.write(f"user: {user_input}\n")
        file.write(f"BABot: {bot_response}\n")

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
            safe_line = line.encode('ascii', 'ignore').decode('ascii')  # Replace non-ASCII characters
            pdf.multi_cell(0, 10, safe_line)

    # Save the converted PDF to the specified path
    pdf.output(pdf_file_path)

def load_split_files(uploaded_files):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

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
                    loader = PyPDFLoader(temp_file_path)
                    doc_file = loader.load()
                elif uploaded_file.type == "text/plain":
                    # Convert txt file to PDF
                    pdf_file_path = os.path.join(temp_dir, f"{os.path.splitext(uploaded_file.name)[0]}.pdf")
                    convert_txt_to_pdf(temp_file_path, pdf_file_path)
                    # Now process the converted PDF
                    loader = PyPDFLoader(pdf_file_path)
                    doc_file = loader.load()
                else:
                    st.error(f"Type de fichier non supporté: {uploaded_file.type}")
                    continue

                # Split the documents and store embeddings in Pinecone
                doc_file_split = text_splitter.split_documents(doc_file)
                pcvs.from_documents(doc_file_split, embeddings, index_name=INDEX_NAME, namespace=NAMESPACE)

                st.success(f"Vectorisation de {uploaded_file.name} bien terminée!")

            except Exception as e:
                st.error(f"Erreur de vectorisation {uploaded_file.name}: {e}")

    return  # Optionally return something if needed

def main():
    """Main function to run the Streamlit app."""
    st.title("BABot_FAQ")
    st.write(f"### Que voulez-vous savoir sur (les documents dans) {NAMESPACE} ?")

    # Ask if the user wants to upload a new document
    upload_choice = st.radio(
        "Souhaitez-vous télécharger de nouveaux documents pour l'analyse?",
        ('OUI', 'NON')
    )

    if upload_choice == 'OUI':
        # st.info("Télécharger de nouveaux documents.")
        uploaded_files = st.file_uploader("Téléchargez vos documents (.txt ou .pdf)", type=["pdf", "txt"], accept_multiple_files=True)

        if uploaded_files:
            with st.spinner('BABot est en train de travailler sur vos documents ...'):
                load_split_files(uploaded_files)
                st.success("L'ensemble de documents bien pris en charge par BABot. Cliquez sur le bouton NON en haut pour démarrer l'analyse.")
        return  # Stop further execution until document is uploaded

    # Proceed with the chatbot for querying the existing document
    st.info("Historique des interrogations:")

    # Initialize session state for conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = load_conversation_history()

    # Display conversation history
    for item in st.session_state.conversation_history:
        item_display = item.replace(":newligne:", "\n")
        if "user" in item:
            st.info(item_display.replace("user: ", ""))
        elif "BABot" in item:
            st.success(item_display.replace("BABot: ", ""))

    # User input box
    user_input = st.chat_input("Posez votre question")

    # Check if user input is not empty
    if user_input:
        bot_response = chatbot_response(retrieval_answer(user_input))

        # Format messages for history
        user_input_formatted = user_input.replace("\n", ":newligne:")
        bot_response_formatted = bot_response.replace("\n", ":newligne:")

        # Update conversation history
        st.session_state.conversation_history.append(f"user: {user_input_formatted}")
        st.session_state.conversation_history.append(bot_response_formatted)

        # Display latest messages
        st.info(user_input.replace(":newligne:", "\n"))
        st.success(bot_response.replace(":newligne:", "\n"))

        # Save to file
        save_conversation_history(user_input_formatted, bot_response_formatted)

# Run the Streamlit app
if __name__ == "__main__":
    main()

