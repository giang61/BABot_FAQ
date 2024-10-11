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

# Streamlit page configuration
st.set_page_config(page_title="BABot_FAQ", page_icon=":white_check_mark:", layout="centered")

# Constants
CONVERSATION_FILE_PATH = "conversation_history.txt"

# Cache resource to load environment variables
@st.cache_resource
def load_env_vars():
    load_dotenv(find_dotenv('.env'))
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENV = os.getenv('PINECONE_ENV')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    INDEX_NAME = os.getenv('INDEX_NAME')
    NAMESPACE = os.getenv('NAMESPACE')
    return PINECONE_API_KEY, PINECONE_ENV, OPENAI_API_KEY, INDEX_NAME, NAMESPACE

# Cache OpenAI embeddings initialization
@st.cache_resource
def init_openai_embeddings(api_key):
    return OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

# Cache LLM initialization
@st.cache_resource
def init_llm():
    return ChatOpenAI(model_name='gpt-4o-2024-05-13', temperature=0.5, max_tokens=1024)

@st.cache_resource
def initialize_pinecone_client():
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Get cloud and region from environment variables
    cloud = os.getenv('PINECONE_CLOUD', 'aws')
    region = os.getenv('PINECONE_REGION', 'us-east-1')

    # Initialize Pinecone serverless specification
    spec = ServerlessSpec(cloud=cloud, region=region)

    return pc, spec

# Load environment variables
PINECONE_API_KEY, PINECONE_ENV, OPENAI_API_KEY, INDEX_NAME, NAMESPACE = load_env_vars()

# Now call the cached function
pc, spec = initialize_pinecone_client()

# Clear namespace function
def clear_namespace_if_exists(pinecone_client, index_name, namespace):
    try:
        index_names = [index['name'] for index in pinecone_client.list_indexes()]
        if index_name in index_names:
            pinecone_client.Index(index_name).delete(delete_all=True, namespace=namespace)
            print(f"Namespace '{namespace}' cleared.")
        else:
            print(f"Namespace '{namespace}' does not exist yet.")
    except Exception as e:
        print(f"Error while clearing namespace '{namespace}': {e}")

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(INDEX_NAME, dimension=1536, metric='cosine', spec=spec)

# Initialize models
llm = init_llm()
embeddings = init_openai_embeddings(OPENAI_API_KEY)
doc_db = pcvs.from_documents('', embeddings, index_name=INDEX_NAME, namespace=NAMESPACE)

def retrieval_answer(query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=doc_db.as_retriever(),
    )
    return qa.run(query)

# Chatbot response logic
def chatbot_response(input_text):
    return input_text  # Placeholder for actual response logic

# Load and save conversation history functions
def load_conversation_history():
    try:
        with open(CONVERSATION_FILE_PATH, "r") as file:
            return file.readlines()
    except FileNotFoundError:
        return []

def save_conversation_history(user_input, bot_response):
    with open(CONVERSATION_FILE_PATH, "a") as file:
        file.write(f"user: {user_input}\n")
        file.write(f"BABot: {bot_response}\n")

def reset_conversation_history():
    """Reinitialize conversation file to a new empty file."""
    # Create and initialize an empty new file
    with open(CONVERSATION_FILE_PATH, "w") as file:
        file.write("")  # Empty content
    st.success(f"L'historique des interrogations re initialisé")

# PDF conversion function (cached)
@st.cache_resource
def convert_txt_to_pdf(txt_file_path, pdf_file_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    with open(txt_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            safe_line = line.encode('ascii', 'ignore').decode('ascii')
            pdf.multi_cell(0, 10, safe_line)

    pdf.output(pdf_file_path)

# Document splitting and loading
def load_split_files(uploaded_files):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    if uploaded_files:
        temp_dir = './temp'
        os.makedirs(temp_dir, exist_ok=True)
        for uploaded_file in uploaded_files:
            try:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if uploaded_file.type == "application/pdf":
                    loader = PyPDFLoader(temp_file_path)
                    doc_file = loader.load()
                elif uploaded_file.type == "text/plain":
                    pdf_file_path = os.path.join(temp_dir, f"{os.path.splitext(uploaded_file.name)[0]}.pdf")
                    convert_txt_to_pdf(temp_file_path, pdf_file_path)
                    loader = PyPDFLoader(pdf_file_path)
                    doc_file = loader.load()
                else:
                    st.error(f"Unsupported file type: {uploaded_file.type}")
                    continue
                doc_file_split = text_splitter.split_documents(doc_file)
                pcvs.from_documents(doc_file_split, embeddings, index_name=INDEX_NAME, namespace=NAMESPACE)
                st.success(f"Vectorization of {uploaded_file.name} completed!")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
    return

# Main app function
def main():
    st.title("BABot_FAQ")
    st.write(f"### Que voulez-vous savoir sur (les documents dans) {NAMESPACE} ?")

    upload_choice = st.radio("Souhaitez-vous télécharger de nouveaux documents pour l'analyse?", ('OUI', 'NON'))

    # Ask if the user wants to upload a new document
    if upload_choice == 'OUI':
        uploaded_files = st.file_uploader("Téléchargez vos documents (.txt ou .pdf)", type=["pdf", "txt"], accept_multiple_files=True)
        if uploaded_files:
            with st.spinner('BABot est en train de travailler sur vos documents ...'):
                # Clear the conversation history
                reset_conversation_history()
                # Clear the namespace if it exists
                clear_namespace_if_exists(pc, INDEX_NAME, NAMESPACE)

                load_split_files(uploaded_files)
                st.success("L'ensemble de documents bien pris en charge par BABot. Cliquez sur le bouton NON en haut pour démarrer l'analyse.")
        return

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
    user_input = st.chat_input("Posez votre question.")
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

if __name__ == "__main__":
    main()
