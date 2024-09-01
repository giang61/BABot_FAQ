import pandas as pd
import os
import numpy as np
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('.env_EricH'))
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Get other configuration details from environment variables
index_name = os.getenv('INDEX_NAME')
namespace = os.getenv('NAMESPACE')

# Define the path to the CSV file
csv_file_path = r'C:/Users/a/PycharmProjects/BABot/data/Transport/Clickdon simple.csv'
# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path)

llm = ChatOpenAI(model_name='gpt-4o', temperature=0.5, max_tokens=1024)
# Convert the DataFrame to a NumPy array
embeddings = df.to_numpy()
doc_db = Pinecone.from_documents('', embeddings, index_name=index_name, namespace=namespace)

def retrieval_answer(query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=doc_db.as_retriever(),
    )
    query = query
    result = qa.run(query)
    return result

# App with history
def load_conversation_history(file_path):
    try:
        with open(file_path, "r") as file:
            return file.readlines()
    except FileNotFoundError:
        return []

def save_conversation_history(file_path, user_input, bot_response):
    with open(file_path, "a") as file:
        file.write(f"user: {user_input}\n")
        file.write(f"bot: {bot_response}\n")

def display_conversation_history(conversation_history):
    for item in conversation_history:
        if "user" in item:
            st.info(item.replace("user: ", ""))
        elif "bot" in item:
            st.success(item.replace("bot: ", ""))

def main():
    '''
    if index_name=='documentation':
        st.title("Que sais-je sur VIF - TICADI - Proxidon ?")
    elif index_name=='analysis':
    '''
    st.title("Quels sont les renseignements à tirer de mes données ?")

    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Load existing conversation history from file if available
    conversation_file_path = "conversation_history.txt"
    if not st.session_state.conversation_history:
        st.session_state.conversation_history = load_conversation_history(conversation_file_path)

    # Display conversation history
    display_conversation_history(st.session_state.conversation_history)

    # User input box
    user_input = st.chat_input("Posez votre question:")

    # Check if user input is not empty
    if user_input:
        # Get chatbot response
        bot_response = retrieval_answer(user_input)

        # Display the latest user input and chatbot response
        st.info(user_input)
        st.success(bot_response)

        # Add messages to the conversation history
        st.session_state.conversation_history.extend([f"user: {user_input}\n", f"bot: {bot_response}\n"])

        # Save conversation history to file
        save_conversation_history(conversation_file_path, user_input, bot_response)

# Run the Streamlit app
if __name__ == "__main__":
    main()

