import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv(find_dotenv())
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Get other configuration details from environment variables
INDEX_NAME = os.getenv('INDEX_NAME')
NAMESPACE = os.getenv('NAMESPACE')

if not all([PINECONE_API_KEY, PINECONE_ENV, OPENAI_API_KEY]):
    raise ValueError("Missing one or more environment variables.")

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Constants
CONVERSATION_FILE_PATH = "conversation_history.txt"

# Initialize models
llm = ChatOpenAI(model_name='gpt-4o', temperature=0.5, max_tokens=1024)
embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
doc_db = Pinecone.from_documents('', embeddings, index_name=INDEX_NAME, namespace=NAMESPACE)

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

def main():
    """Main function to run the Streamlit app."""
    st.title("BABot_FAQ")
    st.write("### Que sais-je sur TICADI?")
    # Initialize session state
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
