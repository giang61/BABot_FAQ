import streamlit as st
from connect_postgre import database_response
from streamlit_chat import message

st.set_page_config(page_title="BABot Business Intelligence")

st.title("Interrogez la base PostgreSQL")

if "user_input" not in st.session_state:
    st.session_state['user_input'] = []

if "database_response" not in st.session_state:
    st.session_state['database_response'] = []

user_input = st.chat_input("Votre question:")

if user_input:
    output = database_response(user_input)
    output = output['result']

    st.session_state.database_response.append(user_input)
    st.session_state.user_input.append(output)

message_history = st.empty()

if message_history:
    for i in range(len(st.session_state['user_input'])):
        message(st.session_state['database_response'][i], is_user=True, key = str(i) + "data_by_user")
        message(st.session_state['user_input'][i], key=str(i))