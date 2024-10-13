import streamlit as st
from business import *


st.title("Intent Classification")

user_input = st.text_input("Enter user utterance:")

# Button to retrieve inference
if st.button("Get intent"):
    handle_get_intent_button(user_input)

# Button to retrieve and display last entries
if st.button("Show Last Entries"):
    handle_last_entries_button()
