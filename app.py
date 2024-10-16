"""
Module: app.py

Description: This module serves as the main entry point for the Streamlit-based intent classification application. It sets up the user interface and handles user interactions.
"""

import streamlit as st
from business import *


st.title("Intent Classification")

# App description - Explain functionalities in an expander box
with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app can predict an intent based on an input utterance. Furthermore, the last five predictions can be displayed in form of a table.')
  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, 1. Enter a new text into the text box and then press the "Get intent" button. 2. Display the last five predictions by pressing the "Show Last Entries" button.')


user_input = st.text_input("Enter user utterance:")

# Button to retrieve inference
if st.button("Get intent"):
    handle_get_intent_button(user_input)

# Button to retrieve and display last entries
if st.button("Show Last Entries"):
    handle_last_entries_button()
