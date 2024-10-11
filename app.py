import streamlit as st
from helpers import *
import infer

# Initialize session state to store the previous input
if 'prev_user_input' not in st.session_state:
    st.session_state.prev_user_input = ""

st.title("Intent Classification")

user_input = st.text_input("Enter user utterance:")

if user_input != st.session_state.prev_user_input:

    sanitized_input = sanitize_user_input(user_input)
    if sanitized_input:
        result = run_ml_inference(sanitized_input)
        st.write(format_ml_result(result))
        store_results_in_db(sanitized_input, result["intent_text"], result["max_class"], result["max_probability"])
    
    # Update the stored user input in session state
    st.session_state.prev_user_input = user_input
                

# Button to retrieve and display last entries
if st.button("Show Last Entries"):
    display_last_entries()     
