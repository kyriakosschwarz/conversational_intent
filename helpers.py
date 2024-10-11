import streamlit as st
import re
from infer import predict_new_sentense
from db import *

def sanitize_user_input(user_input):
    """
    Sanitizes user input to ensure it's safe for both machine learning inference
    and database storage.

    Args:
        user_input: The user's input string.

    Returns:
        A sanitized version of the input, or None if it's invalid.
    """

    # Remove leading/trailing whitespace
    user_input = user_input.strip()

    # Check for empty input
    if not user_input:
        st.warning("Please enter some text.")
        return None

    # Check for at least one English word
    if not re.search(r"[a-zA-Z]+", user_input):
        st.warning("Please enter at least one English word.")
        return None

    # Limit the number of words to 20
    words = user_input.split()
    if len(words) > 20:
        st.warning("Please enter a maximum of 20 words.")
        return None

    # Remove potentially harmful characters (e.g., SQL injection attempts)
    user_input = re.sub(r"[<>;\"']", "", user_input)

    # Replace multiple spaces with single space
    user_input = re.sub(r"\s+", " ", user_input)

    return user_input

def run_ml_inference(sanitized_input):
    """
    Runs machine learning inference on the sanitized user input.
    This is a placeholder function; replace with your actual ML inference logic.

    Args:
        sanitized_input: The sanitized user input string.

    Returns:
        A string representing the ML model's output or an error message.
    """
    try:
        model_output = predict_new_sentense(sanitized_input)
        
        # Find the class with the highest probability
        max_class = max(model_output, key=model_output.get)
        max_probability = model_output[max_class]

        result = {"max_class": max_class, "max_probability": max_probability}

        # If the highest probability is greater than 0.5, display the class name
        if max_probability > 0.5:
            result["intent_text"] = "Intent detected"
        else:
            result["intent_text"] = "Intent unclear. Please reformulate your input more precisely."
        return result
    except Exception as e:
        st.error(f"An error occurred during machine learning inference: {e}")
        return "Error during ML inference. Please try again later."

def format_ml_result(result):
    output_text = f"{result["intent_text"]}"        

    if(result["intent_text"] == "Intent detected"):        
        output_text = output_text + f": {result["max_class"]} ({result["max_probability"]})"
    return(output_text)

def display_last_entries():
    conn = connect_to_db()
    if conn:
        last_five = get_last_entries(conn, 5)
        st.write("Last Entries:")
        if last_five:
            # Create a table to display the results
            st.table(last_five)
        else:
            st.write("No entries found.")
        conn.close()

def store_results_in_db(user_input, output_text, intent, probability):
    conn = connect_to_db()
    if conn:
        store_input_and_output(conn, user_input, output_text, intent, probability)
        conn.close()  
