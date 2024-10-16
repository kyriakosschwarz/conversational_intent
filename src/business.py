"""
Module: business.py

Description: This module contains the business logic for the intent classification system, including ML inference, database interactions, and result handling.
"""

import pandas as pd
from helpers import *
from service import *

def run_ml_inference(sanitized_input):
    """
    Runs machine learning inference on the sanitized user input.

    Args:
        sanitized_input: The sanitized user input string.

    Returns:
        A dictionary containing the ML model's output or an error message.
    """
    try:
        model_output = predict_new_sentence(sanitized_input)
        
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

def display_last_entries(n):
    """
    Retrieves and displays the last five entries from the database in a Streamlit table.
    """
    conn = connect_to_db()
    if conn:
        last_entries = get_last_entries(conn, n)
        st.write("Last Entries:")
        if last_entries:
            df_last_entries = pd.DataFrame(last_entries['data'], columns=last_entries['columns'])
            # Create a table to display the results
            st.table(df_last_entries)
        else:
            st.write("No entries found.")
        conn.close()

def store_results_in_db(user_input, output_text, intent, probability):
    """
    Stores the inference results in the database.

    Args:
        user_input (str): The original user input.
        output_text (str): The output text from the ML model.
        intent (str): The predicted intent.
        probability (float): The probability of the predicted intent.
    """
    conn = connect_to_db()
    if conn:
        store_input_and_output(conn, user_input, output_text, intent, probability)
        conn.close()  

def handle_get_intent_button(user_input):
    """
    Handles the "Get intent" button click event.
    Sanitizes the input, runs ML inference, displays the result, and stores it in the database.

    Args:
        user_input (str): The user's input text.
    """
    if user_input:
        sanitized_input = sanitize_user_input(user_input)
        if sanitized_input:
            result = run_ml_inference(sanitized_input)
            st.info(result)
            st.success(format_ml_result(result))
            store_results_in_db(sanitized_input, result["intent_text"], result["max_class"], result["max_probability"])
    else:
        st.error("Please enter a user utterance.")

def handle_last_entries_button():
    """
    Handles the "Show Last Entries" button click event.
    Displays the last five entries from the database.
    """
    display_last_entries(5)   

def get_dashboard_data():
    """
    Retrieves the last 100 entries from the database in a dataframe.
    """
    conn = connect_to_db()
    if conn:
        last_entries = get_last_entries(conn, 100)
        conn.close()
        if last_entries:
            df_last_entries = pd.DataFrame(last_entries['data'], columns=last_entries['columns'])
            return df_last_entries
        else:
            return None        
    return None

def create_initial_table():
    conn = connect_to_db()
    if conn:
        create_table(conn)  # Create table if it doesn't exist
        conn.close()
        st.success("Initial table created (if not already existed)")
    else:
        st.error("Initial table could not be created")

def plot_probability_hist(data):
    return plot_probability_histogram(data)

def plot_avg_proba(data):
    return plot_avg_probability_by_intent(data)

def get_wordcloud(text, title):
    return create_wordcloud(text, title)
