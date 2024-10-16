"""
This module provides service functions for handling database operations and intent prediction.

It includes functions for retrieving and storing data in the database, as well as predicting intents for new sentences.
"""

from db import get_last_entries as db_get_last_entries, store_input_and_output as db_store_input_and_output, connect_to_db as db_connect_to_db, create_table as db_create_table
from infer import predict_new_sentence as infer_predict_new_sentence
from plots import plot_probability_histogram as plots_plot_probability_histogram, plot_avg_probability_by_intent as plots_plot_avg_probability_by_intent, create_wordcloud as plots_create_wordcloud

def get_last_entries(connection, limit):
    """
    Fetch the last 'limit' number of entries from the database.
    
    Parameters:
        connection (sqlite3.Connection): The database connection object.
        limit (int): The number of last entries to fetch from the database.

    Returns:
        list: A list of the last 'limit' entries from the database, or None if an error occurs.
    """
    try:
        return db_get_last_entries(connection, limit)
    except Exception as e:
        print(f"Error in get_last_entries: {e}")
        return None

def store_input_and_output(connection, user_input, output_text, intent, probability):
    """
    Store the user input and corresponding model output into the database.

    Parameters:
        connection (sqlite3.Connection): The database connection object.
        user_input (str): The original input text from the user.
        output_text (str): The output text produced by the model.
        intent (str): The predicted intent of the input.
        probability (float): The probability/confidence score of the predicted intent.

    Returns:
        bool: True if the data was stored successfully, False otherwise.
    """
    try:
        db_store_input_and_output(connection, user_input, output_text, intent, probability)
        return True
    except Exception as e:
        print(f"Error in store_input_and_output: {e}")
        return False

def predict_new_sentence(sanitized_input):
    """
    Predict the output for a new input sentence using the inference model.
    
    Parameters:
        sanitized_input (str): The sanitized input text to predict the output for.

    Returns:
        dict: A dictionary containing the model's predicted output, intent, and probability, 
              or None if an error occurs.
    """
    try:
        return infer_predict_new_sentence(sanitized_input)
    except Exception as e:
        print(f"Error in predict_new_sentence: {e}")
        return None

def connect_to_db():
    """
    Establish and return a connection to the database.
    
    Returns:
        sqlite3.Connection: The database connection object, or None if an error occurs.
    """
    try:
        return db_connect_to_db()
    except Exception as e:
        print(f"Error in connect_to_db: {e}")
        return None

def create_table(conn):
    db_create_table(conn)

def plot_probability_histogram(data):
    return plots_plot_probability_histogram(data)

def plot_avg_probability_by_intent(data):
    return plots_plot_avg_probability_by_intent(data)

def create_wordcloud(text, title):
    return plots_create_wordcloud(text, title)
