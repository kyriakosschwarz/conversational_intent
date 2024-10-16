import pandas as pd
from helpers import *
from service import *

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

def display_last_entries():
    conn = connect_to_db()
    if conn:
        last_five = get_last_entries(conn, 5)
        st.write("Last Entries:")
        if last_five:
            df_last_five = pd.DataFrame(last_five['data'], columns=last_five['columns'])
            # Create a table to display the results
            st.table(df_last_five)
        else:
            st.write("No entries found.")
        conn.close()

def store_results_in_db(user_input, output_text, intent, probability):
    conn = connect_to_db()
    if conn:
        store_input_and_output(conn, user_input, output_text, intent, probability)
        conn.close()  

def handle_get_intent_button(user_input):
    if user_input:
        sanitized_input = sanitize_user_input(user_input)
        if sanitized_input:
            result = run_ml_inference(sanitized_input)
            st.write(format_ml_result(result))
            store_results_in_db(sanitized_input, result["intent_text"], result["max_class"], result["max_probability"])

def handle_last_entries_button():
    display_last_entries()     
