"""
Module: helpers.py

Description: This module provides utility functions for input sanitization and formatting of machine learning results.
"""

import streamlit as st
import re

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


def format_ml_result(result):
    """
    Formats the machine learning result into a human-readable string.

    Args:
        result: A dictionary containing the machine learning inference results.

    Returns:
        A formatted string representing the inference result.
    """
    output_text = f"{result["intent_text"]}"        

    if(result["intent_text"] == "Intent detected"):        
        output_text = output_text + f": {result["max_class"]} ({result["max_probability"]})"
    return(output_text)
