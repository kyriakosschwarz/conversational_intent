"""
Module: infer.py

Description: This module handles the loading of machine learning resources and provides functions for text preprocessing and intent prediction.
"""

import pandas as pd
import contractions
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from config_loader import ConfigLoader

nltk_data_dir = ConfigLoader.get('paths.nltk_data_dir')

# Tell NLTK where to look for data
nltk.data.path.append(nltk_data_dir)

# Function to download resource if not available
def download_nltk_resource(resource_name):
    try:
        nltk.data.find(resource_name)
    except LookupError:
        nltk.download(resource_name, download_dir=nltk_data_dir)

# Download 'punkt' and 'wordnet' if not available locally
download_nltk_resource('tokenizers/punkt_tab')
download_nltk_resource('corpora/wordnet')

# Assuming you have the necessary files 'trained_model.pickle', 'tfidf_vectorizer.pickle', and 'label_encoder.pickle'


def load_resources():
    """
    Loads the trained model, TfidfVectorizer, and LabelEncoder from pickle files.

    Returns:
        A tuple containing the loaded model, TfidfVectorizer, and LabelEncoder.
    """

    trained_model_path = ConfigLoader.get('paths.trained_model_file')
    tfidf_vectorizer_path = ConfigLoader.get('paths.tfidf_vectorizer_file')
    label_encoder_path = ConfigLoader.get('paths.label_encoder_file')

    # Load the trained model
    with open(trained_model_path, 'rb') as f:
      model = pickle.load(f)

    # Load the TfidfVectorizer
    with open(tfidf_vectorizer_path, 'rb') as f:
      tfidf_vectorizer = pickle.load(f)

    # Load the LabelEncoder
    with open(label_encoder_path, 'rb') as f:
      le = pickle.load(f)

    return model, tfidf_vectorizer, le

model, tfidf_vectorizer, le = load_resources()


# Preprocessing pipeline (same as before)
def preprocess_text(text):
    """
    Applies a series of preprocessing steps to the input text.

    Args:
        text: The input text to be preprocessed.

    Returns:
        The preprocessed text.
    """
    text = text.lower()
    text = fix_contractions(text)
    text = remove_punctuation(text)
    text = lemmatize_text(text)
    return text

def fix_contractions(text):
    """
    Expands contractions in the given text.

    Args:
        text: The input text containing contractions.

    Returns:
        The text with expanded contractions.
    """
    return contractions.fix(text)

def remove_punctuation(text):
    """
    Removes all punctuation from the given text.

    Args:
        text: The input text containing punctuation.

    Returns:
        The text with all punctuation removed.
    """
    return text.translate(str.maketrans('', '', string.punctuation))

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    """
    Lemmatizes each word in the given text.

    Args:
        text: The input text to be lemmatized.

    Returns:
        The text with all words lemmatized.
    """
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)


def predict_new_sentence(new_sentence):
    """
    Predicts the intent probabilities for a new input sentence.

    Args:
        new_sentence: The input sentence to predict intent for.

    Returns:
        A dictionary containing intent probabilities.
    """
    output_dict = {}

    # Preprocess the new sentence
    processed_sentence = preprocess_text(new_sentence)

    # Transform the processed sentence using the loaded TfidfVectorizer
    tfidf_representation = tfidf_vectorizer.transform([processed_sentence])

    # Predict the class probabilities using the loaded model
    class_probabilities = model.predict_proba(tfidf_representation)

    # Get the predicted class label
    predicted_class_int = model.predict(tfidf_representation)[0]

    # Reverse the label encoding to get the original class name
    predicted_class_original = le.inverse_transform([predicted_class_int])[0]


    for i, probability in enumerate(class_probabilities[0]):
      class_name_original = le.inverse_transform([i])[0]
      output_dict[class_name_original] = round(probability, 3).item()

    return output_dict
