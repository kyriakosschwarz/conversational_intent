"""
Module: train.py

Description: This module handles the training of the intent classification model, including data preprocessing, model selection, and evaluation.
"""

import pandas as pd
import contractions
import string
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, f1_score
import pickle
from config_loader import ConfigLoader

nltk.download('punkt_tab')
nltk.download('wordnet')

# Load data
data_file_path = ConfigLoader.get('paths.data_file')
df = pd.read_json(data_file_path)

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

# Initialize the WordNetLemmatizer
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

def preprocess_text(text):
    """
    Applies a series of preprocessing steps to the input text.

    Args:
        text: The input text to be preprocessed.

    Returns:
        The preprocessed text.
    """
    text = text.lower()  # Convert to lowercase
    text = fix_contractions(text)
    text = remove_punctuation(text)
    text = lemmatize_text(text)
    return text

df['text_processed'] = df['text'].apply(preprocess_text)

# Remove duplicate rows based on 'intent' and 'text' columns, keeping the first occurrence
df = df.drop_duplicates(subset=['intent', 'text_processed'], keep='first')

# Create a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the 'text_processed' column
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text_processed'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix, df['intent'], test_size=0.2, stratify=df['intent'], random_state=42
)

# For cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize LabelEncoder
le = LabelEncoder()
y_train_int = le.fit_transform(y_train)

# Calculate class weights to address class imbalance
class_weights = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train_int), y=y_train_int
)
class_weights_dict = dict(enumerate(class_weights))

# Initialize variables to store the best model and its score
best_model = None
best_f1_score = 0

# Perform cross-validation
for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train_int), 1):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train_int[train_index], y_train_int[val_index]

    # Train a Logistic Regression model with class weights
    model = LogisticRegression(class_weight=class_weights_dict)
    model.fit(X_train_fold, y_train_fold)

    # Make predictions on the validation set
    y_pred_fold = model.predict(X_val_fold)

    # Calculate macro-average F1-score
    f1 = f1_score(y_val_fold, y_pred_fold, average='macro')

    print(f"Fold {fold} - Macro-average F1-score: {f1:.4f}")

    # Update the best model if current fold has a better F1-score
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model = model
    
    # Print fold results
    print(classification_report(y_val_fold, y_pred_fold, zero_division=0.0))
    print()

print(f"\nBest macro-average F1-score: {best_f1_score:.4f}")

# Train the best model on the entire training set
best_model.fit(X_train, y_train_int)

# Make predictions on the test set
y_test_int = le.transform(y_test)
y_pred = best_model.predict(X_test)

# Evaluate the model
y_test_original = le.inverse_transform(y_test_int)
y_pred_original = le.inverse_transform(y_pred)

# Create a new classification report with original class names
report_original = classification_report(y_test_original, y_pred_original, zero_division=0.0)

print(f"\nClassification Report with Original Class Names:\n{report_original}")

# Save the trained model, tfidf_vectorizer and labelencoder to files
filename = 'artefacts/trained_model.pickle'
pickle.dump(best_model, open(filename, 'wb'))

filename = 'artefacts/tfidf_vectorizer.pickle'
pickle.dump(tfidf_vectorizer, open(filename, 'wb'))

filename = 'artefacts/label_encoder.pickle'
pickle.dump(le, open(filename, 'wb'))
