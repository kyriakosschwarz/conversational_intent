import pandas as pd
import contractions
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
from sklearn import preprocessing
import pickle

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')


# Assuming 'data.json' is in the current working directory
df = pd.read_json('data/data.json')


# Remove duplicate rows based on 'intent' and 'text' columns, keeping the first occurrence
df = df.drop_duplicates(subset=['intent', 'text'], keep='first')


# Function to fix contractions in a text string
def fix_contractions(text):
  return contractions.fix(text)


# Function to remove punctuation from a text string
def remove_punctuation(text):
  return text.translate(str.maketrans('', '', string.punctuation))


# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# Function to lemmatize a text string
def lemmatize_text(text):
  tokens = nltk.word_tokenize(text)
  lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
  return ' '.join(lemmatized_tokens)


# Preprocessing pipeline
def preprocess_text(text):
  text = text.lower()  # Convert to lowercase
  text = fix_contractions(text)
  text = remove_punctuation(text)
  text = lemmatize_text(text)
  return text


df['text_processed'] = df['text'].apply(preprocess_text)

# Create a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the 'text_processed' column
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text_processed'])


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix, df['intent'], test_size=0.2, random_state=42
)


# Calculate class weights to address class imbalance
le = preprocessing.LabelEncoder()
y_train_int = le.fit_transform(y_train)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train_int), y=y_train_int
)

# Create a dictionary mapping class labels to their weights
class_weights_dict = dict(enumerate(class_weights))

# Train a Logistic Regression model with class weights
model = LogisticRegression(class_weight=class_weights_dict)
model.fit(X_train, y_train_int)


# Make predictions on the test set
y_test_int = le.transform(y_test)
y_pred = model.predict(X_test)

# Evaluate the model
y_test_original = le.inverse_transform(y_test_int)
y_pred_original = le.inverse_transform(y_pred)

# Create a new classification report with original class names
report_original = classification_report(y_test_original, y_pred_original)

print(f"Classification Report with Original Class Names:\n{report_original}")


# Save the trained model, tfidf_vectorizer and labelencoder to files
filename = 'artefacts/trained_model.pickle'
pickle.dump(model, open(filename, 'wb'))

filename = 'artefacts/tfidf_vectorizer.pickle'
pickle.dump(tfidf_vectorizer, open(filename, 'wb'))

filename = 'artefacts/label_encoder.pickle'
pickle.dump(le, open(filename, 'wb'))

