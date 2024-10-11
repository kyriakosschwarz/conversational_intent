import pandas as pd
import contractions
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

# Assuming you have the necessary files 'trained_model.pickle', 'tfidf_vectorizer.pickle', and 'label_encoder.pickle'


def load_resources():
    # Load the trained model
    with open('artefacts/trained_model.pickle', 'rb') as f:
      model = pickle.load(f)

    # Load the TfidfVectorizer
    with open('artefacts/tfidf_vectorizer.pickle', 'rb') as f:
      tfidf_vectorizer = pickle.load(f)

    # Load the LabelEncoder
    with open('artefacts/label_encoder.pickle', 'rb') as f:
      le = pickle.load(f)

    return model, tfidf_vectorizer, le

model, tfidf_vectorizer, le = load_resources()
print("resources loaded")


# Preprocessing pipeline (same as before)
def preprocess_text(text):
  text = text.lower()
  text = fix_contractions(text)
  text = remove_punctuation(text)
  text = lemmatize_text(text)
  return text

def fix_contractions(text):
  return contractions.fix(text)

def remove_punctuation(text):
  return text.translate(str.maketrans('', '', string.punctuation))

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
  tokens = nltk.word_tokenize(text)
  lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
  return ' '.join(lemmatized_tokens)


def predict_new_sentense(new_sentence):

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

#print(predict_new_sentense("Not this time."))

