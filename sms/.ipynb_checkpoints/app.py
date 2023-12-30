import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Define the function first
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Load the models and libraries
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Streamlit code
st.title("Email Spam Classifier")
input_sms = st.text_area("Enter message")

if st.button('Predict'):
    # Preprocess
    transformed_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # Predict
    result = model.predict(vector_input)[0]
    
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
