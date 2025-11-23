import streamlit as st
import re
import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

download_nltk_resources()

model = joblib.load('log_reg_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

stop_words = set(stopwords.words('english'))
lammetizer = WordNetLemmatizer()

def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ' , text)
    words = word_tokenize(text)
    words = [lammetizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

st.title('Amazon Review Sentiment Analyzer')

review = st.text_area('Enter Product Review : ')

if st.button("Analyze"):
    if not review.strip():
        st.warning("Please first enter the review")
    else:
        cleaned = clean_text(review)
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)

        st.subheader("Sentiment : " + ("Positive 😊" if pred == 1 else "Negative 😡"))



