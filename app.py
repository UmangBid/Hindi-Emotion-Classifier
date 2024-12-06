import streamlit as st
import joblib
import pandas as pd

model = joblib.load('sentiment_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title('Hindi Sentiment Analysis Web App')

user_input = st.text_area('Enter your text in Hindi')

def analyze_sentiment(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return prediction[0]

if st.button('Analyze Sentiment'):
    if user_input:
        sentiment = analyze_sentiment(user_input)
        st.write('Sentiment:', sentiment)
    else:
        st.warning('Please enter some text for analysis.')


st.sidebar.markdown("Disclaimer: This is a simplified example using a small dataset and may not accurately predict sentiment for all inputs.")


