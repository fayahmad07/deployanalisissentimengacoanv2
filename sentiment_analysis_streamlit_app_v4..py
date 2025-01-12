import pandas as pd
import numpy as np
import nltk
import streamlit as st
from nltk.corpus import stopwords
from langdetect import detect
from deep_translator import GoogleTranslator
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from googletrans import Translator
import re
import emoji
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from joblib import load

# Unduh data NLTK yang dibutuhkan
with st.spinner("Mengunduh data NLTK, mohon tunggu..."):
    nltk.download('stopwords')  # Untuk stopwords
    nltk.download('punkt')  # Untuk tokenization
    nltk.download('wordnet')  # Untuk lemmatization
    nltk.download('omw-1.4')  # Untuk WordNet corpus
st.success("Data NLTK berhasil diunduh!")

# Initialize translator and stemmer
translator = Translator()
google_translator = GoogleTranslator(source='en', target='id')
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load model dan vectorizer
model = load('C:/Users/MyMSI/Documents/Kuliah S2/Machine Learning/Deployment/sentiment_model.joblib')
tfidf_vectorizer = load('C:/Users/MyMSI/Documents/Kuliah S2/Machine Learning/Deployment/tfidf_vectorizer.joblib')

# Dictionary for sentiment labels
sentiment_labels = {2: 'Positive', 1: 'Neutral', 0: 'Negative'}

# Function for preprocessing text
def preprocess_text(text):
    # Step A.1: Mapping rating
    mapping = {'1/5': 'sangat buruk', '2/5': 'buruk', '3/5': 'cukup', '4/5': 'baik', '5/5': 'sangat baik'}
    for key, value in mapping.items():
        text = text.replace(key, value)

    # Step A.2: Emoji preprocessing
    text = emoji.demojize(text, delimiters=(" ", ""))
    text = re.sub(r'\s*(\_)\s*', ' ', text)

    # Step A.3: Lowercasing, punctuation, and whitespace
    text = re.sub(r'[^a-zA-Z\s\u3040-\u30FF\u31F0-\u31FF\u4E00-\u9FFF\uAC00-\uD7AF]', ' ', text)
    text = text.strip().lower()

    # Step B: Translation (non-Latin scripts)
    try:
        text = google_translator.translate(text)
    except Exception as e:
        st.error(f"Error in translation: {e}")

    # Step C: Standardization
    def standardize(text):
        text = re.sub(r'([a-z])\1+', r'\1', text)
        return text
    text = standardize(text)

    # Step D: Stemming
    text = stemmer.stem(text)

    # Step E: Stopword filtering
    try:
        stop_words = set(stopwords.words('indonesian'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('indonesian'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

# Predict Sentiment Function
def predict_sentiment(text):
    """Fungsi untuk memprediksi sentimen dari teks."""
    processed_text = preprocess_text(text)
    text_vector = tfidf_vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)
    sentiment_label = sentiment_labels[int(prediction[0])]
    return sentiment_label

# Visualisasi WordCloud
def visualize_wordcloud(data):
    valid_data = data.dropna().astype(str)
    combined_text = ' '.join(valid_data)
    wc = WordCloud(width=800, height=400, background_color='white')
    fig, ax = plt.subplots()
    wordcloud = wc.generate(combined_text)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Streamlit Interface
st.title("Aplikasi Analisis Sentimen")

# Form untuk input teks
with st.form("text_input"):
    user_text = st.text_area("Masukkan teks untuk memprediksi sentimennya:")
    submit_text = st.form_submit_button("Prediksi Sentimen")

if submit_text and user_text:
    result = predict_sentiment(user_text)
    st.write(f"Prediksi Sentimen: **{result}**")

# Pengunggahan file CSV
uploaded_file = st.file_uploader("Unggah file CSV untuk analisis", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if 'comment_rests' in data.columns:
        # Membersihkan data
        data['comment_rests'] = data['comment_rests'].fillna('')  # Isi NaN dengan string kosong
        data['comment_rests'] = data['comment_rests'].astype(str)  # Konversi semua menjadi string

        # Preprocess dan prediksi sentimen
        data['comment_cleaned'] = data['comment_rests'].apply(preprocess_text)
        data['predicted_sentiment'] = data['comment_cleaned'].apply(predict_sentiment)

        # Tampilkan hasil
        st.write("Hasil Analisis Sentimen:")
        st.write(data[['comment_rests', 'comment_cleaned', 'predicted_sentiment']])

        # Visualisasi WordCloud
        if st.button('Tampilkan WordCloud'):
            visualize_wordcloud(data['comment_cleaned'])
    else:
        st.error("Pastikan file memiliki kolom 'comment_rests'.")
