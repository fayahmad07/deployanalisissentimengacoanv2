import pandas as pd
import numpy as np
import re
import emoji
import nltk
from nltk.corpus import stopwords
from langdetect import detect
from deep_translator import GoogleTranslator
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from googletrans import Translator
import warnings
import matplotlib.pyplot as plt
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from wordcloud import WordCloud
warnings.filterwarnings("ignore")


# Initialize translator and stemmer
translator = Translator()
google_translator = GoogleTranslator(source='en', target='id')
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load model and vectorizer
from joblib import load
model = load('sentiment_model.joblib')
tfidf_vectorizer = load('tfidf_vectorizer.joblib')

# Dictionary for sentiment labels
sentiment_labels = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}

# Function for preprocessing the text
def preprocess_text(text):
    # Step A.1: Rating converting
    mapping = {'1/5': 'sangat buruk', '2/5': 'buruk', '3/5': 'cukup', '4/5': 'baik', '5/5': 'sangat baik'}
    for key, value in mapping.items():
        text = text.replace(key, value)
    
    # Step A.2: Emoji preprocessing
    text = emoji.demojize(text, delimiters=(" ", ""))
    text = re.sub(r'\s*(\_)\s*', ' ', text)
    
    # Step A.3: Lowercasing, punctuation, numbers, and white space
    text = re.sub(r'[^a-zA-Z\s\u3040-\u30FF\u31F0-\u31FF\u4E00-\u9FFF\uAC00-\uD7AF]', ' ', text)
    text = text.strip().lower()

    # Step B: Translation (non-Latin scripts)
    def translate(text):
        words = text.split()
        translated_words = []
        for word in words:
            if not bool(re.match(r'^[\u0000-\u007F\u0600-\u06FF]+$', word)):  # Non-Latin check
                try:
                    translated_word = translator.translate(word, src='auto', dest='id').text
                    translated_words.append(translated_word)
                except Exception as e:
                    print(f"Error translating word '{word}': {e}")
                    translated_words.append(word)  # Append original word on failure
            else:
                translated_words.append(word)  # Keep original Latin word
        return " ".join(translated_words)
    
    text = translate(text)
    
    # Step C: Standardization (double character)
    def doubel_char(text):
        text = re.sub(r'([a-fh-jl-mo-z])\1+', r'\1', text, flags=re.IGNORECASE)
        text = re.sub(r'(g+|k+|n+)\b', lambda m: m.group(1)[0], text, flags=re.IGNORECASE)
        text = re.sub(r'(?<!\b)(g{3,}|k{3,}|n{3,})(?!\b)', lambda m: m.group(1)[:2], text, flags=re.IGNORECASE)
        return text

    text = doubel_char(text)
    
    # Step C.2: Slang conversion (use a dictionary)
   bagian slang_dict ganti pake code ini buat load-nya:
    slang_dict = {}
    with open('slang_dict.txt', 'r') as f:
        for line in f:
        # Remove whitespace and check if ':' exists
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)  # Split into at most 2 parts
                slang_dict[key.strip()] = value.strip()
            else:
                print(f"Skipping invalid line: {line}")
    with open('slang_dict1.txt', 'r') as f:
        for line in f:
            key, value = line.strip().split(':')
            slang_dict[key] = value

    def slang_norm(text):
        words = text.split()
        return ' '.join([slang_dict.get(word, word) for word in words])

    text = slang_norm(text)
    
    # Step C.3: Remove meaningless words
    del_words = ['rp', 'ny', 'da', 'ah', 'eh', 'pw', 'ob', 'mh', 'se', 'kn', 'ko', 'ni', 'ge', 'na', 'mg', 'du', 'hm', 'ai', 'lt', 'co', 'dn', 'yh', 'wl', 'dh', 'ps', 'ja', 'bm', 'nu', 'ti', 'mm', 'dc', 'de', 'ee', 'oh', 'fa', 'ye', 'pm', 'nd', 'rn', 'ng', 'zu', 'ih', 'fi', 'su', 'dj', 'st', 'ay', 'ii', 'dy', 'ek', 'ne', 'ir', 'oz', 'sx', 'pr', 'wo', 'rs', 'la', 'dt', 'yt', 'eg', 'pn', 'ey', 'mp', 'uy', 'jt', 'lh', 'ea', 'kek', 'kok', 'bah', 'deh', 'nih', 'lek', 'mmk', 'kan', 'lho', 'shay', 'sihh', 'mnto', 'nyaa', 'dong', 'gwsh', 'yaaa', 'yoww', 'myoo', 'nyah', 'yaak', 'pabu', 'kama', 'denk', 'atua', 'donk', 'bisr', 'moai', 'elsi', 'lahh', 'dech', 'nuga', 'leha', 'nggu', 'jrek', 'nong', 'ahhh', 'yahh', 'dehh', 'lohh', 'keun', 'geje', 'siii', 'uhuy', 'lihat', 'terjemahan', 'indonesia']
    text = ' '.join([word for word in text.split() if word not in del_words])

    # Step D: Stemming
    text = stemmer.stem(text)
    
    # Step E: Stopword filtering
    stop_words = set(stopwords.words('indonesian'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

# Predict Sentiment Function
def predict_sentiment(text):
    """Fungsi untuk memprediksi sentimen dari teks."""
    processed_text = preprocess_text(text)
    text_vector = tfidf_vectorizer.transform([processed_text])
    #text_vector = tfidf_vectorizer.transform(text)
    prediction = model.predict(text_vector)
    sentiment_label = sentiment_labels[int(prediction[0])]  # Convert prediction to label
    return sentiment_label

def visualize_wordcloud(data):
    """Fungsi untuk visualisasi WordCloud."""
    # Pastikan data tidak mengandung NaN dan konversi semua entri menjadi string
    valid_data = data.dropna().astype(str)
    combined_text = ' '.join(valid_data)
    wc = WordCloud(width=800, height=400, background_color='white')
    fig, ax = plt.subplots()
    wordcloud = wc.generate(combined_text)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def visualize_pie_chart(sentiments):
    """Fungsi untuk visualisasi pie chart distribusi sentimen."""
    fig, ax = plt.subplots()
    sentiment_counts = sentiments.value_counts(normalize=True)
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

# Streamlit Interface

import streamlit as st

st.title('Aplikasi Analisis Sentimen')

# Form untuk input teks
with st.form("text_input"):
    user_text = st.text_area("Masukkan teks untuk memprediksi sentimennya:")
    submit_text = st.form_submit_button("Prediksi Sentimen")

if submit_text and user_text:
    result = predict_sentiment(user_text)
    st.write(f"Prediksi Sentimen: **{result}**")  # Display sentiment label

# Pengunggahan file CSV dan visualisasi
uploaded_file = st.file_uploader("Unggah file CSV yang berisi teks untuk analisis", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'comment_rests' in data.columns:
        st.write(data.sample(100))
        data['comment_rests'] = data['comment_rests'].fillna('')  # Fill NaN values with an empty string
        data['comment_rests'] = data['comment_rests'].astype(str)  # Convert everything to string

        # Preprocess and predict sentiments
        data['comment_cleaned'] = data['comment_rests'].apply(preprocess_text)
        data['predicted_sentiment'] = data['comment_rests'] .apply(predict_sentiment)
        st.write("Predicted Sentiment Labels:")
        st.write(data[['comment_rests', 'predicted_sentiment']])
        
        if st.button('Tampilkan WordCloud dari Komentar'):
            visualize_wordcloud(data['comment_cleaned'] )
        if st.button('Tampilkan Distribusi Sentimen'):
            visualize_pie_chart(data['predicted_sentiment'])
    else:
        st.error("Pastikan file CSV memiliki kolom 'comment_rests'.")
