import streamlit as st
import pandas as pd
import joblib
import shap
import re
import string
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")

# Load model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Stopwords
stop_words = set(stopwords.words("english"))

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Load and clean dataset for SHAP background
df_real = pd.read_csv("fake-and-real-news-dataset/True.csv")
df_fake = pd.read_csv("fake-and-real-news-dataset/Fake.csv")
df_real["label"] = 1
df_fake["label"] = 0
df = pd.concat([df_real, df_fake])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["content"] = (df["title"] + " " + df["text"]).apply(clean_text)

# SHAP background
X_background = vectorizer.transform(df["content"][:300])
X_background_dense = X_background.toarray()

# SHAP explainer with feature names
explainer = shap.LinearExplainer(
    model,
    X_background_dense,
    feature_names=vectorizer.get_feature_names_out()
)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("Fake News Detector with SHAP Explainability")
st.markdown("Paste a news article below..")

text_input = st.text_area("Paste your news content here:", height=300)

if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please paste some content.")
    else:
        cleaned_text = clean_text(text_input)
        tfidf_input = vectorizer.transform([cleaned_text])
        tfidf_input_dense = tfidf_input.toarray()
        # Predict
        prediction = model.predict(tfidf_input)[0]
        prediction_label = "REAL" if prediction == 1 else "FAKE"
        st.subheader(f"Prediction: {prediction_label}")
        # SHAP analysis
        shap_values = explainer(tfidf_input_dense)
        st.subheader("SHAP Analysis (Top influencing words)")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)
