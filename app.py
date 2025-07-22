import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title("Fake News Detection")
st.write("Upload both Fake.csv and True.csv to train the model, ya demo mode use karein.")

fake_file = st.file_uploader("Upload Fake.csv", type="csv")
true_file = st.file_uploader("Upload True.csv", type="csv")
use_demo = st.checkbox("Demo data use karein (sample data)")

if (fake_file and true_file) or use_demo:
    if use_demo:
        # Demo data (zyada aur diverse)
        fake_df = pd.DataFrame({
            'text': [
                'Aliens landed in New York City.',
                'COVID-19 vaccine causes microchips.',
                'Donald Trump wins the election.',
                'Scientists confirm the earth is flat.',
                'Chocolate cures all diseases.',
                'Elvis Presley spotted in Paris.',
                'Fake news spreads faster than real news.',
                'The moon is made of cheese.',
                'Dinosaurs still live in the Amazon.',
                'Bill Gates plans to block the sun.'
            ],
            'label': ['fake']*10
        })
        true_df = pd.DataFrame({
            'text': [
                'Modi is the prime minister of India.',
                'NASA discovers water on Mars.',
                'The stock market reached a new high.',
                'COVID-19 vaccine is safe and effective.',
                'Scientists discover new species in the Amazon.',
                'Chocolate may have health benefits.',
                'Elvis Presley was a famous singer.',
                'Real news is verified by journalists.',
                'The moon is made of rock.',
                'Dinosaurs went extinct millions of years ago.',
                'Bill Gates invests in renewable energy.'
            ],
            'label': ['real']*11
        })
    else:
        fake_df = pd.read_csv(fake_file)
        true_df = pd.read_csv(true_file)
        fake_df['label'] = 'fake'
        true_df['label'] = 'real'
    data = pd.concat([fake_df[['text', 'label']], true_df[['text', 'label']]], ignore_index=True)
    data = data.dropna()

    def clean_text(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    data['text'] = data['text'].apply(clean_text)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(data['text'])
    y = data['label']

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(zip(np.unique(y), class_weights))

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict, max_depth=10, n_jobs=-1)
    model.fit(X, y)

    user_input = st.text_area("News text yahan likhein:")
    if st.button("Predict"):
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)
        st.write(f"Prediction: **{pred[0]}**")
else:
    st.warning("Fake.csv aur True.csv upload karein ya demo mode select karein.")
