import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load data and train model (same as your script)
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')
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

# Streamlit UI
st.title("Fake News Detection")
user_input = st.text_area("Enter news text to check:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)
    st.write(f"Prediction: **{pred[0]}**")