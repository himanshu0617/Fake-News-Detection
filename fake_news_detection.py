import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load and combine real-world dataset
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

model = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight=class_weight_dict, max_depth=None)
model.fit(X, y)

y_pred_all = model.predict(X)
print("Accuracy on all data:", accuracy_score(y, y_pred_all))
print(classification_report(y, y_pred_all))

cm = confusion_matrix(y, y_pred_all, labels=['real', 'fake'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['real', 'fake'])
disp.plot()
plt.title('Confusion Matrix (All Data)')
plt.show()

sample = ["NASA discovers water on Mars, confirming possibility of life."]
sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)
print("Prediction for sample:", prediction[0])
