import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "../models/vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = model.coef_[0]

# Top positive weights → Fake
top_fake_indices = np.argsort(coefficients)[-20:]
top_fake_words = feature_names[top_fake_indices]

# Top negative weights → Real
top_real_indices = np.argsort(coefficients)[:20]
top_real_words = feature_names[top_real_indices]

print("Top words indicating FAKE news:\n")
print(top_fake_words)

print("\nTop words indicating REAL news:\n")
print(top_real_words)