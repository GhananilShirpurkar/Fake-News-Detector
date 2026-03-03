import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from utils import clean_text

# ==========================
# Paths
# ==========================

DATA_FAKE = "../data/Fake.csv"
DATA_TRUE = "../data/True.csv"
MODEL_DIR = "../models"

# ==========================
# Load Dataset
# ==========================

def load_data():
    fake = pd.read_csv(DATA_FAKE)
    true = pd.read_csv(DATA_TRUE)

    fake["label"] = 1
    true["label"] = 0

    df = pd.concat([fake, true], axis=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df["content"] = df["title"] + " " + df["text"]
    df["content"] = df["content"].apply(clean_text)

    return df["content"], df["label"]


# ==========================
# Train Model
# ==========================

def train():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        max_features=10000
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))

    print("Model saved successfully.")


if __name__ == "__main__":
    train()