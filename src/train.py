import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import calibration_curve

from utils import clean_text
import matplotlib.pyplot as plt


# ==========================
# Robust Path Handling
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_FAKE = os.path.join(PROJECT_ROOT, "data", "Fake.csv")
DATA_TRUE = os.path.join(PROJECT_ROOT, "data", "True.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


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

    df["content"] = (df["title"] + " " + df["text"]).apply(clean_text)

    return df["content"], df["label"]


# ==========================
# Train & Compare Models
# ==========================

def train():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        max_features=20000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # ===============================
    # Logistic Regression
    # ===============================

    lr_model = LogisticRegression(max_iter=1000)

    lr_cv_scores = cross_val_score(
        lr_model,
        X_train_vec,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    lr_model.fit(X_train_vec, y_train)
    lr_test_preds = lr_model.predict(X_test_vec)

    lr_cv_mean = lr_cv_scores.mean()
    lr_test_acc = accuracy_score(y_test, lr_test_preds)

    print("\n=== Logistic Regression ===")
    print("CV Scores:", lr_cv_scores)
    print("Mean CV Accuracy:", lr_cv_mean)
    print("Test Accuracy:", lr_test_acc)
    print(classification_report(y_test, lr_test_preds))

    # ===============================
    # Naive Bayes
    # ===============================

    nb_model = MultinomialNB()

    nb_cv_scores = cross_val_score(
        nb_model,
        X_train_vec,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    nb_model.fit(X_train_vec, y_train)
    nb_test_preds = nb_model.predict(X_test_vec)

    nb_cv_mean = nb_cv_scores.mean()
    nb_test_acc = accuracy_score(y_test, nb_test_preds)

    print("\n=== Naive Bayes ===")
    print("CV Scores:", nb_cv_scores)
    print("Mean CV Accuracy:", nb_cv_mean)
    print("Test Accuracy:", nb_test_acc)
    print(classification_report(y_test, nb_test_preds))

    # ===============================
    # Probability Analysis (LR only)
    # ===============================

    probs = lr_model.predict_proba(X_test_vec)[:, 1]

    # --- Calibration Curve ---
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)

    plt.figure()
    plt.plot(prob_pred, prob_true)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Calibration Curve (Logistic Regression)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.show()

    # --- Confidence Distribution ---
    plt.figure()
    plt.hist(probs, bins=20)
    plt.title("Prediction Probability Distribution")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.show()

    # ===============================
    # Error Analysis
    # ===============================

    misclassified_mask = y_test != lr_test_preds
    misclassified_texts = X_test[misclassified_mask]

    print("\nNumber of Misclassified Samples:", len(misclassified_texts))
    print("\nSample Misclassifications:\n")
    print(misclassified_texts.head())

    # ===============================
    # Select Best Model Automatically
    # ===============================

    if lr_cv_mean >= nb_cv_mean:
        best_model = lr_model
        best_name = "Logistic Regression"
    else:
        best_model = nb_model
        best_name = "Naive Bayes"

    print(f"\nBest Model Selected: {best_name}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))

    print("Best model saved successfully.")