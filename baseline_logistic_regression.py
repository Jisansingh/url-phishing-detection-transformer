"""
Logistic Regression Baseline for Phishing URL Detection.

This script implements a non-deep-learning baseline using:
- Character-level n-gram features (n=3-5)
- TF-IDF vectorization
- Logistic Regression with balanced class weights

This serves as a benchmark to compare against the Transformer model.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import time
import os

# --- Configuration ---
TRAIN_PATH = "train.csv"
VAL_PATH = "val.csv"
TEST_PATH = "test.csv"

# N-gram range for character-level features
NGRAM_RANGE = (3, 5)

# TF-IDF settings
MAX_FEATURES = 10000  # Limit vocabulary size for efficiency


def load_data(filepath):
    """Load dataset from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")
    df = pd.read_csv(filepath)
    return df['url'].values, df['label'].values


def main():
    print("\n" + "#"*60)
    print("#  LOGISTIC REGRESSION BASELINE - PHISHING DETECTION")
    print("#"*60 + "\n")

    # --- Load Data ---
    print("Loading datasets...")
    X_train, y_train = load_data(TRAIN_PATH)
    X_val, y_val = load_data(VAL_PATH)
    X_test, y_test = load_data(TEST_PATH)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # --- Feature Extraction with TF-IDF ---
    print(f"\nApplying TF-IDF with character n-grams (n={NGRAM_RANGE[0]}-{NGRAM_RANGE[1]})...")
    
    vectorizer = TfidfVectorizer(
        analyzer='char',           # Character-level analysis
        ngram_range=NGRAM_RANGE,   # n-grams from 3 to 5
        max_features=MAX_FEATURES, # Limit features
        lowercase=False,           # URLs are case-sensitive
        sublinear_tf=True          # Apply sublinear TF scaling
    )

    start_time = time.time()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    vectorize_time = time.time() - start_time

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    print(f"Vectorization time: {vectorize_time:.2f}s")

    # --- Train Logistic Regression ---
    print("\nTraining Logistic Regression with balanced class weights...")
    
    model = LogisticRegression(
        class_weight='balanced',  # Handle class imbalance
        max_iter=1000,            # Ensure convergence
        solver='lbfgs',           # Efficient solver
        random_state=42,          # Reproducibility
        n_jobs=-1                 # Use all cores
    )

    start_time = time.time()
    model.fit(X_train_tfidf, y_train)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f}s")

    # --- Evaluate on Validation Set ---
    print("\n" + "="*60)
    print("VALIDATION SET PERFORMANCE")
    print("="*60)
    
    y_val_pred = model.predict(X_val_tfidf)
    y_val_proba = model.predict_proba(X_val_tfidf)[:, 1]
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_proba)
    
    print(f"Accuracy:  {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall:    {val_recall:.4f}")
    print(f"F1-Score:  {val_f1:.4f}")
    print(f"ROC-AUC:   {val_roc_auc:.4f}")

    # --- Evaluate on Test Set (Final Benchmark) ---
    print("\n" + "="*60)
    print("TEST SET PERFORMANCE (FINAL BENCHMARK)")
    print("="*60)
    
    y_test_pred = model.predict(X_test_tfidf)
    y_test_proba = model.predict_proba(X_test_tfidf)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\nAccuracy:  {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"ROC-AUC:   {test_roc_auc:.4f}")

    # --- Classification Report ---
    print("\n" + "-"*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("-"*60)
    target_names = ['Legitimate (0)', 'Phishing (1)']
    print(classification_report(y_test, y_test_pred, target_names=target_names))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n" + "-"*60)
    print("CONFUSION MATRIX")
    print("-"*60)
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")

    # --- Save Results for Comparison ---
    results = {
        'Model': 'Logistic Regression (char n-grams 3-5)',
        'Accuracy': test_accuracy,
        'Precision': test_precision,
        'Recall': test_recall,
        'F1-Score': test_f1,
        'ROC-AUC': test_roc_auc,
        'Training_Time_Seconds': train_time
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('baseline_results.csv', index=False)
    print("\nResults saved to 'baseline_results.csv'")

    # --- Print Summary for Comparison ---
    print("\n" + "#"*60)
    print("#  BASELINE EVALUATION COMPLETE")
    print("#"*60)
    print("\nUse these metrics to compare against the Transformer model.")

    return results


if __name__ == "__main__":
    main()
