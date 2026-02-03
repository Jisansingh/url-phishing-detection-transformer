# Transformer-Based Phishing URL Detection

A deep learning–based phishing website detection system using a Transformer architecture and URL sequence analysis.  
The model learns structural and contextual patterns in URLs to classify websites as **phishing** or **legitimate**, enabling effective detection of zero-day attacks beyond traditional blacklist-based methods.

---

## 📌 Problem Statement

Phishing websites are a major cybersecurity threat that deceive users into revealing sensitive information such as login credentials, banking details, and personal data.  
Attackers continuously generate new phishing URLs, making rule-based and blacklist-based detection systems ineffective.

This project aims to design and implement a **Transformer-based deep learning model** that classifies a website URL as phishing or legitimate using learned URL patterns.

---

## 🎯 Objectives

- Build an end-to-end phishing detection pipeline using deep learning  
- Treat URLs as sequential data and apply self-attention mechanisms  
- Detect zero-day phishing attacks without relying on blacklists  
- Evaluate the model using appropriate performance metrics  

---

## 🧠 Model Architecture

- **Embedding Layer** – Converts URL characters into dense vector representations  
- **Transformer Encoder**  
  - Multi-Head Self-Attention  
  - Feed-Forward Networks  
  - Layer Normalization & Residual Connections  
- **Classification Head** – Fully connected layers with sigmoid activation  

The Transformer architecture enables the model to capture long-range dependencies and structural patterns in URLs.

---

## 📊 Dataset

A custom-built dataset created from real-world sources.

### Sources
- **Phishing URLs**: OpenPhish, Phishing.Database  
- **Legitimate URLs**: Tranco top-ranked domains  

### Dataset Statistics
- Total URLs: **10,000**
- Phishing URLs: **5,000**
- Legitimate URLs: **5,000**
- Balanced dataset (50:50)

### Preprocessing
- Duplicate URLs removed  
- Character-level tokenization  
- Padding and truncation applied  

---

## ⚙️ Implementation Pipeline

1. Data collection and preprocessing  
2. URL tokenization and sequence encoding  
3. Transformer model training  
4. Model evaluation and error analysis  
5. Performance visualization  

---

## 📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- Confusion Matrix  

These metrics ensure robust evaluation, especially for imbalanced and security-critical classification tasks.

---

## 🚧 Challenges Faced

- Handling variable-length URLs  
- Capturing meaningful patterns from short text sequences  
- Avoiding overfitting on structured URL data  

### Solutions
- Character-level encoding  
- Transformer self-attention mechanism  
- Regularization and validation-based evaluation  

---

## 🏁 Results & Conclusion

The Transformer-based model demonstrates strong performance in distinguishing phishing URLs from legitimate ones by learning contextual and structural URL patterns.  
This approach outperforms traditional rule-based systems and provides a scalable foundation for real-world phishing detection systems.

---

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  
