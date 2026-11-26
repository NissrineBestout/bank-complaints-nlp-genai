# FinSight Analytics — Banking Complaint Classification & Resolution (NLP + GenAI)

A hybrid **Natural Language Processing (NLP)** and **Generative AI** pipeline designed to automate banking customer complaint routing and prescriptive decision support, built on the **CFPB Consumer Complaint Database**.

**Live Demo:** [Open Streamlit App](https://bank-complaints-nlp.streamlit.app/)

---

## Architecture

The system separates fast, low-cost predictive classification from high-value prescriptive reasoning:

1. **Data Preprocessing & Feature Engineering:** Regex cleaning, removal of privacy masks (`XXXX`), and TF-IDF vectorization (unigrams & bigrams, 15,000 features) on a stratified sample of 19,969 valid complaints across the top 5 banking products.
2. **Predictive Layer (Classical NLP):** A class-balanced **Linear SVM** routes incoming complaints to the appropriate banking department in milliseconds.
3. **Prescriptive Layer (Generative AI):** **Google Gemini** (with automatic Flash model fallback) generates a structured JSON analysis containing:
   - **Manager Summary** (French)
   - **Root Cause Diagnosis** (French)
   - **Urgency Level & Justification** (`Faible` / `Moyen` / `Élevé`)
   - **Ready-to-Send Empathetic Client Reply** (English)

---

## Model Performance (Test Set: 3,994 Complaints)

| Model | Accuracy | F1-Weighted | F1-Macro |
| :--- | :---: | :---: | :---: |
| **Linear SVM (`class_weight="balanced"`)** | **89.73%** | **0.8971** | **0.8515** |
| Logistic Regression (Balanced) | 88.91% | 0.8910 | 0.8466 |
| Multinomial Naive Bayes | 86.10% | 0.8398 | 0.7050 |

---

## Repository Structure

```text
├── app.py                   # Streamlit web application (Fintech UI)
├── requirements.txt         # Python dependencies
├── best_nlp_model.pkl       # Trained Linear SVM model
├── tfidf_vectorizer.pkl     # Fitted TF-IDF vectorizer
├── eda_charts.png           # Exploratory Data Analysis visualizations
└── confusion_matrix.png     # Test set confusion matrix
