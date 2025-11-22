# 📌 Amazon Product Review Sentiment Analysis (NLP + ML + Streamlit)
# 🔍 Project Overview

This project performs Sentiment Analysis on Amazon product reviews using Natural Language Processing (NLP) techniques and Machine Learning models.
The goal is to automatically classify customer reviews as Positive or Negative, helping businesses understand customer opinions and improve product decisions.

# The project includes:
Text preprocessing (tokenization, stopword removal, lemmatization)
TF-IDF vectorization
Model training & evaluation
Comparison of Logistic Regression vs Multinomial Naive Bayes
Real-time prediction Web App built with Streamlit

# 🎯 Problem Statement

E-commerce platforms receive thousands of textual reviews daily.
Manually analyzing customer sentiment is time-consuming and inefficient.

# This project solves the problem by:

Automatically predicting sentiment from review text
Saving analysis time and improving decision-making
Helping companies monitor product performance and customer satisfaction

# 📂 Dataset

Amazon Product Reviews Dataset
Size: ~20,000 reviews
Columns:
Text — customer review
Label — 1 = Positive, 0 = Negative

# 🧠 Approach
Step	                          Description
Data                Loading & EDA	Inspection of text & label balance
Text Preprocessing	Lowercasing, punctuation removal, stopwords, lemmatization
Vectorization	      TF-IDF with 10,000 features (1–2 n-grams)
Model Training	    Logistic Regression, Multinomial Naive Bayes
Evaluation	        Accuracy, F1-score, confusion matrix
Deployment	        Streamlit-based UI for live predictions

# 📊 Model Results
Model	Accuracy	F1-Score
Logistic Regression	0.89	0.88
Multinomial Naive Bayes	0.86	0.86
Confusion Matrix (Logistic Regression)
	Pred 0	Pred 1
Actual 0	587	366
Actual 1	91	2956

# 🚀 Deployment

This project includes a fully-working Streamlit app where users can input any review text and get instant sentiment prediction.

Run Locally
pip install -r requirements.txt
streamlit run app.py

🖥 Application Interface

(Add screenshot here)

# 📦 Repository Structure
├── Sentiment_Analysis.ipynb       # Training notebook
├── app.py                         # Streamlit application
├── sentiment_model.pkl            # Saved ML model
├── tfidf_vectorizer.pkl           # Saved TF-IDF transformer
├── requirements.txt               # Dependencies
└── README.md                      # Documentation

# 💡 Business Use-Cases

Customer feedback analytics
Real-time review monitoring
Product improvement decisions
Automatic moderation / filtering
Brand sentiment tracking

# ✨ Future Improvements

Add neutral sentiment → 3-class model
Deploy to Render / HuggingFace / AWS
Use BERT / Transformer models
Add dashboard for analytics

# 👤 Author

Rohit Vastani
AI & ML Student | Data Science & NLP Enthusiast
📍 India

🔗 LinkedIn: https://www.linkedin.com/in/rohit-vastani-3a9a18301/?utm_source=share
🔗 GitHub: https://github.com/Rohit-8626

⭐ If you found this project useful, please star this repo!
