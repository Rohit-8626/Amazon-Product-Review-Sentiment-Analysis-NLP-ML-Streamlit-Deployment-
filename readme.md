📌 Amazon Product Review Sentiment Analysis (NLP + ML + Streamlit)
🔍 Project Overview

This project performs Sentiment Analysis on Amazon product reviews using Natural Language Processing (NLP) and Machine Learning.
The system classifies customer feedback into Positive or Negative, enabling businesses to quickly understand product satisfaction without manual review.

The solution includes:

Full NLP preprocessing pipeline

TF-IDF feature extraction

Model comparison & evaluation

Real-time web app using Streamlit

🎯 Problem Statement

E-commerce platforms receive thousands of text reviews daily.
Manually analyzing sentiment is time-consuming, expensive, and prone to human error.

This project solves the issue by:

Automatically detecting customer opinion from review text

Providing fast & scalable sentiment classification

Supporting product improvement and customer experience insights

📂 Dataset

Amazon Product Reviews Dataset

Size: ~20,000 reviews

Columns:

Column	Description
Text	Review content
Label	1 = Positive, 0 = Negative

🧠 Approach
Step	Description
Data Loading	Read dataset & inspect distribution
Text Cleaning	Lowercasing, punctuation removal, stopwords, lemmatization
Feature Extraction	TF-IDF Vectorization (10k features, 1–2 n-grams)
Model Training	Logistic Regression, Multinomial Naive Bayes
Evaluation	Accuracy, Precision, Recall, F1, Confusion Matrix
Deployment	Streamlit app for real-time sentiment predictions

📊 Model Performance
Model	Accuracy	F1-Score
Logistic Regression	0.89	0.88
Multinomial Naive Bayes	0.86	0.86
Confusion Matrix (Logistic Regression)
	Pred 0	Pred 1
Actual 0	587	366
Actual 1	91	2956

🚀 Deployment

This project includes a working Streamlit application that predicts sentiment from any user-typed review.

Run Locally
pip install -r requirements.txt
streamlit run app.py

🖥 Application Preview

(Add streamlit screenshots here)

📁 Repository Structure
├── Sentiment_Analysis.ipynb       # Training + evaluation notebook
├── app.py                         # Streamlit web app
├── sentiment_model.pkl            # Saved ML model
├── tfidf_vectorizer.pkl           # Saved TF-IDF vectorizer
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation

💡 Business Use Cases

Automated review monitoring

Product rating improvement insights

Customer feedback mining

Real-time moderation systems

Brand reputation tracking

✨ Future Enhancements

Add Neutral class (3-class sentiment)

Deploy on Render / HuggingFace / AWS

Integrate BERT / Transformer models

Build dashboard insights & analytics

👤 Author

Rohit Vastani
AI & ML Student | Data Science & NLP Enthusiast
📍 India

🔗 LinkedIn: https://www.linkedin.com/in/rohit-vastani-3a9a18301/?utm_source=share

🔗 GitHub: https://github.com/Rohit-8626

⭐ If you found this useful, consider giving the repository a star!
