# 📌 Amazon Product Review Sentiment Analysis (NLP + ML + Streamlit)

## 💡 Project Overview

This project implements **Sentiment Analysis** on Amazon product reviews using a classic **Natural Language Processing (NLP)** and **Machine Learning (ML)** pipeline. The system is designed to automatically classify customer feedback into **Positive (1)** or **Negative (0)** sentiment, providing businesses with a fast, scalable, and automated way to understand product satisfaction without manual review.

---

## 🎯 Problem Statement

E-commerce platforms generate an immense volume of customer feedback daily. Manually reading and classifying the sentiment of thousands of text reviews is time-consuming, expensive, and often inconsistent due to human error.

This project addresses this challenge by:

* **Automatically detecting** the underlying customer opinion from raw review text.
* Providing **fast and scalable** sentiment classification.
* Generating **actionable insights** for product improvement and enhancing customer experience.

---

## 🧠 Technical Approach

The solution follows a standard supervised machine learning workflow, focusing on effective text preprocessing and feature engineering.

### Approach Steps

| Step | Description |
| :--- | :--- |
| **Data Loading** | Reading the dataset and inspecting class distribution. |
| **Text Cleaning** | Applying **Lowercasing**, removal of **punctuation**, eliminating **stopwords**, and **lemmatization** to normalize the text data. |
| **Feature Extraction** | Using **TF-IDF Vectorization** (with 10k features and 1–2 n-grams) to convert text into numerical feature vectors. |
| **Model Training** | Training and evaluating two classification models: **Logistic Regression** and **Multinomial Naive Bayes**. |
| **Evaluation** | Measuring performance using **Accuracy**, **Precision**, **Recall**, **F1-Score**, and analyzing the **Confusion Matrix**. |
| **Deployment** | Creating a real-time, interactive web application using **Streamlit**. |

### 📂 Dataset

* **Source:** Amazon Product Reviews Dataset
* **Size:** Approximately 20,000 reviews
* **Columns:**
    | Column | Description |
    | :--- | :--- |
    | `Text` | Review content |
    | `Label` | **1** = Positive, **0** = Negative |

---

## 📊 Model Performance

Two models were trained and evaluated on the dataset. **Logistic Regression** demonstrated superior performance.

### Model Metrics

| Model | Accuracy | F1-Score |
| :--- | :--- | :--- |
| **Logistic Regression** | **0.89** | **0.88** |
| Multinomial Naive Bayes | 0.86 | 0.86 |

### Confusion Matrix (Logistic Regression)

| | **Predicted 0 (Negative)** | **Predicted 1 (Positive)** |
| :--- | :--- | :--- |
| **Actual 0 (Negative)** | 587 (True Negatives) | 366 (False Positives) |
| **Actual 1 (Positive)** | 91 (False Negatives) | 2956 (True Positives) |

---

## 🚀 Deployment (Streamlit Web App)

The project is deployed as an interactive web application using **Streamlit**, allowing users to input a review and receive a real-time sentiment prediction.

### Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [Your GitHub Repo URL]
    cd [Your Repo Directory]
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

### 🖥 Application Preview

* **Streamlit Link: ** [images/Application_Preview.png](images/Application_Preview.png)

---

## 📁 Repository Structure

The project directory is organized to separate the core ML development from the deployment artifacts.

| File/Directory | Description |
| :--- | :--- |
| `Sentiment_Analysis.ipynb` | **Jupyter Notebook** containing the complete data loading, preprocessing, feature extraction, model training, evaluation, and selection steps. |
| `app.py` | The main **Streamlit script** used to create the interactive web application for real-time predictions. |
| `sentiment_model.pkl` | The **serialized machine learning model** (Logistic Regression) saved after training. |
| `tfidf_vectorizer.pkl` | The **saved TF-IDF vectorizer object**, essential for transforming new input text into the numerical format expected by the model. |
| `requirements.txt` | Lists all necessary **Python libraries and dependencies** required to run the project. |
| `README.md` | This **project documentation file**. |

---

## 💡 Business Use Cases

This Sentiment Analysis solution provides significant value across various business functions:

* **Automated Review Monitoring:** **Instantly flag** highly negative reviews that require immediate customer service intervention.
* **Product Improvement Insights:** Identify recurring themes or features associated with negative feedback (Customer feedback mining) to **guide engineering and product development**.
* **Real-time Moderation Systems:** Implement a filter to **automatically moderate** abusive, spam, or highly inappropriate content before it goes live.
* **Brand Reputation Tracking:** Monitor the overall **sentiment trend** of the brand's products over specified time periods.
* **Product Rating Improvement:** Understand the core drivers of low ratings to prioritize changes that will have the biggest impact.

---

## ✨ Future Enhancements

The following features are planned to expand the scope and robustness of the project:

1.  **Add Neutral Class (3-Class Sentiment):** Expand the current binary classification (Positive/Negative) to include a **Neutral** category for more granular analysis.
2.  **Advanced Deployment:** Move the application from local hosting to public cloud platforms like **Render**, **HuggingFace Spaces**, or **AWS/GCP** for wider accessibility.
3.  **Integrate BERT / Transformer Models:** Replace the traditional ML pipeline with **Deep Learning** models (e.g., BERT, RoBERTa) to leverage contextual embeddings for higher accuracy.
4.  **Build Dashboard Insights & Analytics:** Develop an interactive front-end dashboard to visualize sentiment distribution, keyword analysis, and model confidence scores.

---

## 👤 Author

**Rohit Vastani**

* **Role:** AI & ML Student | Data Science & NLP Enthusiast
* **Location:** 📍 India
* **LinkedIn:** [https://www.linkedin.com/in/rohit-vastani-3a9a18301/?utm_source=share](https://www.linkedin.com/in/rohit-vastani-3a9a18301/?utm_source=share)
* **GitHub:** [https://github.com/Rohit-8626](https://github.com/Rohit-8626)

⭐ **If you found this useful, consider giving the repository a star!**
