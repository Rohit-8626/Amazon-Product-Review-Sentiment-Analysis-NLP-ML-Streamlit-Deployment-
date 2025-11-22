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



---

## 📁 Repository Structure
