# 📉 Customer Churn Prediction Using Machine Learning

## 🧾 Overview

This repository contains the code and resources for a machine learning project focused on predicting customer churn in a subscription-based telecommunication business.  
Customer churn — the loss of customers — is a critical metric for companies, and predicting it enables proactive strategies to retain valuable clients.

This project was completed as part of the Machine Learning module at the Higher School of Technology of Meknès.

---

## 🎯 Objective

The main objective is to develop an accurate machine learning model to predict the likelihood of customer churn based on historical behavior and subscription data.

We leverage multiple models and select the one that performs best, ensuring a robust predictive system suitable for deployment.

---

---

## 📂 Dataset

The dataset used includes features such as:

- Customer demographics
- Subscription type and duration
- Historical interactions
- Churn labels (Yes/No)

It was cleaned and preprocessed to ensure quality training input.

---

## 📊 Model Selection and Evaluation

Before finalizing the model, we tested and compared the following classifiers:

- **Random Forest**
- **Logistic Regression**
- **Gradient Boosting**
- **XGBoost**

All models were evaluated.  
✅ **XGBoost** achieved the best performance and was selected as the final model used in deployment.

Evaluation metrics used include:

- Accuracy
- Precision
- Recall
- F1-Score

---

## 🛠️ Features

- Cleaned and preprocessed real-world telecom churn dataset
- Model training and performance analysis using multiple classifiers
- Interactive Streamlit app for live testing and visualization
- Fully documented workflow in a Jupyter Notebook

---

## 📁 Project Structure

```
.
├── code-source.ipynb           # Jupyter Notebook with full ML workflow
├── app.py                      # Streamlit application for deployment
├── requirements.txt            # Python dependencies
├── telecom_customer_churn.csv  # Dataset file (CSV)
├── rapport_ML.pdf              # Project Report (French Version)
├── images                      # Images Folders
└── README.md                   # Project documentation
```

---

## ⚙️ Installation

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/sana-18/Customer-Churn-Prediction-Using-ML.git
cd Customer-Churn-Prediction-Using-ML
```

### 2. Set Up Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```
---

## 🚀 How to Use

### ▶️ Run the Jupyter Notebook

To analyze data and review the model training process:

```bash
jupyter notebook code-source.ipynb
```

### ▶️ Launch the Streamlit App

To open the interactive app that summarizes the pipeline:

```bash
streamlit run app.py
```

This app provides:

- Data preview
- Project Workflow 
- Live churn predictions






