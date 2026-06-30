# 🏦 Enterprise Customer Churn Prediction using Artificial Neural Networks

## 📌 Project Overview

This project predicts whether a bank customer is likely to leave the bank (customer churn) using an Artificial Neural Network (ANN) built with TensorFlow and Keras.

The trained model is deployed through a Streamlit web application where users can enter customer information and receive an instant churn probability prediction.

---

## 🚀 Live Features

- Deep Learning based prediction
- Interactive Streamlit Web Application
- Clean Enterprise Dashboard UI
- Real-time customer churn probability
- Automatic preprocessing pipeline
- One-Hot Encoding
- Label Encoding
- Feature Scaling
- TensorFlow ANN inference

---

## 🧠 Problem Statement

Banks lose a significant amount of revenue whenever customers close their accounts and move to competitors.

Instead of waiting for customers to leave, banks can use Machine Learning to identify high-risk customers beforehand.

This project predicts the probability that a customer will churn based on demographic, financial and engagement information.

---

## 📊 Dataset

The project uses the **Churn_Modelling.csv** dataset.

Features include:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

Target Variable

- Exited
    - 1 → Customer left the bank
    - 0 → Customer stayed

---

## 🏗 Project Workflow

Dataset

↓

Data Preprocessing

↓

Label Encoding

↓

One-Hot Encoding

↓

Feature Scaling

↓

Train-Test Split

↓

Artificial Neural Network

↓

Model Training

↓

Model Saving (.h5)

↓

Streamlit Deployment

↓

Real-Time Prediction

---

## 🧠 ANN Architecture

Input Layer

↓

Dense Layer (12 neurons, ReLU)

↓

Dense Layer (6 neurons, ReLU)

↓

Output Layer (1 neuron, Sigmoid)

---

## ⚙ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-Learn
- Streamlit
- Pickle

---

## 📁 Project Structure

```
ChurnPredANN/

│── app.py
│── model.h5
│── scaler.pkl
│── label_encoder_gender.pkl
│── onehot_encoder_geo.pkl
│── Churn_Modelling.csv
│── experiments.ipynb
│── prediction.ipynb
│── requirements.txt
│── README.md
│── .gitignore
```

---

## 📈 Model Pipeline

1. Load trained ANN model
2. Load encoders and scaler
3. Accept user inputs
4. Encode categorical variables
5. Scale numerical features
6. Predict churn probability
7. Display prediction on dashboard

---

## 💻 User Interface

The Streamlit dashboard includes

- Demographic Information
- Financial Profile
- Engagement Metrics
- Risk Probability Score
- Customer Status
- Financial Visualization

---



