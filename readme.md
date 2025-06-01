# 🚢 Titanic Survival Prediction App

This is an interactive Streamlit web app that predicts whether a passenger would survive the Titanic disaster based on their characteristics using a **Logistic Regression** model.

---

## 🎯 Features

- Input passenger details through a simple sidebar interface.
- Predict survival outcome using a trained logistic regression model.
- Displays survival probability as a percentage.
- Automatically trains and caches the model if not already saved.

---

## 🧠 Model

The model is trained using the classic **Titanic dataset** (`titanic.csv`) with the following features:

- Pclass (Passenger class)
- Sex
- Age
- SibSp (Siblings/Spouses aboard)
- Parch (Parents/Children aboard)
- Fare
- Embarked (Q and S encoded)

Model: `LogisticRegression` from `scikit-learn`  
Scaler: `StandardScaler` for feature normalization

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/titanic-survival-app.git
cd titanic-survival-app
