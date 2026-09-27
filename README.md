# CHURNGUARD AI: Customer Churn Prediction & Retention Intelligence System

CHURNGUARD AI is an end-to-end Machine Learning classification project designed to analyze customer data (such as tenure, contract type, monthly charges, and subscribed services) and predict customer churn risk. The system processes historical customer information, trains predictive models, and outputs a customer's churn probability alongside actionable risk levels (**HIGH**, **MEDIUM**, or **LOW**) to help businesses take proactive retention measures.

---

## 📌 Features

- **Data Preprocessing & Cleaning:** Automatic handling of missing values, duplicate records, and data type formatting.
- **Exploratory Data Analysis (EDA):** Visual insights into churn drivers such as contract duration, tenure, and monthly charges.
- **Pipeline Processing:** Scikit-Learn `ColumnTransformer` and `OneHotEncoder` for standard feature encoding.
- **Multi-Model Comparison:** Evaluates both **Logistic Regression** and **Random Forest Classifier**.
- **Risk Assessment System:** Classifies predictions into HIGH (>70%), MEDIUM (40–70%), and LOW (<40%) risk tiers.

---

## 📁 Repository Structure

```text
CHURNGUARD-AI/
├── churn_prediction.py    # Complete Python ML pipeline
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
