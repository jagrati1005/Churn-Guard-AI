import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    df.drop(columns=['customerID'], inplace=True, errors='ignore')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def train_and_evaluate(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)

    results = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest'],
        'Accuracy': [accuracy_score(y_test, lr_preds), accuracy_score(y_test, rf_preds)],
        'Precision': [precision_score(y_test, lr_preds), precision_score(y_test, rf_preds)],
        'Recall': [recall_score(y_test, lr_preds), recall_score(y_test, rf_preds)],
        'F1 Score': [f1_score(y_test, lr_preds), f1_score(y_test, rf_preds)]
    })
    
    return rf_pipeline, results, X_test, y_test

def predict_customer_risk(model, customer_data):
    customer_df = pd.DataFrame([customer_data])
    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0][1]
    
    if probability > 0.70:
        risk = "HIGH"
    elif probability > 0.40:
        risk = "MEDIUM"
    else:
        risk = "LOW"
        
    result = "Customer likely to churn" if prediction == 1 else "Customer likely to stay"
    
    return {
        "prediction": result,
        "churn_probability": f"{probability * 100:.2f}%",
        "risk_level": risk
    }

if __name__ == "__main__":
    dataset_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp-for-data/master/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    print("Loading data...")
    df = load_and_preprocess_data(dataset_url)
    
    print("Training models...")
    rf_model, performance_df, X_test, y_test = train_and_evaluate(df)
    
    print("\n================ MODEL PERFORMANCE ================")
    print(performance_df.to_string(index=False))
    
    sample_customer = X_test.iloc[0].to_dict()
    print("\n================ SAMPLE PREDICTION ================")
    prediction_result = predict_customer_risk(rf_model, sample_customer)
    for key, value in prediction_result.items():
        print(f"{key.capitalize().replace('_', ' ')}: {value}")