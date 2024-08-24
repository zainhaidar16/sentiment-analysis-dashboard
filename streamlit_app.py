import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Sidebar options
option = st.sidebar.selectbox("Select Input Method", ["Manual Text Input", "CSV File Upload"])

# Data Input and Preprocessing
df = None
if option == "Manual Text Input":
    st.subheader("Manual Text Input")
    text = st.text_area("Enter text")
    # Split text into train and test data

elif option == "CSV File Upload":
    st.subheader("CSV File Upload")
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding='latin1')
        # Display overview of the CSV
        st.write(df.head())
        # Allow user to select column containing text data

# Model Selection
st.subheader("Model Selection")
models = ["Naive Bayes", "Logistic Regression", "Support Vector Machine (SVM)", "Random Forest", "XGBoost"]
selected_models = st.multiselect("Select model(s) to train", models)

# Model Training and Evaluation
if st.button("Train Models", key="train_button") and df is not None:
    st.subheader("Model Training and Evaluation")
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    # Train selected models
    trained_models = []
    for model_name in selected_models:
        if model_name == "Naive Bayes":
            model = MultinomialNB()
        elif model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "Support Vector Machine (SVM)":
            model = SVC()
        elif model_name == "Random Forest":
            model = RandomForestClassifier()
        elif model_name == "XGBoost":
            model = XGBClassifier()
        else:
            continue
        
        model.fit(X_train, y_train)
        trained_models.append(model)
    
    # Evaluate model performance
    performance_metrics = []
    for model in trained_models:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        
        performance_metrics.append({
            "Model": model.__class__.__name__,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Confusion Matrix": confusion
        })
    
    # Extract the performance metrics
    model_names = [metrics['Model'] for metrics in performance_metrics]
    accuracies = [metrics['Accuracy'] for metrics in performance_metrics]
    precisions = [metrics['Precision'] for metrics in performance_metrics]
    recalls = [metrics['Recall'] for metrics in performance_metrics]
    f1_scores = [metrics['F1 Score'] for metrics in performance_metrics]

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, label='Accuracy')
    plt.bar(model_names, precisions, label='Precision')
    plt.bar(model_names, recalls, label='Recall')
    plt.bar(model_names, f1_scores, label='F1 Score')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
else:
    st.warning("Please upload a CSV file to proceed.")