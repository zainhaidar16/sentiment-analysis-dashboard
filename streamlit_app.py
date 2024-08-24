import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from collections import Counter

# Sidebar options
option = st.sidebar.selectbox("Select Input Method", ["Manual Text Input", "CSV File Upload"])

# Pre-trained model from Hugging Face
sentiment_pipeline = pipeline("sentiment-analysis")

# Data Input and Preprocessing
df = None
try:
    if option == "Manual Text Input":
        st.subheader("Manual Text Input")
        text = st.text_area("Enter text for sentiment analysis")
        
        if text:
            # Use the pre-trained model to predict sentiment
            result = sentiment_pipeline(text)
            label = result[0]['label']
            score = result[0]['score']
            
            st.write(f"Sentiment: **{label}**")
            st.write(f"Confidence Score: **{score:.2f}**")

    elif option == "CSV File Upload":
        st.subheader("CSV File Upload")
        file = st.file_uploader("Upload CSV file", type=["csv"])
        if file is not None:
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file, encoding='latin1')
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
                    st.stop()
            except Exception as e:
                st.error(f"Error processing the file: {e}")
                st.stop()

            if df is not None:
                st.write(df.head())
                # Allow user to select column containing text data
                text_column = st.selectbox("Select the text column for analysis", df.columns)
                label_column = st.selectbox("Select the label column for analysis (optional)", [None] + list(df.columns))
                
                if text_column:
                    df['text'] = df[text_column]
                    if label_column:
                        df['label'] = df[label_column]
                    else:
                        df['label'] = [1 if "good" in text.lower() else 0 for text in df['text']]
                else:
                    st.error("Please select a valid text column.")
                    st.stop()

                # Handle model selection
                st.subheader("Model Selection")
                models = ["Naive Bayes", "Logistic Regression", "Support Vector Machine (SVM)", "Random Forest", "XGBoost"]
                selected_models = st.multiselect("Select model(s) to train", models)

                # Handle model training and evaluation
                if st.button("Train Models", key="train_button") and df is not None:
                    st.subheader("Model Training and Evaluation")
                    
                    # Ensure the DataFrame has necessary columns
                    if 'text' not in df.columns or 'label' not in df.columns:
                        st.error("DataFrame does not have the required 'text' and 'label' columns.")
                        st.stop()
                    
                    try:
                        # Vectorize the text data
                        vectorizer = TfidfVectorizer()
                        X = vectorizer.fit_transform(df['text'])
                        y = df['label']

                        if len(df) > 1:
                            # Split data into training and test sets
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        else:
                            # Handle the case with insufficient data for splitting
                            X_train, X_test = X, X
                            y_train, y_test = y, y
                        
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
                                st.warning(f"Model {model_name} is not recognized.")
                                continue
                            
                            model.fit(X_train, y_train)
                            trained_models.append(model)
                        
                        # Evaluate model performance
                        performance_metrics = []
                        all_predictions = []
                        for model in trained_models:
                            y_pred = model.predict(X_test)
                            all_predictions.extend(y_pred)  # Collect all predictions for sentiment counts
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            # Use 'macro' average for multiclass
                            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
                            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                            
                            confusion = confusion_matrix(y_test, y_pred)
                            
                            performance_metrics.append({
                                "Model": model.__class__.__name__,
                                "Accuracy": accuracy,
                                "Precision": precision,
                                "Recall": recall,
                                "F1 Score": f1,
                                "Confusion Matrix": confusion
                            })
                        
                        # Display performance metrics
                        for metrics in performance_metrics:
                            st.write(f"Model: {metrics['Model']}")
                            st.write(f"Accuracy: {metrics['Accuracy']:.2f}")
                            st.write(f"Precision: {metrics['Precision']:.2f}")
                            st.write(f"Recall: {metrics['Recall']:.2f}")
                            st.write(f"F1 Score: {metrics['F1 Score']:.2f}")
                            st.write(f"Confusion Matrix:\n{metrics['Confusion Matrix']}")

                        # Count the number of each sentiment category
                        sentiment_counts = Counter(all_predictions)
                        st.subheader("Sentiment Counts")
                        for sentiment, count in sentiment_counts.items():
                            st.write(f"{sentiment}: {count}")

                        # Plot the bar chart
                        model_names = [metrics['Model'] for metrics in performance_metrics]
                        accuracies = [metrics['Accuracy'] for metrics in performance_metrics]
                        precisions = [metrics['Precision'] for metrics in performance_metrics]
                        recalls = [metrics['Recall'] for metrics in performance_metrics]
                        f1_scores = [metrics['F1 Score'] for metrics in performance_metrics]

                        plt.figure(figsize=(10, 6))
                        plt.bar(model_names, accuracies, label='Accuracy')
                        plt.bar(model_names, precisions, label='Precision', alpha=0.7)
                        plt.bar(model_names, recalls, label='Recall', alpha=0.7)
                        plt.bar(model_names, f1_scores, label='F1 Score', alpha=0.7)
                        plt.xlabel('Model')
                        plt.ylabel('Score')
                        plt.title('Performance Metrics')
                        plt.legend()
                        plt.xticks(rotation=45)
                        st.pyplot(plt)

                    except Exception as e:
                        st.error(f"An error occurred during model training or evaluation: {e}")

    else:
        st.warning("Please upload a CSV file or enter text to proceed.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
