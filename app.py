import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report, confusion_matrix

st.set_page_config(page_title="ML Model Evaluator", layout="wide")
st.title("Classification Model Evaluator")

# a. Dataset upload option
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Test Data Preview", data.head())
    
    # Separate features and target (assuming target is the last column named 'target')
    X_test = data.iloc[:, :-1]
    y_test = data.iloc[:, -1]
    
    # b. Model selection dropdown
    model_options = ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Support Vector Machine']
    selected_model_name = st.selectbox("Select a Machine Learning Model", model_options)
    
    if st.button("Evaluate Model"):
        model_filename = selected_model_name.replace(' ', '_') + '.joblib'
        model_path = os.path.join('model', model_filename)
        
        if os.path.exists(model_path):
            # Load the pre-trained model
            model = joblib.load(model_path)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
            
            # c. Display of evaluation metrics
            st.write("### Evaluation Metrics")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC", f"{auc:.4f}" if isinstance(auc, float) else auc)
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")
            
            # d. Confusion matrix & classification report
            col_rep, col_cm = st.columns(2)
            
            with col_rep:
                st.write("### Classification Report")
                # Convert the report to a dictionary, then to a styled DataFrame
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format(precision=3), use_container_width=True)
                
            with col_cm:
                st.write("### Confusion Matrix")
                # Corrected labels: 0 is Malignant, 1 is Benign
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(cm, 
                                     index=['Actual 0 (Malignant)', 'Actual 1 (Benign)'], 
                                     columns=['Predicted 0 (Malignant)', 'Predicted 1 (Benign)'])
                st.dataframe(cm_df.style.background_gradient(cmap='Blues'), use_container_width=True)
                
        else:
            st.error(f"Model file {model_filename} not found in the 'model' directory.")
