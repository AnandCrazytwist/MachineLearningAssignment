import pandas as pd
import os
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 1. Load Dataset
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

# 2. Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save test data to CSV for Streamlit app
test_df = pd.concat([X_test, y_test], axis=1)
test_df.rename(columns={data.target_names[0]: 'target'}, inplace=True) # Ensure target column is identifiable
test_df.to_csv('test_data.csv', index=False)

# 3. Define Models
# Using pipelines for models that require feature scaling (LR, KNN, SVM)
models = {
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression()),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier()),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)) 
    # probability=True is required for calculating AUC Score
}

# 4. Train and Save Models
os.makedirs('model', exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    filename = name.replace(' ', '_') + '.joblib'
    joblib.dump(model, f'model/{filename}')
    print(f"Trained and saved {name}")

print("Training complete. test_data.csv and model artifacts generated.")