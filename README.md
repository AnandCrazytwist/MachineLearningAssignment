# ML Classification Evaluator

**a. Problem statement**
To build and evaluate multiple machine learning classification models on a clinical dataset to accurately predict outcomes, and to deploy the evaluation system as an interactive web application.

**b. Dataset description**
- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** scikit-learn datasets (originally from UCI Machine Learning Repository)
- **Instances:** 569
- **Features:** 30 (Numeric computed from a digitized image of a fine needle aspirate (FNA) of a breast mass)
- **Target:** Binary (Malignant vs. Benign)

**c. Github Repository Link**
(https://github.com/AnandCrazytwist/MachineLearningAssignment/)

streamlit link : https://machinelearningassignment-xkwljep3h3mfwmrbwtzqff.streamlit.app/

**d. Models used & Comparison Table**


| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | 0.9737 | 0.9974 | 0.9722 | 0.9859 | 0.9790 | 0.9439 |
| **Decision Tree** | 0.9474 | 0.9440 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| **KNN** | 0.9474 | 0.9820 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| **Naive Bayes** | 0.9737 | 0.9984 | 0.9595 | 1.0000 | 0.9793 | 0.9447 |
| **Support Vector Machine** | 0.9825 | 0.9974 | 0.9726 | 1.0000 | 0.9861 | 0.9630 |

**Observations on model performance**

| ML Model Name | Observation about model performance |
| --- | --- |
| **Logistic Regression** | Provided a strong baseline model. The accuracy was high at 97.37%, showing a good balance between precision and recall. A simple linear decision boundary proved to be quite effective for this dataset. |
| **Decision Tree** | Resulted in the lowest accuracy (94.74%) and MCC (88.80%) among the models tested. This drop in performance indicates that the tree likely overfit the training data and struggled to generalize to the unseen test set. |
| **KNN** | Tied with the Decision Tree on accuracy, precision, and recall, but achieved a much higher AUC (98.20%). This suggests the underlying probability estimates were better, even if the final hard classifications resulted in some errors. |
| **Naive Bayes** | Achieved a perfect recall score of 1.000 with zero false negatives. Despite the algorithm's assumption of feature independence, flagging 100% of the positive cases makes it a highly useful model for medical diagnostics. |
| **Support Vector Machine** | Maintained the perfect 1.000 recall seen in Naive Bayes while improving precision to 97.26%. By finding an optimal hyperplane, this model delivered the highest overall accuracy (98.25%), F1 score (98.61%), and MCC (96.30%). |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)**. It achieved the highest overall metrics (Accuracy, F1, and MCC) while maintaining a 100% recall rate, which is critical for minimizing missed positive cases in clinical data. |
