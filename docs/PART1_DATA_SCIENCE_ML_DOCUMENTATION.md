# Part 1 – Data Science and Machine Learning
## Financial Fraud Detection System

### 1. Overview

This document describes the Data Science and Machine Learning work completed for the Financial Fraud Detection System.

The objective of this part was to build a machine learning pipeline that analyses financial transaction information, identifies suspicious patterns, compares multiple classification algorithms, evaluates their performance, and provides a reusable prediction function for the backend.

### 2. Dataset Understanding

The project dataset contains 5,000 financial transaction records and 14 original columns.

The target column is `Fraudulent`:
- `0` – Genuine transaction
- `1` – Fraudulent transaction

Class distribution:
- Genuine transactions: 4,518 (90.36%)
- Fraudulent transactions: 482 (9.64%)

The dataset includes transaction amount, date, merchant category, payment method, device type, location, international transaction indicator, previous transaction count, average spending, account age, and other transaction information.

**Important:** The dataset is relatively small and imbalanced. Results from this dataset should not be interpreted as proof of production-level fraud detection performance. Its source and licence should be documented before redistribution.

### 3. Data Understanding and Validation

The following activities were completed:

- Inspected dataset dimensions, columns, data types, and sample records.
- Examined target-class distribution.
- Checked missing values and duplicate records.
- Checked uniqueness of transaction identifiers.
- Parsed transaction dates using the appropriate day-first format.
- Examined numerical and categorical variables.
- Identified `Suspicious_Keyword` as a potential target-leakage risk and excluded it from model inputs.

The initial dataset contained no missing values or duplicate rows, and transaction identifiers were unique.

### 4. Data Cleaning

The cleaning stage prepared a consistent dataset for further analysis.

Activities included:
- Date conversion and validation.
- Data type checks.
- Duplicate and missing-value checks.
- Validation of required columns and target values.
- Saving the cleaned dataset separately while preserving the original raw dataset.

### 5. Exploratory Data Analysis (EDA)

Exploratory analysis was performed to understand transaction patterns and possible relationships with fraud.

The analysis included:
- Fraud versus genuine transaction distribution.
- Transaction amount distribution.
- Numerical feature boxplots.
- Correlation heatmap.
- Fraud distribution across merchant categories.
- Fraud distribution across payment methods.
- Fraud distribution across device types and locations.
- International versus domestic transaction patterns.
- Monthly transaction trends.
- Analysis of suspicious keywords.

Charts and supporting outputs are stored under `reports/eda/`.

### 6. Feature Engineering

Seven additional features were created from the existing transaction information:

1. `Transaction_Hour` – hour of the transaction.
2. `Transaction_Day` – day of the month.
3. `Transaction_Month` – month of the transaction.
4. `Day_of_Week` – day of the week.
5. `Is_Weekend` – indicates whether the transaction occurred on a weekend.
6. `Amount_to_Average_Ratio` – transaction amount divided by average spending.
7. `Amount_Difference` – transaction amount minus average spending.

These features were designed to represent transaction timing and spending behaviour more effectively.

### 7. Data Preprocessing

The model input preparation included:

- Chronological sorting of transactions.
- An 80:20 chronological training/testing split for the initial modelling stage.
- Exclusion of identifier and non-model input columns.
- Exclusion of `Suspicious_Keyword` to reduce target-leakage risk.
- Median imputation and standardisation for numerical features.
- Most-frequent imputation and one-hot encoding for categorical features.
- Saving the fitted preprocessing pipeline for reuse.

The final model input contained 16 features: 12 numerical and 4 categorical. One-hot encoding produced 35 transformed features.

The preprocessing transformations were fitted on training data rather than on the complete dataset.

### 8. Model Development

Three classification algorithms were trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Class weighting was used to address the imbalanced target distribution.

The models were evaluated using metrics appropriate for fraud detection:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall AUC (PR-AUC)
- Confusion matrix

Accuracy was not treated as the only selection criterion because the dataset contains substantially more genuine transactions than fraudulent transactions.

### 9. Validation and Threshold Selection

A further chronological validation approach was used:

- 3,200 records for sub-training.
- 800 records for validation.
- 1,000 records for testing.

The preprocessing pipeline was fitted on the sub-training portion. Candidate probability thresholds were assessed using validation data, with F1-score used for threshold selection.

This allowed the classification threshold to be adjusted rather than automatically using 0.50.

**Evaluation limitation:** The 1,000-record test partition was used in earlier model comparisons during development. It should therefore not be described as a completely untouched final holdout. A fresh, later-period dataset is recommended for independent final evaluation.

### 10. Feature Importance and Analysis

Permutation feature importance was used to examine how input features affected validation performance.

The most prominent features included:
- `Transaction_Hour`
- `Is_International`
- `Account_Age_Days`
- `Day_of_Week`

Additional fraud-rate analysis was performed by transaction hour and international transaction status.

Feature importance indicates model dependence on a feature; it does not establish causation.

### 11. Hyperparameter Tuning

Hyperparameter tuning was performed for Logistic Regression and Random Forest.

- Random Forest: randomized search with 20 candidate configurations and four-fold TimeSeriesSplit cross-validation.
- Logistic Regression: randomized search with five candidate configurations.
- Average precision was used as the cross-validation scoring metric.

The selected Random Forest configuration was:
- `n_estimators`: 300
- `min_samples_split`: 5
- `min_samples_leaf`: 2
- `max_features`: `sqrt`
- `max_depth`: `None`

The tuned models were evaluated using validation-selected thresholds.

### 12. Model Comparison and Current Candidate

The tuned Random Forest was selected as the current preferred candidate because it achieved a stronger PR-AUC and a higher recall and F1-score than the tuned Logistic Regression in the reported test comparison.

Tuned Random Forest results at threshold 0.19:

- Accuracy: 0.777
- Precision: 0.2545
- Recall: 0.7955
- F1-score: 0.3857
- ROC-AUC: 0.8322
- PR-AUC: 0.3089

These figures are development results on the available dataset, not a guarantee of performance on real-world transactions. The relatively low precision means that a substantial number of genuine transactions may also be flagged. In a real financial system, flagged transactions should be reviewed using an appropriate risk and operational policy.

### 13. Prediction Module and Backend Handoff

`src/prediction.py` provides a reusable prediction function:

`predict_transaction(transaction_data)`

It:
- Loads the tuned Random Forest candidate.
- Accepts transaction information in the expected raw input format.
- Generates the engineered features consistently.
- Uses the saved model pipeline for preprocessing and prediction.
- Returns fraud probability, predicted class, and a suggested action.

The current threshold is `0.19`.

The returned fields are:
- `Fraud_Probability`
- `Prediction`
- `Suggested_Action`

Possible prediction labels:
- `Fraudulent`
- `Genuine`

Possible suggested actions:
- `Flag for Review`
- `Proceed`

The backend should pass the expected raw transaction fields to this function. It should not independently repeat scaling or one-hot encoding because those steps are already handled by the saved pipeline.

### 14. Main Project Files

| Folder | Purpose |
|---|---|
| `src/` | Data processing, EDA, feature engineering, modelling, evaluation, tuning, and prediction scripts |
| `data/raw/` | Original dataset |
| `data/processed/` | Cleaned and feature-engineered datasets |
| `data/processed/model_ready/` | Train/test data and preprocessing metadata |
| `models/` | Saved preprocessing and trained model pipelines |
| `reports/` | Evaluation results, data-quality reports, feature analysis, and charts |
| `docs/` | Part 1 technical documentation |

### 15. Dependencies

The project uses Python and the following main libraries:

- pandas
- NumPy
- scikit-learn
- joblib
- matplotlib
- openpyxl

The exact package versions and additional dependencies are recorded in `requirements.txt`.

### 16. Challenges and Learning Outcomes

During this part, the following practical skills were developed:

- Understanding and validating a real-world-style transaction dataset.
- Cleaning and preparing data for machine learning.
- Performing exploratory data analysis.
- Creating useful features from transaction dates and spending behaviour.
- Handling categorical and numerical features.
- Understanding class imbalance in fraud detection.
- Training and comparing classification algorithms.
- Understanding precision, recall, F1-score, ROC-AUC, and PR-AUC.
- Applying chronological validation and threshold selection.
- Performing hyperparameter tuning.
- Analysing feature importance.
- Saving models and preprocessing pipelines.
- Building a reusable prediction function for backend integration.

Important modelling challenges included class imbalance, the risk of target leakage, selecting an appropriate classification threshold, and balancing fraud detection recall against false positives.

### 17. Conclusion and Handoff

Part 1 established the Data Science and Machine Learning foundation of the Financial Fraud Detection System.

The work covers the process from dataset understanding and validation through cleaning, EDA, feature engineering, preprocessing, model development, evaluation, tuning, and transaction-level prediction.

The current preferred model candidate is the tuned Random Forest, supported by a saved pipeline and a reusable prediction module.

The output of this part is ready for integration with the next application/backend stage. Further work should include independent evaluation on new transactions, threshold calibration based on business costs, monitoring for data drift, and production-level testing before deployment.

**Status:** Part 1 – Data Science and Machine Learning development completed; integration and independent production validation remain future tasks.