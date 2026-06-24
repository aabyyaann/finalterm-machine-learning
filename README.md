# Final Term Machine Learning  
## UAS Machine Learning - Individual Task  
### Hands-on End-to-End Models Machine Learning & Deep Learning

---

## Author Identity
- **Name:** Naufal Alif Abyan  
- **NIM:** 101032300032  
- **Class:** S1 Teknik Komputer  
- **Course:** Machine Learning  
- **University:** Telkom University  

---

## Project Overview
This repository contains my **Final Term Project** for the **Machine Learning course**.  
The project focuses on building **end-to-end machine learning and deep learning pipelines** using the datasets provided in the UAS instructions.

There are **two main tasks** in this repository:

1. **Fraud Detection Classification**
   - Predict whether an online transaction is fraudulent or not.
   - Target variable: **`isFraud`**

2. **Song Release Year Regression**
   - Predict the release year of a song based on numerical audio features.
   - Target variable: **the first column of the dataset (Year)**

Both projects implement a complete machine learning workflow, including:
- data loading
- data understanding
- preprocessing
- feature engineering / feature selection
- model training
- hyperparameter tuning using **Optuna**
- evaluation using appropriate metrics
- model interpretation using **LIME**
- experiment tracking using **MLflow**

---

# Repository Purpose
The purpose of this repository is to demonstrate practical implementation of **end-to-end machine learning and deep learning workflows** for both **classification** and **regression** tasks.

This repository was created to fulfill the requirements of the **UAS Machine Learning Individual Task**, which emphasizes:
- end-to-end pipeline development
- data preprocessing and feature engineering
- model training and comparison
- hyperparameter tuning
- evaluation and analysis
- experiment tracking using MLflow
- documentation and reproducibility through GitHub

---

# Project 1 — Fraud Detection Classification

## Objective
The objective of this project is to build an **end-to-end classification model** that predicts the probability of an online transaction being fraudulent.

## Dataset
The dataset used for this project is:

- **`train_transaction.csv`**

### Dataset Description
This dataset contains transaction-related features such as:
- transaction amount
- product code
- card information
- address information
- email domains
- V-series anonymized variables
- and the target label **`isFraud`**

Each row represents one online transaction.

### Note
The project mainly uses **`train_transaction.csv`** as the primary dataset for fraud classification.  
If `train_identity.csv` is not used in the notebook, it is because the implemented pipeline focuses on the transaction table only.

---

## Fraud Detection Workflow
The classification workflow includes:

1. **Data Loading**
   - Load transaction dataset
   - Sample data if necessary to avoid memory crash in Google Colab

2. **Data Understanding**
   - Inspect shape, data types, and class distribution
   - Explore missing values and feature characteristics

3. **Missing Value Analysis**
   - Calculate missing percentage for each column
   - Drop columns with very high missing values

4. **Data Cleaning**
   - Separate numeric and categorical features
   - Impute missing numeric values with median
   - Impute missing categorical values with most frequent category

5. **Categorical Encoding**
   - Encode categorical variables using label encoding

6. **Feature Scaling**
   - Standardize numerical values using `StandardScaler`

7. **Train-Test Split**
   - Split the dataset into training and testing subsets

8. **Class Imbalance Handling**
   - Compute class weights because fraud data is highly imbalanced

9. **Baseline Deep Learning Model**
   - Build a simple MLP as the initial benchmark

10. **Model Improvement**
   - Add dropout and batch normalization
   - apply callbacks such as EarlyStopping and ReduceLROnPlateau

11. **Hyperparameter Tuning with Optuna**
   - Tune number of hidden units, dropout rate, and learning rate

12. **Model Evaluation**
   - Evaluate using classification metrics appropriate for imbalanced data

13. **MLflow Tracking**
   - Log parameters, metrics, and model artifacts

14. **Model Interpretation**
   - Use **LIME** to interpret one prediction from the best fraud model

15. **Save Artifacts**
   - Save the trained model and result files

---

## Models Used for Fraud Detection
The models implemented in this project are:

| Model | Description |
|------|-------------|
| **Baseline MLP** | Basic neural network model used as the first benchmark |
| **MLP + Dropout + BatchNormalization** | Improved deep learning architecture to reduce overfitting |
| **Optuna Tuned MLP** | Deep learning model with optimized hyperparameters |

---

## Fraud Detection Evaluation Metrics
Because fraud detection is an **imbalanced classification problem**, the model is evaluated using:

| Metric | Description |
|------|-------------|
| **Accuracy** | Measures overall prediction correctness |
| **Precision** | Measures how many predicted fraud cases are actually fraud |
| **Recall** | Measures how many actual fraud cases are detected |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Measures separability between fraud and non-fraud classes |
| **PR-AUC** | Important metric for imbalanced classification |
| **Confusion Matrix** | Shows TP, TN, FP, and FN predictions |

---

## Fraud Detection Result Summary
> **Important:** Replace the numbers below with the final results from your own notebook if they differ.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|------|----------|-----------|--------|---------|---------|--------|
| Baseline MLP | 0.8540 | 0.1605 | 0.7498 | 0.2644 | 0.8853 | 0.5044 |
| MLP + Dropout + BatchNorm | 0.7900 | 0.1172 | 0.7658 | 0.2034 | 0.8609 | 0.4347 |
| Optuna Tuned MLP | 0.8915 | 0.1986 | 0.6922 | 0.3086 | 0.8832 | 0.4919 |

---

## Fraud Detection Interpretation
In fraud detection, **accuracy alone is not enough** because the dataset is highly imbalanced.  
Therefore, the evaluation focuses more on:

- **Recall**, because the model should detect as many fraudulent transactions as possible
- **Precision**, to reduce false fraud alerts
- **ROC-AUC** and **PR-AUC**, because they are more informative for imbalanced classification

LIME is used to interpret one fraud prediction and show which features contribute most to the model decision.

---

# Project 2 — Song Release Year Regression

## Objective
The objective of this project is to build an **end-to-end regression pipeline** that predicts the **release year of a song** based on numerical audio features.

## Dataset
The dataset used in this project is:

- **`midterm-regresi-dataset.csv`**

### Dataset Description
The dataset structure is:

- **First column** = target label (**song release year**)
- **Remaining columns** = numerical audio features extracted from the music signal

Because the feature columns do not have human-readable names, they are renamed as:

- `feature_1`
- `feature_2`
- `feature_3`
- ...
- `feature_n`

---

## Regression Workflow
The regression workflow includes:

1. **Data Loading**
   - Load regression dataset from Google Drive or local path

2. **Column Renaming**
   - Rename the first column as `Year`
   - Rename remaining columns as `feature_1`, `feature_2`, ..., `feature_n`

3. **Data Understanding**
   - Check dataset shape, data types, and summary statistics
   - Visualize target distribution

4. **Missing Value Handling**
   - Check for missing values
   - Fill missing numeric values with median if necessary

5. **Outlier Handling**
   - Handle outliers on selected features using IQR capping if needed
   - Keep the process efficient to avoid memory issues in Colab

6. **Feature and Target Separation**
   - Separate input features (`X`) and target (`y`)

7. **Train-Test Split**
   - Split dataset into train and test subsets

8. **Feature Scaling**
   - Apply `StandardScaler` because deep learning models are sensitive to feature scale

9. **Optional Dimensionality Reduction**
   - Apply **PCA** to reduce dimensionality and improve training efficiency

10. **Baseline Models**
   - Train simple regression baselines for comparison, such as:
     - Linear Regression
     - Ridge Regression
     - Deep Learning MLP

11. **Deep Learning Regression Model**
   - Build an MLP regression model using TensorFlow/Keras

12. **Hyperparameter Tuning with Optuna**
   - Tune architecture and learning rate of the MLP model

13. **Model Evaluation**
   - Evaluate using regression metrics such as MSE, RMSE, MAE, and R²

14. **Visualization**
   - Compare actual vs predicted values
   - Plot residuals and error distributions

15. **MLflow Tracking**
   - Log experiment parameters, metrics, and trained model artifacts

16. **Model Interpretation**
   - Use **LIME Regression** to explain one prediction of the best regression model

17. **Save Artifacts**
   - Save best model, scaler, and result CSV files

---

## Models Used for Regression
The models implemented in this project are:

| Model | Description |
|------|-------------|
| **Linear Regression** | Baseline machine learning regression model |
| **Ridge Regression** | Regularized linear regression baseline |
| **Deep Learning MLP** | Neural network regression model |
| **Optuna Tuned Deep Learning MLP** | Tuned deep learning model with optimized hyperparameters |

> If your final notebook only compares some of these models, just keep the models that are actually used in your notebook.

---

## Regression Evaluation Metrics
The regression models are evaluated using:

| Metric | Description |
|------|-------------|
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error |
| **MAE** | Mean Absolute Error |
| **R² Score** | Proportion of variance explained by the model |

---

## Regression Result Summary
> **Important:** Replace the numbers below with the final results from your own regression notebook if they differ.

| Model | MSE | RMSE | MAE | R² |
|------|------|------|------|------|
| Linear Regression | 88.7762 | 9.4221 | 6.7022 | 0.2541 |
| Ridge Regression | 90.4204 | 9.5090 | 6.8373 | 0.2403 |
| Deep Learning MLP | 102.9840 | 10.1481 | 7.3246 | 0.1347 |
| Optuna Tuned Deep Learning MLP | 17473.5996 | 132.1877 | 62.2828 | -145.8185 |

---

## Regression Interpretation
The best regression model is selected based on:

- **lowest RMSE**
- **lowest MAE**
- **highest R²**

Interpretation of metrics:
- **RMSE** and **MAE** show how far the predicted year is from the actual year
- **R²** indicates how much target variance is explained by the model

LIME Regression is used to interpret one prediction and identify which audio features most influence the predicted song year.

---

# MLflow Tracking
Both projects use **MLflow** to track experiments.

The following items are logged in MLflow:
- model parameters
- Optuna best hyperparameters
- evaluation metrics
- trained model artifacts
- comparison reports
- additional experiment outputs

This helps make the experiments reproducible and easier to analyze.

---

# Repository Structure
Below is the recommended structure of this repository:

```bash
finalterm-machine-learning/
│
├── notebooks/
│   ├── fraud_detection_classification.ipynb
│   └── song_year_regression.ipynb
│
├── reports/
│   ├── fraud_metrics.csv
│   ├── fraud_confusion_matrix.png
│   ├── fraud_roc_curve.png
│   └── fraud_pr_curve.png
│
├── reports_regression/
│   ├── regression_metrics.csv
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   ├── error_distribution.png
│   └── lime_regression_plot.png
│
├── models/
│   ├── best_fraud_detection_model.keras
│   └── best_song_year_regression_model.keras
│
├── README.md
└── requirements.txt
