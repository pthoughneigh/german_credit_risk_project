# German Credit Risk Analysis & Logistic Regression (From Scratch)

A structured machine learning project for credit risk classification, built using Python, NumPy, Pandas, and Matplotlib.

## Goal

The goal of this project is to predict whether a loan applicant is a **good** or **bad** credit risk based on financial and demographic features.

This is a binary classification problem with real-world importance in banking, where incorrect predictions can lead to financial losses.

---

## Project Overview

This project implements a complete machine learning pipeline:

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Statistical Testing (t-test, chi-square)
4. Feature Engineering
5. Data Preprocessing
   - Target encoding
   - Ordinal encoding
   - One-hot encoding
   - Feature scaling
6. Stratified Train/Test Split
7. Logistic Regression (implemented from scratch using NumPy)
8. Model Evaluation

---

## Model Implementation

The logistic regression model is implemented from scratch without using high-level ML libraries.

Key components:
- Linear model: `z = Xw + b`
- Sigmoid activation function
- Binary cross-entropy (log loss)
- Gradient descent optimization

This approach demonstrates a deep understanding of the underlying mathematics of machine learning models.

---

## Results

### Train Set

| Threshold | Accuracy | Precision | Recall | F1 Score |
|----------|----------|----------|--------|----------|
| 0.5      | 0.71     | 0.54     | 0.18   | 0.27     |
| 0.3      | 0.64     | 0.43     | 0.63   | 0.52     |

### Test Set

| Threshold | Accuracy | Precision | Recall | F1 Score |
|----------|----------|----------|--------|----------|
| 0.5      | 0.705    | 0.52     | 0.18   | 0.27     |
| 0.3      | 0.63     | 0.43     | 0.68   | 0.53     |

---

## Key Insights

- Accuracy alone is misleading in imbalanced datasets.
- Lowering the classification threshold significantly improves recall.
- There is a clear trade-off between precision and recall.
- In credit risk modeling, minimizing false negatives (missing risky clients) is more important than maximizing accuracy.

---

## Project Structure
```bash
german_credit_risk_project/
├── data/
│   ├── raw/                    # original dataset
│   │   └── german_credit_data.csv
│   ├── processed/              # cleaned datasets
│   │   └── .gitkeep
│
├── outputs/
│   ├── figures/                # generated plots
│   │   └── .gitkeep
│   ├── reports/                # exported reports
│   │   ├── .gitkeep
│   │   └── export.py
│
├── src/
│   ├── data/                   # data loading and cleaning
│   │   ├── loader.py
│   │   └── cleaning.py
│   │
│   ├── features/               # feature engineering & preprocessing
│   │   ├── engineering.py
│   │   ├── preprocessing.py
│   │   └── splitting.py
│   │
│   ├── analysis/               # EDA and statistical tests
│   │   └── eda.py
│   │
│   ├── models/                 # ML models (from scratch)
│   │   └── logistic_regression.py
│   │
│   ├── evaluation/             # metrics and ROC/AUC
│   │   ├── metrics.py
│   │   └── roc_auc.py
│   │
│   ├── visualization/          # plotting utilities
│   │   └── plots.py
│   │
│   ├── config.py               # configuration (paths, columns)
│   └── main.py                 # pipeline entry point
│
├── .gitignore
├── README.md
├── requirements.txt


## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- SciPy

---

## Future Improvements
- Add regularization (L1/L2)
- Perform cross-validation