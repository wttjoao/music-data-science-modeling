# Music Data Science: Regression, Classification & Feature Selection

End-to-end data science project using a real-world music dataset to explore relationships between audio and metadata features, build predictive models, compare validation strategies, and study the impact of regularization and feature selection.

## Highlights

- Multiple Linear Regression improved the regression performance from **R² = 0.629** with the strongest single feature to **R² = 0.682** using a selected feature set.
- Logistic Regression achieved **76.6% test accuracy** and remained stable across 5-fold CV, 10-fold CV, LOOCV, and bootstrap validation.
- LDA achieved **75.3% test accuracy**, with similar cross-validation performance and low variance.
- Base QDA achieved **63.3% test accuracy**, while regularization with `reg_param=0.1` improved test accuracy to **71.4%**.
- The project compares model behaviour across regression, classification, resampling, regularization, and feature-selection techniques.

---

## Project Overview

Each observation represents a music track and contains metadata and audio-derived characteristics related to factors such as:

- artist popularity;
- album frequency;
- intensity;
- mood;
- signal characteristics;
- acoustic properties;
- duration;
- tempo;
- engineered features.

The project investigates two predictive tasks:

1. **Regression** — predicting a continuous target related to track success/popularity.
2. **Classification** — predicting one of three target classes.

The objective was not only to fit models, but to understand:

- which features are most informative;
- how different models compare;
- how validation strategy affects estimated performance;
- whether regularization improves generalization;
- whether feature selection can simplify models without sacrificing performance.

---

## Machine Learning Workflow

```text
Data Understanding
        ↓
Exploratory Data Analysis
        ↓
Preprocessing
        ↓
Feature Analysis
        ↓
Regression / Classification
        ↓
Model Validation
        ↓
Regularization
        ↓
Feature Selection
        ↓
Model Comparison
        ↓
Interpretation
```

---

## Exploratory Data Analysis

The analysis included:

- dataset structure and data types;
- descriptive statistics;
- missing-value analysis;
- duplicate detection;
- univariate distributions;
- bivariate relationships;
- target distributions;
- correlation analysis;
- relationships between features and target variables.

The working dataset contains **3,000 observations**.

The initial analysis found:

- no missing values requiring treatment;
- no duplicated observations requiring removal;
- three balanced target classes for classification.

EDA was also used to guide feature selection for the modelling stages.

---

# Regression

## Simple Linear Regression

Simple Linear Regression was evaluated independently across numerical features to identify the strongest single predictor of the regression target.

The best-performing feature was:

`artists_avg_popularity`

### Result

| Feature | R² | MAE | RMSE |
|---|---:|---:|---:|
| `artists_avg_popularity` | **0.629** | **0.244** | **0.491** |

Approximately 63% of the target variance could be explained using this feature alone.

This suggested a strong relationship between artist popularity and the regression target, while also showing that a single variable was not sufficient to explain the full behaviour of the data.

---

## Multiple Linear Regression

Multiple feature combinations were tested to determine whether additional predictors improved performance.

The best-performing combination was:

- `artists_avg_popularity`
- `album_freq`
- `mood_pca`
- `signal_strength`

### Result

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Simple Linear Regression | 0.629 | **0.244** | 0.491 |
| Multiple Linear Regression | **0.682** | 0.253 | **0.454** |

The multiple regression model increased explained variance and reduced RMSE compared with the strongest single-feature model.

The experiments also showed that adding more variables beyond this combination produced very limited additional improvement.

---

# Classification

The classification task predicts one of three target classes:

- `class_101`
- `class_41`
- `class_56`

The following models were compared:

- Logistic Regression
- Linear Discriminant Analysis
- Quadratic Discriminant Analysis

The classification models were evaluated using several validation strategies rather than relying on a single train/test split.

---

## Logistic Regression

Logistic Regression produced the strongest overall base classification performance.

### Holdout Performance

With the main model configuration:

- Training accuracy: **76.2%**
- Test accuracy: **76.6%**

The model showed very similar train and test performance, suggesting limited overfitting.

`class_41` was the strongest-performing class, while `class_56` was consistently more difficult to distinguish from the remaining classes.

### Resampling Results

| Validation Method | Mean Accuracy | Std. Dev. |
|---|---:|---:|
| 5-Fold Cross-Validation | **0.754** | 0.021 |
| 10-Fold Cross-Validation | 0.753 | 0.032 |
| LOOCV | 0.757 | — |
| Bootstrap | 0.753 | 0.017 |

The close agreement between holdout, cross-validation, and bootstrap performance suggests stable generalization.

> Note: the standard deviation of individual LOOCV scores is not interpreted in the same way as fold-level variance, since each validation set contains only one observation.

---

## Logistic Regression Regularization

Both L1 and L2 regularization were evaluated.

### Ridge / L2

- Training accuracy: **76.2%**
- Test accuracy: **76.3%**

### Lasso / L1

- Training accuracy: **75.9%**
- Test accuracy: **76.4%**

Regularization produced results very close to the original Logistic Regression model, suggesting that the base classifier already had a good bias-variance balance.

L1 regularization also provides a mechanism for reducing the number of active features through sparse coefficients.

---

## Linear Discriminant Analysis

LDA achieved performance close to Logistic Regression.

### Holdout Performance

- Training accuracy: **75.0%**
- Test accuracy: **75.3%**

### Resampling Results

| Validation Method | Mean Accuracy | Std. Dev. |
|---|---:|---:|
| 5-Fold Cross-Validation | **0.744** | 0.012 |
| 10-Fold Cross-Validation | 0.742 | 0.024 |
| LOOCV | 0.741 | — |
| Bootstrap | 0.746 | 0.019 |

The results indicate that LDA was relatively stable across different validation strategies.

`class_41` again showed the strongest classification performance, while `class_56` remained the most difficult class.

---

## LDA Regularization and Feature Selection

Two approaches were explored:

### Shrinkage / Ridge-style regularization

Using:

`LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")`

Results:

- Training accuracy: **74.9%**
- Test accuracy: **74.9%**

### L1-based feature selection

L1 Logistic Regression was used as a feature selector before training LDA.

Results:

- Training accuracy: **75.1%**
- Test accuracy: **75.8%**

This approach slightly improved test performance while also simplifying the input feature space.

---

## Quadratic Discriminant Analysis

The base QDA model performed considerably worse than Logistic Regression and LDA.

### Holdout Performance

- Training accuracy: **63.3%**
- Test accuracy: **63.3%**

### Resampling Results

| Validation Method | Mean Accuracy | Std. Dev. |
|---|---:|---:|
| 5-Fold Cross-Validation | 0.620 | 0.025 |
| 10-Fold Cross-Validation | 0.612 | 0.030 |
| LOOCV | 0.613 | — |
| Bootstrap | 0.637 | 0.023 |

The model showed consistent but weaker performance.

The class-level results showed:

- `class_41` recall: **0.96**
- `class_56` recall: **0.28**

This suggests that QDA's more flexible quadratic decision boundaries did not match this feature space particularly well.

---

## QDA Regularization

QDA was also tested with regularization.

### Ridge-style QDA

Using:

`QDA(reg_param=0.1)`

Results:

- Training accuracy: **74.2%**
- Test accuracy: **71.4%**
- 5-Fold CV mean accuracy: **0.711**
- 10-Fold CV mean accuracy: **0.709**

This represented a substantial improvement over the unregularized QDA model.

### L1-based Feature Selection + QDA

Results:

- Training accuracy: **63.5%**
- Test accuracy: **63.3%**

The L1-based feature reduction did not improve QDA performance.

---

# Overall Model Comparison

| Model | Test Accuracy | Main Observation |
|---|---:|---|
| Logistic Regression | **76.6%** | Best overall base classification performance |
| Logistic Regression + L1 | 76.4% | Similar performance with regularization |
| Logistic Regression + L2 | 76.3% | Stable relative to base model |
| LDA + L1 Feature Selection | 75.8% | Slight improvement over base LDA |
| LDA | 75.3% | Stable linear classifier |
| LDA + Shrinkage | 74.9% | Similar to base LDA |
| QDA + Regularization | 71.4% | Strong improvement over base QDA |
| QDA | 63.3% | Weakest base classifier |

The experiments indicate that the simpler linear decision boundaries used by Logistic Regression and LDA were better suited to this dataset than the base quadratic model.

---

# Validation Strategy

Several validation methods were used:

- Holdout
- 5-Fold Cross-Validation
- 10-Fold Cross-Validation
- Leave-One-Out Cross-Validation
- Bootstrap

Using multiple validation strategies helped assess whether the model conclusions were dependent on a particular train/test split.

For Logistic Regression and LDA, the resulting accuracies remained relatively close across methods, supporting the conclusion that both models generalized consistently.

---

# Feature Selection and Regularization

The project explored whether model complexity could be reduced while preserving predictive performance.

Methods included:

- L1 regularization;
- L2 regularization;
- Logistic Regression with regularization;
- LDA shrinkage;
- L1-based feature selection before LDA;
- QDA regularization;
- L1-based feature selection before QDA.

The experiments showed that regularization had limited impact on already stable linear classifiers, while it substantially improved QDA when using `reg_param`.

---

# Key Findings

- `artists_avg_popularity` was the strongest individual predictor for the regression task.
- Simple Linear Regression achieved **R² = 0.629** using this feature alone.
- Multiple Linear Regression improved performance to **R² = 0.682** using four selected features.
- Logistic Regression achieved the strongest base classification result with **76.6% test accuracy**.
- LDA performed similarly at **75.3% test accuracy** and remained stable across validation strategies.
- L1-based feature selection increased LDA test accuracy slightly to **75.8%**.
- Base QDA underperformed at **63.3%**, but regularization improved test accuracy to **71.4%**.
- `class_41` was consistently the easiest class to identify.
- `class_56` was consistently the most difficult class to distinguish.
- Using multiple resampling strategies produced more reliable conclusions than relying on a single holdout split.
- More complex decision boundaries did not automatically produce better predictive performance.

---

# Technologies

## Language

- Python

## Data Analysis

- Pandas
- NumPy

## Machine Learning

- Scikit-learn
- Linear Regression
- Logistic Regression
- Linear Discriminant Analysis
- Quadratic Discriminant Analysis
- Feature Selection
- L1 / L2 Regularization

## Model Validation

- Holdout
- K-Fold Cross-Validation
- Leave-One-Out Cross-Validation
- Bootstrap

## Evaluation

### Regression

- R²
- MAE
- RMSE

### Classification

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Visualization

- Matplotlib
- Seaborn

---

# Repository Structure

Adapt this section to the actual repository structure before publishing.

```text
.
├── data/
│   ├── dataset.csv
│   └── readme_dataset.txt
├── notebook.ipynb
├── results/
├── requirements.txt
└── README.md
```

---

# Running the Project

Clone the repository:

```bash
git clone https://github.com/<your-username>/music-data-science-modeling.git
cd music-data-science-modeling
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

---

# Academic Context

This project was developed as part of the **Machine Learning** coursework of the MSc in Informatics Engineering at ISEP.

The public repository has been reorganized to present the work as a reproducible data science case study, with emphasis on:

- exploratory data analysis;
- statistical modelling;
- model comparison;
- validation methodology;
- regularization;
- feature selection;
- interpretation of results.
