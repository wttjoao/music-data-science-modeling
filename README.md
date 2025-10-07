# 🎧 APRAU Project 2025/26 — Group 13  
**Master in Informatics Engineering – ISEP**

## 📘 General Description
This project is part of the **Machine Learning (APRAU)** course and aims to apply different *machine learning* methods to a dataset containing musical track characteristics.  
Each instance in the dataset represents a music track and includes metadata and audio analysis measurements (such as energy, tempo, intensity, purity, and others).  

The main goals are:
- Perform **exploratory data analysis (EDA)**;  
- Develop and evaluate **regression** and **classification** models;  
- Investigate how **feature selection** impacts model performance.  

---

## 🎯 Specific Objectives

### 1. Exploratory Data Analysis (EDA)
- Descriptive statistics;  
- Univariate analysis (feature distributions);  
- Bivariate analysis (correlations between features and target variables).  

### 2. Regression
- **Simple Linear Regression** — using a single feature;  
- **Multiple Linear Regression** — using several features;  
- Model evaluation and comparison using the following metrics:
  - R²  
  - MAE (*Mean Absolute Error*)  
  - RMSE (*Root Mean Squared Error*)  

### 3. Classification
- Applied methods:
  - **Logistic Regression**
  - **Linear Discriminant Analysis (LDA)**
  - **Quadratic Discriminant Analysis (QDA)**
- Evaluation using different resampling methods:
  - Holdout  
  - Cross-Validation (k=5 and k=10)  
  - Leave-One-Out Cross Validation (LOOCV)  
  - Bootstrap  
- Discuss how variance is affected by each resampling approach.  

### 4. Feature Selection
- Test whether models using fewer features achieve better performance;  
- Apply **regularization-based feature selection** methods.

---

## 🧠 Project Structure
```ruby
📦 APRAU_Group13
┣ 📂 data
┃ ┣ dataset.csv
┃ ┗ readme_dataset.txt
┣ 📂 notebooks
┃ ┣ step1_exploratory_analysis.ipynb
┃ ┣ step2_regression_models.ipynb
┃ ┣ step3_classification_models.ipynb
┃ ┗ step4_feature_selection.ipynb
┣ 📂 results
┃ ┣ regression_metrics.csv
┃ ┣ classification_results.csv
┃ ┗ feature_importance.png
┣ 📜 README.md
┗ 📜 requirements.txt
```

---

## ⚙️ Technologies and Libraries
- **Language:** Python 3.11+  
- **Main Libraries:**
  - `pandas`, `numpy`, `matplotlib`, `seaborn` → Data analysis and visualization  
  - `scikit-learn` → Machine learning models and validation  

---

## 🧩 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/aprau_group13.git
   cd aprau_group13
