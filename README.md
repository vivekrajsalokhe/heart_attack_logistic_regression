# 🩺 Heart Attack Prediction in China using Logistic Regression

## 📘 Project Overview
This project aims to predict the likelihood of a **heart attack** among individuals in China using **Logistic Regression**, one of the most interpretable and effective algorithms for binary classification problems.

The notebook demonstrates the entire process of a typical data science workflow — from data loading, cleaning, and exploratory analysis to model building, evaluation, and interpretation.

---

## 📂 Dataset Information
**Dataset Name:** `heart_attack_china.csv`

This dataset includes various health and demographic factors that could influence heart disease risk.  
It helps in building a model that predicts whether an individual is at high risk of a heart attack.

### 🧾 Key Features (Columns)
| Feature | Description |
|----------|-------------|
| `Patient_ID` | Unique identifier for each record |
| `Age` | Age of the person |
| `Gender` | Male/Female |
| `Cholesterol` | Serum cholesterol level (mg/dL) |
| `Blood_Pressure` | Blood pressure in mm Hg |
| `Smoking` | 1 if the person smokes, else 0 |
| `Exercise` | 1 if the person exercises regularly, else 0 |
| `Diabetes` | 1 if diabetic, else 0 |
| `Obesity` | Indicator based on BMI or weight |
| `Heart_Attack` | Target variable — 1 if heart attack occurred, else 0 |

> 🧠 Note: The column names and values may differ slightly depending on the dataset used.

---

## ⚙️ Steps Performed

### 1️⃣ Data Preparation
- Imported essential libraries (`numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn`).
- Loaded the dataset and checked its structure using `df.info()` and `df.describe()`.
- Identified and handled missing or duplicate data.
- Normalized and encoded features as needed.

### 2️⃣ Exploratory Data Analysis (EDA)
- Visualized feature distributions and relationships.
- Created correlation heatmaps to identify key relationships.
- Used boxplots, countplots, and histograms to detect patterns and outliers.
- Compared the effect of risk factors like smoking, cholesterol, and exercise on heart attacks.

### 3️⃣ Feature Engineering
- Converted categorical data into numeric form using label or one-hot encoding.
- Scaled features using **StandardScaler** for model optimization.
- Split the dataset into training and testing sets (e.g., 80% train, 20% test).

### 4️⃣ Model Building
- Implemented **Logistic Regression** using `sklearn.linear_model`.
- Trained the model on the training set.
- Predicted outcomes on the test set.

### 5️⃣ Model Evaluation
- Evaluated using the following metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
- Visualized performance using Seaborn heatmaps and classification reports.

---

## 📊 Results Summary

| Metric | Value |
|--------|--------|
| **Accuracy** | *ADD YOUR VALUE HERE (e.g., 0.86)* |
| **Precision** | *ADD VALUE HERE* |
| **Recall** | *ADD VALUE HERE* |
| **F1-score** | *ADD VALUE HERE* |

### 🔍 Key Insights
- *Add your findings here* — for example:
  - High cholesterol and smoking are major contributors to heart attacks.
  - Regular exercise and healthy BMI lower heart disease risk.
  - Age and gender play significant roles in predicting outcomes.

---

## 🧠 Key Learnings
- Logistic Regression provides excellent interpretability for medical data.
- Data visualization helps uncover relationships that are not obvious numerically.
- Proper preprocessing (scaling, encoding) is essential for consistent model performance.

---

## 🚀 Future Scope
- Experiment with advanced models such as Random Forest, Gradient Boosting, or XGBoost.
- Perform hyperparameter tuning using GridSearchCV.
- Deploy the model via **Streamlit** or **Flask** for real-time user predictions.
- Include additional medical and lifestyle data for more accuracy.

---

## 🧩 Tech Stack

| Category | Tools |
|-----------|-------|
| **Language** | Python 3.x |
| **Libraries** | NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn |
| **Environment** | Jupyter Notebook |
| **Model Used** | Logistic Regression |

---

## 🧠 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Heart-Attack-China-Logistic-Regression.git
cd Heart-Attack-China-Logistic-Regression
