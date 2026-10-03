# Diabetes Risk Prediction Using Machine Learning

An academic machine learning exam project designed for predicting diabetes risk based on diagnostic measurements from the **Pima Indians Diabetes Dataset**.

---

## 📌 Project Overview & Objective
The primary objective of this project is to build an intelligent, beginner-friendly Machine Learning screening-support system that predicts whether a patient is at **Low Risk** (`Outcome = 0`) or **High Risk** (`Outcome = 1`) of diabetes. 

It implements five classification algorithms:
1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Tree**
4. **Random Forest**
5. **Gradient Boosting**

---

## 📊 Dataset Overview

- **Source:** Pima Indians Diabetes Dataset (`data/diabetes.csv` & `data/cleaned_diabetes.csv`).
- **Records:** 768 patient samples, 9 attributes.

### Features & Columns:
1. **Pregnancies**: Number of times pregnant.
2. **Glucose**: Plasma glucose concentration (2-hour oral test, mg/dL).
3. **BloodPressure**: Diastolic blood pressure (mm Hg).
4. **SkinThickness**: Triceps skin fold thickness (mm).
5. **Insulin**: 2-Hour serum insulin level (mu U/ml).
6. **BMI**: Body Mass Index (weight in kg / (height in m)²).
7. **DiabetesPedigreeFunction**: Genetic diabetes family history score.
8. **Age**: Patient age in years.
9. **Outcome** *(Target Variable)*: `0` = Low Risk / Negative, `1` = High Risk / Positive.

---

## 🏆 Key Results Summary

| Model Name | Scaled Data Used? | Accuracy | Precision | **Recall** ⭐ | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | No | **77.92%** 🥇 | **72.73%** | 59.26% | **65.31%** |
| **Gradient Boosting** | No | **75.97%** | 68.89% | 57.41% | 62.63% |
| **KNN** | Yes | **75.32%** | 66.00% | **61.11%** 🏆 | 63.46% |
| **Logistic Regression** | Yes | **70.78%** | 60.00% | 50.00% | 54.55% |
| **Decision Tree** | No | **68.18%** | 55.32% | 48.15% | 51.49% |

- **Top Predictor Features:** **Glucose** (Importance $> 0.34$), **BMI** ($> 0.17$), and **Age** ($> 0.11$).

---

## 📁 Project Folder Structure

```text
DiabetesRiskPrediction/
│
├── data/
│   ├── diabetes.csv                       # Raw Pima Indians Dataset
│   └── cleaned_diabetes.csv               # Preprocessed & median-imputed dataset
│
├── models/
│   ├── scaler.pkl                         # Fitted StandardScaler object
│   ├── logistic_regression.pkl            # Trained Logistic Regression classifier
│   ├── knn.pkl                            # Trained KNN classifier
│   ├── decision_tree.pkl                  # Trained Decision Tree classifier
│   ├── random_forest.pkl                  # Trained Random Forest classifier
│   └── gradient_boosting.pkl              # Trained Gradient Boosting classifier
│
├── notebooks/
│   ├── diabetes_analysis.ipynb            # Phase 2 EDA notebook
│   ├── model_evaluation.ipynb             # Phase 4 Model Evaluation notebook
│   └── feature_and_overfitting_analysis.ipynb # Phase 5 Feature Importance notebook
│
├── results/
│   ├── model_comparison.csv               # Phase 4 Evaluation metrics table (CSV)
│   ├── model_metrics_comparison.png       # Phase 4 Metric bar charts
│   ├── confusion_matrices.png             # Phase 4 Confusion matrix visual grid
│   ├── feature_importance.csv             # Phase 5 Feature importances table (CSV)
│   ├── feature_importance.png             # Phase 5 Feature importance bar chart
│   ├── logistic_coefficients.csv          # Phase 5 Logistic regression coefficients
│   └── overfitting_analysis.csv           # Phase 5 Train vs Test overfitting table
│
├── app.py                                 # Phase 6 Streamlit Web Application
├── train_models.py                        # Phase 3 ML Model training script
├── evaluate_models.py                     # Phase 4 Model evaluation script
├── analyze_features_and_overfitting.py    # Phase 5 Feature importance script
├── test_integration.py                    # Phase 7 Integration test suite
├── VIVA_PREPARATION_GUIDE.md              # Phase 8 Exam & Viva study guide
├── requirements.txt                       # Project Python dependencies
└── README.md                              # Comprehensive project documentation
```

---

## ⚙️ How to Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python train_models.py
```

### 3. Run Web Application
```bash
streamlit run app.py
```

Access the application in your browser at `http://localhost:8501`.

---

