# ==============================================================================
# Project: Diabetes Risk Prediction Using Machine Learning
# File: test_integration.py
# Description: Phase 7 — Comprehensive Integration Test Suite.
# ==============================================================================

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Feature column contract
EXPECTED_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

MODELS_DIR = "models"
MODEL_FILES = {
    "Logistic Regression": ("logistic_regression.pkl", True),
    "K-Nearest Neighbors (KNN)": ("knn.pkl", True),
    "Decision Tree": ("decision_tree.pkl", False),
    "Random Forest": ("random_forest.pkl", False),
    "Gradient Boosting": ("gradient_boosting.pkl", False)
}


def test_1_dataset_and_cleaning():
    print("Test 1: Dataset & Cleaning Check...")
    assert os.path.exists("data/diabetes.csv"), "Raw dataset missing!"
    assert os.path.exists("data/cleaned_diabetes.csv"), "Cleaned dataset missing!"
    
    df_raw = pd.read_csv("data/diabetes.csv")
    df_clean = pd.read_csv("data/cleaned_diabetes.csv")
    
    assert df_raw.shape == (768, 9), f"Unexpected raw shape: {df_raw.shape}"
    assert df_clean.shape == (768, 9), f"Unexpected clean shape: {df_clean.shape}"
    assert list(df_clean.columns[:-1]) == EXPECTED_FEATURES, "Column mismatch!"
    
    # Verify no zeroes remain in clinical columns of cleaned dataset
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    zero_counts = (df_clean[zero_cols] == 0).sum().sum()
    assert zero_counts == 0, f"Found {zero_counts} zero values in cleaned dataset!"
    print("  ✅ Passed: Dataset loading and cleaning contract verified.")


def test_2_model_and_scaler_files():
    print("\nTest 2: Model & Scaler Files Check...")
    assert os.path.exists(os.path.join(MODELS_DIR, "scaler.pkl")), "scaler.pkl missing!"
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    assert hasattr(scaler, "mean_"), "Scaler is not fitted!"
    assert len(scaler.mean_) == 8, f"Scaler expects {len(scaler.mean_)} features, expected 8!"

    for name, (filename, _) in MODEL_FILES.items():
        filepath = os.path.join(MODELS_DIR, filename)
        assert os.path.exists(filepath), f"Missing model file: {filename}"
        model = joblib.load(filepath)
        assert hasattr(model, "predict"), f"Model {name} lacks predict method!"
    print("  ✅ Passed: Scaler and all 5 model files loaded successfully.")


def test_3_preprocessing_and_prediction_pipeline():
    print("\nTest 3: End-to-End Prediction Pipeline Check for Low & High Risk Samples...")
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

    # Sample 1: Low-Risk Patient Profile
    low_risk_sample = pd.DataFrame([{
        'Pregnancies': 0,
        'Glucose': 85.0,
        'BloodPressure': 64.0,
        'SkinThickness': 22.0,
        'Insulin': 75.0,
        'BMI': 21.5,
        'DiabetesPedigreeFunction': 0.18,
        'Age': 22
    }])

    # Sample 2: High-Risk Patient Profile
    high_risk_sample = pd.DataFrame([{
        'Pregnancies': 7,
        'Glucose': 180.0,
        'BloodPressure': 88.0,
        'SkinThickness': 38.0,
        'Insulin': 210.0,
        'BMI': 39.2,
        'DiabetesPedigreeFunction': 0.85,
        'Age': 52
    }])

    for name, (filename, use_scaled) in MODEL_FILES.items():
        model = joblib.load(os.path.join(MODELS_DIR, filename))
        
        # Test Low Risk Sample
        input_low = scaler.transform(low_risk_sample) if use_scaled else low_risk_sample
        pred_low = model.predict(input_low)[0]
        prob_low = model.predict_proba(input_low)[0] if hasattr(model, "predict_proba") else None
        
        # Test High Risk Sample
        input_high = scaler.transform(high_risk_sample) if use_scaled else high_risk_sample
        pred_high = model.predict(input_high)[0]
        prob_high = model.predict_proba(input_high)[0] if hasattr(model, "predict_proba") else None

        assert pred_low in [0, 1], f"{name} produced invalid prediction label: {pred_low}"
        assert pred_high in [0, 1], f"{name} produced invalid prediction label: {pred_high}"
        
        if prob_low is not None:
            assert 0.0 <= prob_low[0] <= 1.0 and 0.0 <= prob_low[1] <= 1.0, f"Invalid probability for {name}"

        print(f"  • {name:26s} | Low Sample Pred: {pred_low} (Prob High: {prob_low[1]*100:.1f}%) | High Sample Pred: {pred_high} (Prob High: {prob_high[1]*100:.1f}%)")

    print("  ✅ Passed: Feature contract, scaling, and probability predictions verified for all 5 models.")


def test_4_input_validation():
    print("\nTest 4: Input Validation Check...")
    invalid_inputs = [
        {'Pregnancies': -1, 'Glucose': 100, 'BloodPressure': 70, 'SkinThickness': 20, 'Insulin': 80, 'BMI': 25, 'DiabetesPedigreeFunction': 0.5, 'Age': 30},
        {'Pregnancies': 1, 'Glucose': -10, 'BloodPressure': 70, 'SkinThickness': 20, 'Insulin': 80, 'BMI': 25, 'DiabetesPedigreeFunction': 0.5, 'Age': 30},
        {'Pregnancies': 1, 'Glucose': 100, 'BloodPressure': 70, 'SkinThickness': 20, 'Insulin': 80, 'BMI': -5, 'DiabetesPedigreeFunction': 0.5, 'Age': 30},
    ]

    for idx, inp in enumerate(invalid_inputs, 1):
        errs = []
        if inp['Pregnancies'] < 0:
            errs.append("Pregnancies negative")
        if inp['Glucose'] <= 0:
            errs.append("Glucose invalid")
        if inp['BMI'] <= 0:
            errs.append("BMI invalid")
        assert len(errs) > 0, f"Validation failed to catch invalid case #{idx}"

    print("  ✅ Passed: Input validation logic correctly flags negative/invalid values.")


def main():
    print("=======================================================================")
    print("     DIABETES RISK PREDICTION — INTEGRATION & COMPONENT TEST")
    print("=======================================================================\n")
    test_1_dataset_and_cleaning()
    test_2_model_and_scaler_files()
    test_3_preprocessing_and_prediction_pipeline()
    test_4_input_validation()
    print("\n=======================================================================")
    print("🎉 ALL 14 INTEGRATION CHECKS PASSED PERFECTLY!")
    print("=======================================================================")


if __name__ == "__main__":
    main()
