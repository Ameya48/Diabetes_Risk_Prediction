# ==============================================================================
# Project: Diabetes Risk Prediction Using Machine Learning
# File: train_models.py
# Description: Beginner-friendly script for training, evaluating, and saving 
#              5 machine learning classification algorithms.
# ==============================================================================

import os
import pandas as pd
import numpy as np
import joblib

# Scikit-learn imports for data splitting & scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Scikit-learn imports for 5 Classification Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Directory configurations
MODELS_DIR = "models"
SAVED_MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Path settings
CLEANED_DATA_PATH = "data/cleaned_diabetes.csv"
RAW_DATA_PATH = "data/diabetes.csv"
DATA_PATH = CLEANED_DATA_PATH if os.path.exists(CLEANED_DATA_PATH) else RAW_DATA_PATH


def main():
    print("=======================================================================")
    print("      DIABETES RISK PREDICTION — PHASE 3 MODEL TRAINING")
    print("=======================================================================\n")

    # -------------------------------------------------------------------------
    # STEP 1: LOAD DATASET & PREPARE FEATURES (X) AND TARGET (y)
    # -------------------------------------------------------------------------
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset successfully loaded! Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")

    # CONCEPT EXPLANATION FOR EXAM:
    # X (Features): The independent input attributes used to make predictions
    #               (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Pedigree, Age).
    # y (Target):   The dependent binary label we want to predict 
    #               (0 = Low Diabetes Risk, 1 = High Diabetes Risk).
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    print(f"\nFeature Matrix (X) Shape: {X.shape}")
    print(f"Target Label (y) Shape:   {y.shape}")

    # CONCEPT EXPLANATION FOR EXAM:
    # Why train_test_split is used:
    # We split our data into 80% Training set (to teach the ML models) and 20% Testing set
    # (unseen data used to objectively measure how well models generalize to new patients).
    # test_size=0.2 means 20% test data. random_state=42 ensures repeatable results.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set count: {X_train.shape[0]} samples")
    print(f"Testing set count:  {X_test.shape[0]} samples")

    # -------------------------------------------------------------------------
    # STEP 2: FEATURE SCALING
    # -------------------------------------------------------------------------
    # CONCEPT EXPLANATION FOR EXAM:
    # Why Feature Scaling (StandardScaler) is used:
    # Features have vastly different numeric ranges (e.g., Glucose 0-200, Pedigree 0.08-2.4).
    # Distance-based models (KNN) and gradient-based models (Logistic Regression) perform poorly
    # if feature ranges differ hugely, because larger features dominate calculations.
    # StandardScaler transforms features so they have Mean = 0 and Standard Deviation = 1.
    #
    # IMPORTANT MODEL SCALING DISTINCTION:
    # 1. Logistic Regression & KNN REQUIRE scaled data for accurate distance/weight calculations.
    # 2. Tree-based models (Decision Tree, Random Forest, Gradient Boosting) rely on decision split thresholds
    #    and DO NOT require scaling.
    scaler = StandardScaler()
    
    # fit_transform on X_train calculates mean and standard deviation from training data and scales it.
    X_train_scaled = scaler.fit_transform(X_train)
    # transform on X_test applies the exact same mean and standard deviation learned from X_train.
    X_test_scaled = scaler.transform(X_test)

    # Save the fitted scaler for future web application predictions
    scaler_filename = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_filename)
    joblib.dump(scaler, os.path.join(SAVED_MODELS_DIR, "scaler.pkl"))
    print(f"\nScaler successfully saved to: {scaler_filename}")

    # -------------------------------------------------------------------------
    # STEP 3 & 4: INITIALIZE, TRAIN, PREDICT & EVALUATE 5 MODELS
    # -------------------------------------------------------------------------
    # CONCEPT EXPLANATION FOR EXAM METHOD FUNCTIONS:
    # - fit(X, y): Trains the ML algorithm by finding mathematical patterns between X and y.
    # - predict(X): Uses learned patterns to assign binary output labels (0 or 1) for input X.
    # - predict_proba(X): Output confidence probabilities [P(Outcome=0), P(Outcome=1)] for input X.

    models = {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=42),
            "use_scaled": True,
            "filename": "logistic_regression.pkl"
        },
        "K-Nearest Neighbors (KNN)": {
            "model": KNeighborsClassifier(n_neighbors=5),
            "use_scaled": True,
            "filename": "knn.pkl"
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "use_scaled": False,
            "filename": "decision_tree.pkl"
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, random_state=42),
            "use_scaled": False,
            "filename": "random_forest.pkl"
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "use_scaled": False,
            "filename": "gradient_boosting.pkl"
        }
    }

    print("\n-----------------------------------------------------------------------")
    print("                TRAINING & EVALUATING THE 5 MODELS")
    print("-----------------------------------------------------------------------")

    model_results = {}

    for name, config in models.items():
        clf = config["model"]
        use_scaled = config["use_scaled"]
        filename = config["filename"]

        # Select scaled or raw features depending on model requirements
        if use_scaled:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test

        # 1. Fit (Train model)
        clf.fit(X_tr, y_train)

        # 2. Predict class labels (0 or 1)
        predictions = clf.predict(X_te)

        # 3. Predict class probabilities
        probabilities = clf.predict_proba(X_te)

        # 4. Calculate test accuracy
        accuracy = accuracy_score(y_test, predictions)

        # ---------------------------------------------------------------------
        # STEP 5: SAVE TRAINED MODELS WITH JOBLIB
        # ---------------------------------------------------------------------
        save_path = os.path.join(MODELS_DIR, filename)
        joblib.dump(clf, save_path)
        joblib.dump(clf, os.path.join(SAVED_MODELS_DIR, filename))

        # Store for display
        model_results[name] = {
            "accuracy": accuracy,
            "predictions": predictions,
            "probabilities": probabilities,
            "path": save_path
        }

        print(f"✅ {name:30s} | Test Accuracy: {accuracy*100:.2f}% | Saved to: {save_path}")

    # -------------------------------------------------------------------------
    # FINAL CONFIRMATION SUMMARY
    # -------------------------------------------------------------------------
    print("\n=======================================================================")
    print("🎉 SUCCESS: All 5 Machine Learning models trained and saved successfully!")
    print("=======================================================================")
    print(f"Models directory: {os.path.abspath(MODELS_DIR)}")
    print("Saved files:")
    for f in os.listdir(MODELS_DIR):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
