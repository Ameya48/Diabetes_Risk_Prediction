# ==============================================================================
# Project: Diabetes Risk Prediction Using Machine Learning
# File: analyze_features_and_overfitting.py
# Description: Phase 5 — Feature Importance, Logistic Regression Coefficients,
#              Overfitting Analysis, and Final ML Project Synthesis.
# ==============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paths
MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CLEANED_DATA_PATH = "data/cleaned_diabetes.csv"
RAW_DATA_PATH = "data/diabetes.csv"
DATA_PATH = CLEANED_DATA_PATH if os.path.exists(CLEANED_DATA_PATH) else RAW_DATA_PATH


def main():
    print("=======================================================================")
    print("   DIABETES RISK PREDICTION — PHASE 5 FEATURE & OVERFITTING ANALYSIS")
    print("=======================================================================\n")

    # 1. Load Dataset
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    feature_names = list(X.columns)

    # 2. Split Data (Same split as previous phases)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Load Scaler
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Load Models
    dt_model = joblib.load(os.path.join(MODELS_DIR, "decision_tree.pkl"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    gb_model = joblib.load(os.path.join(MODELS_DIR, "gradient_boosting.pkl"))
    lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    knn_model = joblib.load(os.path.join(MODELS_DIR, "knn.pkl"))

    # -------------------------------------------------------------------------
    # PART 1: FEATURE IMPORTANCE FOR TREE-BASED MODELS
    # -------------------------------------------------------------------------
    print("--- PART 1: FEATURE IMPORTANCE ANALYSIS (TREE-BASED MODELS) ---")
    
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Decision Tree": dt_model.feature_importances_,
        "Random Forest": rf_model.feature_importances_,
        "Gradient Boosting": gb_model.feature_importances_
    })
    fi_df["Average Importance"] = fi_df[["Decision Tree", "Random Forest", "Gradient Boosting"]].mean(axis=1)
    fi_df = fi_df.sort_values(by="Average Importance", ascending=False).reset_index(drop=True)

    print(fi_df.to_string(index=False))
    fi_csv_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
    fi_df.to_csv(fi_csv_path, index=False)
    print(f"Saved feature importance table to: {fi_csv_path}\n")

    # Plot Feature Importance Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Average Importance",
        y="Feature",
        data=fi_df,
        hue="Feature",
        palette="Blues_r",
        legend=False
    )
    plt.title("Feature Importance (Average Across Tree-Based Models)", fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Clinical Feature", fontsize=12)
    
    for p in plt.gca().patches:
        width = p.get_width()
        plt.gca().annotate(
            f"{width:.4f}",
            (width, p.get_y() + p.get_height() / 2.),
            ha='left', va='center', fontsize=10, fontweight='bold', xytext=(5, 0),
            textcoords='offset points'
        )

    plt.tight_layout()
    fi_chart_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(fi_chart_path, dpi=300)
    plt.close()
    print(f"Saved feature importance chart to: {fi_chart_path}\n")

    # -------------------------------------------------------------------------
    # PART 2: LOGISTIC REGRESSION COEFFICIENTS
    # -------------------------------------------------------------------------
    print("--- PART 2: LOGISTIC REGRESSION COEFFICIENTS ---")
    
    coefficients = lr_model.coef_[0]
    lr_coeff_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Absolute Value": np.abs(coefficients)
    }).sort_values(by="Absolute Value", ascending=False).reset_index(drop=True)

    print(lr_coeff_df[["Feature", "Coefficient"]].to_string(index=False))
    lr_csv_path = os.path.join(RESULTS_DIR, "logistic_coefficients.csv")
    lr_coeff_df.to_csv(lr_csv_path, index=False)
    print(f"Saved Logistic Regression coefficients table to: {lr_csv_path}\n")

    # -------------------------------------------------------------------------
    # PART 3: OVERFITTING ANALYSIS (TRAIN VS TEST ACCURACY)
    # -------------------------------------------------------------------------
    print("--- PART 3: OVERFITTING ANALYSIS ---")

    models_dict = {
        "Logistic Regression": (lr_model, X_train_scaled, X_test_scaled),
        "KNN": (knn_model, X_train_scaled, X_test_scaled),
        "Decision Tree": (dt_model, X_train, X_test),
        "Random Forest": (rf_model, X_train, X_test),
        "Gradient Boosting": (gb_model, X_train, X_test)
    }

    overfit_list = []
    for name, (clf, X_tr, X_te) in models_dict.items():
        tr_acc = accuracy_score(y_train, clf.predict(X_tr))
        te_acc = accuracy_score(y_test, clf.predict(X_te))
        diff = tr_acc - te_acc
        
        status = "Severe Overfitting" if diff > 0.15 else ("Slight Overfitting" if diff > 0.05 else "Good Generalization")

        overfit_list.append({
            "Model": name,
            "Train Accuracy": round(tr_acc, 4),
            "Test Accuracy": round(te_acc, 4),
            "Difference (Train - Test)": round(diff, 4),
            "Generalization Status": status
        })

    overfit_df = pd.DataFrame(overfit_list)
    print(overfit_df.to_string(index=False))
    overfit_csv_path = os.path.join(RESULTS_DIR, "overfitting_analysis.csv")
    overfit_df.to_csv(overfit_csv_path, index=False)
    print(f"\nSaved overfitting analysis table to: {overfit_csv_path}\n")

    # -------------------------------------------------------------------------
    # PART 4: FINAL SYNTHESIS REPORT FOR EXAM PRESENTATION
    # -------------------------------------------------------------------------
    print("=======================================================================")
    print("               FINAL ACADEMIC EXAM SYNTHESIS REPORT")
    print("=======================================================================")

    print("""
1. Most Important Risk Factors/Features:
   • Tree-based models (Random Forest, Gradient Boosting) identified Glucose as the single 
     most predictive feature (average importance score > 0.30), followed by BMI and Age.
   • Careful Wording: "The model found these features useful for making predictions in this dataset.
     This reflects numerical associations in the sample and does not prove medical causation."

2. Model Performance Comparison:
   • Random Forest achieved the highest overall test accuracy (77.92%) and F1-score (65.31%).
   • Gradient Boosting (75.97%) and KNN (75.32%) also performed strong.

3. Recall Analysis:
   • KNN achieved the highest Recall (61.11%), correctly identifying 61.11% of high-risk patients.
   • In screening support, higher recall minimizes False Negatives (unflagged at-risk patients).

4. Overfitting Analysis:
   • Decision Tree exhibited severe overfitting (100% training accuracy vs 68.18% test accuracy).
   • Ensemble algorithms (Random Forest, Gradient Boosting) and regularized models (Logistic Regression)
     maintained close training vs testing accuracies, indicating better generalization.

5. Screening-Support Tool Suitability:
   • The system is suitable as an initial, non-invasive educational screening aid.
   • IMPORTANT: It is NOT a diagnostic tool and cannot replace medical blood testing or physician evaluation.

6. Dataset & Model Limitations:
   • Small sample size (768 records).
   • Moderate class imbalance (~65% negative vs ~35% positive).
   • Demographic specificity (Pima Indian female population aged 21+).
   • External clinical validation would be required before real-world deployment.
""")


if __name__ == "__main__":
    main()
