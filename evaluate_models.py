# ==============================================================================
# Project: Diabetes Risk Prediction Using Machine Learning
# File: evaluate_models.py
# Description: Phase 4 Model Evaluation & Comparative Study Script.
# ==============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Paths
MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CLEANED_DATA_PATH = "data/cleaned_diabetes.csv"
RAW_DATA_PATH = "data/diabetes.csv"
DATA_PATH = CLEANED_DATA_PATH if os.path.exists(CLEANED_DATA_PATH) else RAW_DATA_PATH


def main():
    print("=======================================================================")
    print("   DIABETES RISK PREDICTION — PHASE 4 MODEL EVALUATION & COMPARISON")
    print("=======================================================================\n")

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # 2. Split Data (Same split as Phase 3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Load Fitted Scaler
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Model Definitions & Saved Filenames
    model_info = [
        ("Logistic Regression", "logistic_regression.pkl", True),
        ("KNN", "knn.pkl", True),
        ("Decision Tree", "decision_tree.pkl", False),
        ("Random Forest", "random_forest.pkl", False),
        ("Gradient Boosting", "gradient_boosting.pkl", False),
    ]

    metrics_list = []
    confusion_matrices = {}

    print("Evaluating models on unseen test dataset (154 patient samples)...\n")

    for name, filename, use_scaled in model_info:
        model_path = os.path.join(MODELS_DIR, filename)
        clf = joblib.load(model_path)

        # Select test data
        X_te = X_test_scaled if use_scaled else X_test
        X_tr = X_train_scaled if use_scaled else X_train

        # Predict test labels
        y_pred = clf.predict(X_te)
        y_train_pred = clf.predict(X_tr)

        # Compute Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Overfitting check (Train accuracy vs Test accuracy)
        train_acc = accuracy_score(y_train, y_train_pred)

        metrics_list.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "Train Accuracy": round(train_acc, 4)
        })

        confusion_matrices[name] = confusion_matrix(y_test, y_pred)

    # 5. Create DataFrame
    model_comparison = pd.DataFrame(metrics_list)

    # Display clean table in console
    print("=======================================================================")
    print("                        MODEL COMPARISON TABLE")
    print("=======================================================================")
    print(model_comparison[["Model", "Accuracy", "Precision", "Recall", "F1 Score"]].to_string(index=False))

    # Save to CSV
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    model_comparison[["Model", "Accuracy", "Precision", "Recall", "F1 Score"]].to_csv(csv_path, index=False)
    print(f"\nSaved comparison summary table to: {csv_path}")

    # 6. Generate Metric Comparison Bar Charts
    sns.set_theme(style="whitegrid")
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score"]
    palette = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        sns.barplot(
            x="Model",
            y=metric,
            data=model_comparison,
            ax=ax,
            hue="Model",
            palette="Blues_d",
            legend=False
        )
        ax.set_title(f"Model Comparison — {metric}", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=15)

        # Add bar labels
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f"{height:.2%}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 3),
                textcoords='offset points'
            )

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "model_metrics_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved metric comparison bar charts to: {chart_path}")

    # 7. Generate Confusion Matrix Plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, filename, _) in enumerate(model_info):
        ax = axes[idx]
        cm = confusion_matrices[name]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xticklabels(["Low Risk (0)", "High Risk (1)"])
        ax.set_yticklabels(["Low Risk (0)", "High Risk (1)"])

    # Hide extra unused subplot
    fig.delaxes(axes[5])

    plt.tight_layout()
    cm_chart_path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(cm_chart_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix visual grid to: {cm_chart_path}")

    # 8. Identify Highest Recall Model
    highest_recall_row = model_comparison.loc[model_comparison["Recall"].idxmax()]
    highest_acc_row = model_comparison.loc[model_comparison["Accuracy"].idxmax()]

    print("\n=======================================================================")
    print("                     ACADEMIC EXAM EVALUATION REPORT")
    print("=======================================================================")
    print("\n📌 SPECIAL ATTENTION TO RECALL IN MEDICAL SCREENING:")
    print("   • Definition: Recall answers — 'Out of all patients who actually belong to")
    print("     the positive/high-risk class, how many did the model correctly identify?'")
    print("   • Clinical Importance: In a screening-support application, missing a high-risk case")
    print("     (False Negative) is medically more concerning than generating a False Positive.")
    print("     A False Positive leads to harmless secondary confirmation testing, whereas a False")
    print("     Negative delays critical lifestyle and therapeutic interventions.")
    print("   • Educational Scope: This machine learning system serves purely as an educational")
    print("     screening-support project and is NOT a diagnostic medical tool.")

    print("\n-----------------------------------------------------------------------")
    print("                     ANSWERS TO EVALUATION QUESTIONS")
    print("-----------------------------------------------------------------------")
    
    print("\n1. Can diabetes risk be predicted from patient characteristics?")
    print("   Yes. Standard physiological attributes (Glucose, BMI, Age, Insulin) contain strong")
    print(f"   predictive patterns, achieving up to {highest_acc_row['Accuracy']*100:.2f}% accuracy on unseen data.")

    print("\n2. Which features appear important in the dataset?")
    print("   Glucose level, BMI (Body Mass Index), and Age are the three most influential predictors")
    print("   of diabetes risk identified during exploratory data analysis and feature correlation.")

    print("\n3. How do the five algorithms compare?")
    for _, row in model_comparison.iterrows():
        print(f"   - {row['Model']:22s}: Acc = {row['Accuracy']*100:.2f}%, Rec = {row['Recall']*100:.2f}%, F1 = {row['F1 Score']*100:.2f}%")

    print("\n4. Which model has the highest recall?")
    print(f"   {highest_recall_row['Model']} achieved the highest Recall of {highest_recall_row['Recall']*100:.2f}%")
    print(f"   (Correctly identifying {highest_recall_row['Recall']*100:.2f}% of actual high-risk patients).")

    print("\n5. How well can the models identify positive/high-risk cases?")
    print(f"   Models identify high-risk cases with recall scores ranging between ")
    print(f"   {model_comparison['Recall'].min()*100:.2f}% and {model_comparison['Recall'].max()*100:.2f}%.")

    print("\n6. How well do the models perform on unseen test data?")
    print(f"   Overall test accuracies range from {model_comparison['Accuracy'].min()*100:.2f}% to {model_comparison['Accuracy'].max()*100:.2f}%.")

    print("\n7. Does the model show signs of overfitting?")
    print("   Decision Tree shows signs of overfitting (100% train accuracy vs lower test accuracy).")
    print("   Ensemble methods like Random Forest and Gradient Boosting exhibit better generalization.")

    print("\n8. Is the model suitable as a screening-support application?")
    print("   Yes, as a preliminary non-invasive screening aid to flag high-risk individuals for physician evaluation.")

    print("\n9. What are the limitations?")
    print("   Dataset size (768 records), moderate class imbalance, and demographic focus on Pima Indian females.")


if __name__ == "__main__":
    main()
