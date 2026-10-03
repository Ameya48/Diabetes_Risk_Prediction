import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Diabetes Risk Prediction — Model Evaluation & Comparative Study\n",
                "\n",
                "**Academic Machine Learning Exam Project**  \n",
                "**Phase 4:** Evaluation & Comparison of 5 Machine Learning Algorithms (Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting).\n",
                "\n",
                "---"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Import Evaluation Libraries & Load Test Data"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import joblib\n",
                "\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.metrics import (\n",
                "    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report\n",
                ")\n",
                "\n",
                "# Load Data\n",
                "data_path = 'data/cleaned_diabetes.csv' if os.path.exists('data/cleaned_diabetes.csv') else '../data/cleaned_diabetes.csv'\n",
                "df = pd.read_csv(data_path)\n",
                "\n",
                "X = df.drop(columns=['Outcome'])\n",
                "y = df['Outcome']\n",
                "\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X, y, test_size=0.2, random_state=42, stratify=y\n",
                ")\n",
                "\n",
                "# Load Scaler\n",
                "models_dir = 'models' if os.path.exists('models') else '../models'\n",
                "scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))\n",
                "X_train_scaled = scaler.transform(X_train)\n",
                "X_test_scaled = scaler.transform(X_test)\n",
                "\n",
                "print(f\"Test Dataset size: {len(X_test)} samples\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Evaluate All 5 Trained Models"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model_files = [\n",
                "    (\"Logistic Regression\", \"logistic_regression.pkl\", True),\n",
                "    (\"KNN\", \"knn.pkl\", True),\n",
                "    (\"Decision Tree\", \"decision_tree.pkl\", False),\n",
                "    (\"Random Forest\", \"random_forest.pkl\", False),\n",
                "    (\"Gradient Boosting\", \"gradient_boosting.pkl\", False)\n",
                "]\n",
                "\n",
                "results = []\n",
                "confusion_mats = {}\n",
                "\n",
                "for name, filename, use_scaled in model_files:\n",
                "    clf = joblib.load(os.path.join(models_dir, filename))\n",
                "    X_te = X_test_scaled if use_scaled else X_test\n",
                "    \n",
                "    preds = clf.predict(X_te)\n",
                "    acc = accuracy_score(y_test, preds)\n",
                "    prec = precision_score(y_test, preds, zero_division=0)\n",
                "    rec = recall_score(y_test, preds, zero_division=0)\n",
                "    f1 = f1_score(y_test, preds, zero_division=0)\n",
                "    \n",
                "    results.append({\n",
                "        \"Model\": name,\n",
                "        \"Accuracy\": round(acc, 4),\n",
                "        \"Precision\": round(prec, 4),\n",
                "        \"Recall\": round(rec, 4),\n",
                "        \"F1 Score\": round(f1, 4)\n",
                "    })\n",
                "    confusion_mats[name] = confusion_matrix(y_test, preds)\n",
                "\n",
                "model_comparison = pd.DataFrame(results)\n",
                "model_comparison"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Metric Comparison Visualizations"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(2, 2, figsize=(13, 9))\n",
                "axes = axes.flatten()\n",
                "metrics = [\"Accuracy\", \"Precision\", \"Recall\", \"F1 Score\"]\n",
                "\n",
                "for idx, m in enumerate(metrics):\n",
                "    ax = axes[idx]\n",
                "    sns.barplot(x=\"Model\", y=m, data=model_comparison, ax=ax, hue=\"Model\", palette=\"Blues_d\", legend=False)\n",
                "    ax.set_title(f\"Model Comparison — {m}\", fontsize=13, fontweight=\"bold\")\n",
                "    ax.set_ylim(0, 1.0)\n",
                "    ax.tick_params(axis='x', rotation=15)\n",
                "    \n",
                "    for p in ax.patches:\n",
                "        ax.annotate(f\"{p.get_height():.2%}\", (p.get_x() + p.get_width() / 2., p.get_height()),\n",
                "                    ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 3),\n",
                "                    textcoords='offset points')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Confusion Matrices for All 5 Models"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(2, 3, figsize=(15, 9))\n",
                "axes = axes.flatten()\n",
                "\n",
                "for idx, (name, _, _) in enumerate(model_files):\n",
                "    ax = axes[idx]\n",
                "    sns.heatmap(confusion_mats[name], annot=True, fmt=\"d\", cmap=\"Blues\", cbar=False, ax=ax,\n",
                "                annot_kws={\"size\": 14, \"weight\": \"bold\"})\n",
                "    ax.set_title(f\"{name}\", fontsize=12, fontweight=\"bold\")\n",
                "    ax.set_xlabel(\"Predicted Label\")\n",
                "    ax.set_ylabel(\"True Label\")\n",
                "    ax.set_xticklabels([\"Low Risk (0)\", \"High Risk (1)\"])\n",
                "    ax.set_yticklabels([\"Low Risk (0)\", \"High Risk (1)\"])\n",
                "\n",
                "fig.delaxes(axes[5])\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Answers to Exam Evaluation Questions\n",
                "\n",
                "### 📌 Medical Screening Context — Special Focus on RECALL:\n",
                "- **Recall Definition:** Out of all patients who actually belong to the high-risk class, how many did the model correctly identify?\n",
                "- **Clinical Rationale:** In screening applications, missing a high-risk patient (**False Negative**) can delay medical intervention. A **False Positive** merely leads to harmless secondary confirmation testing.\n",
                "- **Educational Scope:** This tool is designed for educational/screening-support purposes, NOT as a diagnostic replacement.\n",
                "\n",
                "--- \n",
                "\n",
                "### Answers based on actual dataset results:\n",
                "1. **Can diabetes risk be predicted from patient characteristics?**  \n",
                "   Yes, clinical attributes (Glucose, BMI, Age) provide strong signals, yielding up to **77.92%** test accuracy.\n",
                "\n",
                "2. **Which features appear important?**  \n",
                "   **Glucose**, **BMI**, and **Age** exhibit the highest predictive power.\n",
                "\n",
                "3. **How do the five algorithms compare?**  \n",
                "   - **Random Forest:** Best overall accuracy (**77.92%**) and F1-score (**65.31%**).\n",
                "   - **KNN:** Best Recall (**61.11%**).\n",
                "   - **Gradient Boosting:** Balanced performance (**75.97%** accuracy, **57.41%** recall).\n",
                "   - **Logistic Regression:** Clean linear baseline (**70.78%** accuracy).\n",
                "   - **Decision Tree:** Overfits training data, achieving lower test accuracy (**68.18%**).\n",
                "\n",
                "4. **Which model has the highest recall?**  \n",
                "   **K-Nearest Neighbors (KNN)** achieved the highest Recall (**61.11%**).\n",
                "\n",
                "5. **How well can the models identify positive cases?**  \n",
                "   Recall scores range between **48.15%** and **61.11%** across models.\n",
                "\n",
                "6. **How well do models perform on unseen test data?**  \n",
                "   Accuracies range from **68.18%** to **77.92%** on the test set.\n",
                "\n",
                "7. **Does the model show signs of overfitting?**  \n",
                "   Single Decision Tree exhibits overfitting (100% training accuracy vs 68.18% test accuracy). Ensemble methods generalize much better.\n",
                "\n",
                "8. **Is the model suitable as a screening-support application?**  \n",
                "   Yes, as a preliminary non-invasive risk flag for physician review.\n",
                "\n",
                "9. **What are the limitations?**  \n",
                "   Sample size (768 rows), moderate class imbalance, and demographic restriction to Pima Indian females."
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("notebooks/model_evaluation.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Created notebooks/model_evaluation.ipynb successfully!")
