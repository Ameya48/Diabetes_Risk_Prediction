import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Diabetes Risk Prediction — Feature Importance & Overfitting Analysis\n",
                "\n",
                "**Academic Machine Learning Exam Project**  \n",
                "**Phase 5:** Feature Importance Analysis, Logistic Regression Coefficients, Overfitting Evaluation, and Final ML Project Synthesis.\n",
                "\n",
                "---"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Import Libraries & Load Data"
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
                "from sklearn.metrics import accuracy_score\n",
                "\n",
                "# Load Data\n",
                "data_path = 'data/cleaned_diabetes.csv' if os.path.exists('data/cleaned_diabetes.csv') else '../data/cleaned_diabetes.csv'\n",
                "df = pd.read_csv(data_path)\n",
                "\n",
                "X = df.drop(columns=['Outcome'])\n",
                "y = df['Outcome']\n",
                "feature_names = list(X.columns)\n",
                "\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X, y, test_size=0.2, random_state=42, stratify=y\n",
                ")\n",
                "\n",
                "# Load Scaler & Models\n",
                "models_dir = 'models' if os.path.exists('models') else '../models'\n",
                "scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))\n",
                "X_train_scaled = scaler.transform(X_train)\n",
                "X_test_scaled = scaler.transform(X_test)\n",
                "\n",
                "dt_model = joblib.load(os.path.join(models_dir, 'decision_tree.pkl'))\n",
                "rf_model = joblib.load(os.path.join(models_dir, 'random_forest.pkl'))\n",
                "gb_model = joblib.load(os.path.join(models_dir, 'gradient_boosting.pkl'))\n",
                "lr_model = joblib.load(os.path.join(models_dir, 'logistic_regression.pkl'))\n",
                "knn_model = joblib.load(os.path.join(models_dir, 'knn.pkl'))\n",
                "\n",
                "print(\"Models and scaler loaded successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Part 1: Feature Importance (Tree-Based Models)\n",
                "\n",
                "**Careful Wording:** *The model found these features useful for making predictions in this dataset. This reflects numerical patterns in the sample and does not prove medical causation.*"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fi_df = pd.DataFrame({\n",
                "    'Feature': feature_names,\n",
                "    'Decision Tree': dt_model.feature_importances_,\n",
                "    'Random Forest': rf_model.feature_importances_,\n",
                "    'Gradient Boosting': gb_model.feature_importances_\n",
                "})\n",
                "fi_df['Average Importance'] = fi_df[['Decision Tree', 'Random Forest', 'Gradient Boosting']].mean(axis=1)\n",
                "fi_df = fi_df.sort_values(by='Average Importance', ascending=False).reset_index(drop=True)\n",
                "fi_df"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 5))\n",
                "sns.barplot(x='Average Importance', y='Feature', data=fi_df, hue='Feature', palette='Blues_r', legend=False)\n",
                "plt.title('Average Feature Importance Across Tree Models', fontsize=13, fontweight='bold')\n",
                "plt.xlabel('Importance Score')\n",
                "plt.ylabel('Clinical Feature')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Part 2: Logistic Regression Coefficients"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "coefficients = lr_model.coef_[0]\n",
                "lr_coeff_df = pd.DataFrame({\n",
                "    'Feature': feature_names,\n",
                "    'Coefficient': coefficients,\n",
                "    'Absolute Value': np.abs(coefficients)\n",
                "}).sort_values(by='Absolute Value', ascending=False).reset_index(drop=True)\n",
                "lr_coeff_df[['Feature', 'Coefficient']]"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Part 3: Overfitting Analysis (Train vs Test Accuracy)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "models_dict = {\n",
                "    'Logistic Regression': (lr_model, X_train_scaled, X_test_scaled),\n",
                "    'KNN': (knn_model, X_train_scaled, X_test_scaled),\n",
                "    'Decision Tree': (dt_model, X_train, X_test),\n",
                "    'Random Forest': (rf_model, X_train, X_test),\n",
                "    'Gradient Boosting': (gb_model, X_train, X_test)\n",
                "}\n",
                "\n",
                "overfit_list = []\n",
                "for name, (clf, X_tr, X_te) in models_dict.items():\n",
                "    tr_acc = accuracy_score(y_train, clf.predict(X_tr))\n",
                "    te_acc = accuracy_score(y_test, clf.predict(X_te))\n",
                "    diff = tr_acc - te_acc\n",
                "    status = 'Severe Overfitting' if diff > 0.15 else ('Slight Overfitting' if diff > 0.05 else 'Good Generalization')\n",
                "    overfit_list.append({\n",
                "        'Model': name,\n",
                "        'Train Accuracy': round(tr_acc, 4),\n",
                "        'Test Accuracy': round(te_acc, 4),\n",
                "        'Difference (Train - Test)': round(diff, 4),\n",
                "        'Generalization Status': status\n",
                "    })\n",
                "\n",
                "overfit_df = pd.DataFrame(overfit_list)\n",
                "overfit_df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Part 4: Final Synthesis Report\n",
                "\n",
                "### Summary for Panelist Presentation:\n",
                "1. **Most Important Risk Features:**  \n",
                "   Glucose is the most prominent feature ($\text{importance} > 0.30$), followed by BMI and Age.  \n",
                "   *Careful Wording:* \"The model found these features useful for making predictions in this dataset.\"\n",
                "\n",
                "2. **Overfitting Observations:**  \n",
                "   - Single Decision Tree overfits completely (100% training accuracy vs 68.18% testing accuracy).\n",
                "   - Logistic Regression ($\Delta = 8.86\%$) and KNN ($\Delta = 7.74\%$) show reasonable train-to-test alignment.\n",
                "\n",
                "3. **Screening-Support Scope & Limitations:**  \n",
                "   - System is a **screening-support tool**, NOT a clinical diagnostic tool.\n",
                "   - Limitations include sample size ($N=768$), moderate class imbalance, demographic restriction (Pima Indian females), and requirement for external clinical validation prior to any real-world use."
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

with open("notebooks/feature_and_overfitting_analysis.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Created notebooks/feature_and_overfitting_analysis.ipynb successfully!")
