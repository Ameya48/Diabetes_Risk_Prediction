import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Diabetes Risk Prediction — Data Cleaning, Preprocessing & EDA\n",
                "\n",
                "**Academic Machine Learning Exam Project**  \n",
                "**Phase 2:** Data Understanding, Data Cleaning, Missing Value Handling, and Exploratory Data Analysis (EDA).\n",
                "\n",
                "---"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Import Libraries\n",
                "We begin by importing the core Python libraries required for data manipulation and visualization."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Set plot style\n",
                "sns.set_theme(style='whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (8, 5)\n",
                "plt.rcParams['font.size'] = 11\n",
                "\n",
                "print(\"Libraries successfully imported!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Load Dataset\n",
                "Load the raw Pima Indians Diabetes Dataset from the `data` directory."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load raw dataset\n",
                "import os\n",
                "data_path = 'data/diabetes.csv' if os.path.exists('data/diabetes.csv') else '../data/diabetes.csv'\n",
                "df = pd.read_csv(data_path)\n",
                "print(f\"Successfully loaded dataset from: {data_path}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Display First 5 Rows\n",
                "Inspect the first 5 records to understand feature structures and values."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Display Dataset Shape\n",
                "Check the total number of patient records (rows) and attributes (columns)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(f\"Dataset Shape: {df.shape[0]} rows and {df.shape[1]} columns\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Display Column Names\n",
                "List all 9 column feature names present in the dataset."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for i, col in enumerate(df.columns, 1):\n",
                "    print(f\"{i}. {col}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Display Data Types\n",
                "Check the data type of each attribute to ensure compatibility with numerical operations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df.dtypes"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Check Missing Values\n",
                "Verify standard `NaN` / null values across all columns."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df.isnull().sum()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Check Duplicate Rows\n",
                "Check if there are any duplicate patient records."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "duplicate_count = df.duplicated().sum()\n",
                "print(f\"Number of duplicate rows in dataset: {duplicate_count}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Generate Basic Statistical Summary\n",
                "Examine descriptive statistics (mean, std, min, median, max) for numerical columns."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df.describe().T"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Analyze Target Variable (`Outcome`)\n",
                "Understand the class distribution of the target variable `Outcome` (0 = Low Risk, 1 = High Risk)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "outcome_counts = df['Outcome'].value_counts()\n",
                "outcome_percentages = df['Outcome'].value_counts(normalize=True) * 100\n",
                "\n",
                "summary_df = pd.DataFrame({\n",
                "    'Count': outcome_counts,\n",
                "    'Percentage (%)': outcome_percentages.round(2)\n",
                "})\n",
                "summary_df.index = ['Low Risk / No Diabetes (0)', 'High Risk / Diabetes (1)']\n",
                "summary_df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. Data Cleaning (Handling Biologically Zero Values)\n",
                "\n",
                "### ⚠️ Explanation of Medical Zero Values:\n",
                "In clinical datasets, certain physiological measurements cannot biologically be zero in a living patient:\n",
                "- **Glucose**: A glucose level of 0 mg/dL is medically impossible.\n",
                "- **BloodPressure**: A diastolic pressure of 0 mm Hg indicates absence of circulation.\n",
                "- **SkinThickness**: Triceps skin fold thickness of 0 mm is biologically invalid.\n",
                "- **Insulin**: Serum insulin level of 0 mu U/ml signifies unrecorded data.\n",
                "- **BMI**: Body Mass Index of 0 kg/m² is physically impossible.\n",
                "\n",
                "These zeroes represent **missing values** encoded as 0. We handle them by:\n",
                "1. Replacing 0 with `np.nan` in these specific clinical columns.\n",
                "2. Imputing `NaN` values using column **median** values (a robust measure that is resilient to outliers)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Columns where 0 represents missing clinical data\n",
                "zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']\n",
                "\n",
                "print(\"Count of invalid zero values per column before cleaning:\")\n",
                "print((df[zero_columns] == 0).sum())\n",
                "\n",
                "# Create a copy of the dataframe for cleaning\n",
                "df_cleaned = df.copy()\n",
                "\n",
                "# Replace 0 with NaN in physiological columns\n",
                "df_cleaned[zero_columns] = df_cleaned[zero_columns].replace(0, np.nan)\n",
                "\n",
                "print(\"\\nMissing (NaN) values after replacing zeroes:\")\n",
                "print(df_cleaned.isnull().sum())\n",
                "\n",
                "# Impute NaN values with median of each column\n",
                "for col in zero_columns:\n",
                "    median_val = df_cleaned[col].median()\n",
                "    df_cleaned[col] = df_cleaned[col].fillna(median_val)\n",
                "\n",
                "print(\"\\nMissing (NaN) values after Median Imputation:\")\n",
                "print(df_cleaned.isnull().sum())"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Save the cleaned dataset to CSV\n",
                "output_dir = 'data' if os.path.exists('data') else '../data'\n",
                "cleaned_csv_path = os.path.join(output_dir, 'cleaned_diabetes.csv')\n",
                "df_cleaned.to_csv(cleaned_csv_path, index=False)\n",
                "print(f\"Cleaned dataset saved successfully to: {cleaned_csv_path}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 12. Exploratory Data Analysis (EDA Visualizations)\n",
                "We now generate clean, clear visualizations to explore distributions and target relationships."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Graph 1: Target Outcome Distribution"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(7, 4.5))\n",
                "ax = sns.countplot(x='Outcome', data=df_cleaned, palette=['#2ecc71', '#e74c3c'])\n",
                "plt.title('Target Variable Distribution (Outcome)', fontsize=14, fontweight='bold')\n",
                "plt.xlabel('Outcome (0 = Low Risk, 1 = High Risk)', fontsize=11)\n",
                "plt.ylabel('Patient Count', fontsize=11)\n",
                "plt.xticks([0, 1], ['Low Risk (0)', 'High Risk (1)'])\n",
                "\n",
                "# Add count annotations above bars\n",
                "for p in ax.patches:\n",
                "    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),\n",
                "                ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 3),\n",
                "                textcoords='offset points')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Explanation for Panelist:**  \n",
                "This count plot displays the distribution of the target variable `Outcome`. Out of 768 total patients, 500 patients (~65.1%) are labeled as Low Risk (`Outcome = 0`) and 268 patients (~34.9%) are labeled as High Risk (`Outcome = 1`). This shows the dataset has moderate class imbalance."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Graph 2: Age Distribution"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(8, 4.5))\n",
                "sns.histplot(df_cleaned['Age'], kde=True, color='#3498db', bins=25)\n",
                "plt.title('Patient Age Distribution', fontsize=14, fontweight='bold')\n",
                "plt.xlabel('Age (Years)', fontsize=11)\n",
                "plt.ylabel('Frequency', fontsize=11)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Explanation for Panelist:**  \n",
                "The Age distribution is right-skewed, indicating that the majority of patients in this dataset are young adults aged between 20 and 30 years old, with fewer older patients."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Graph 3: Glucose Level Distribution"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(8, 4.5))\n",
                "sns.histplot(df_cleaned['Glucose'], kde=True, color='#9b59b6', bins=25)\n",
                "plt.title('Glucose Concentration Distribution (Post Imputation)', fontsize=14, fontweight='bold')\n",
                "plt.xlabel('Glucose Level (mg/dL)', fontsize=11)\n",
                "plt.ylabel('Frequency', fontsize=11)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Explanation for Panelist:**  \n",
                "After replacing non-physiological zero values with the median glucose level, Glucose displays an approximately normal distribution centered around 120 mg/dL."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Graph 4: BMI Distribution"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(8, 4.5))\n",
                "sns.histplot(df_cleaned['BMI'], kde=True, color='#e67e22', bins=25)\n",
                "plt.title('Body Mass Index (BMI) Distribution', fontsize=14, fontweight='bold')\n",
                "plt.xlabel('BMI (kg/m²)', fontsize=11)\n",
                "plt.ylabel('Frequency', fontsize=11)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Explanation for Panelist:**  \n",
                "The BMI distribution follows a bell-shaped normal curve peaking around 32 kg/m². Most patients in this study fall into the overweight or clinical obesity range (>25 kg/m²)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Graph 5: BloodPressure Distribution"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(8, 4.5))\n",
                "sns.histplot(df_cleaned['BloodPressure'], kde=True, color='#1abc9c', bins=25)\n",
                "plt.title('Diastolic Blood Pressure Distribution', fontsize=14, fontweight='bold')\n",
                "plt.xlabel('Blood Pressure (mm Hg)', fontsize=11)\n",
                "plt.ylabel('Frequency', fontsize=11)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Explanation for Panelist:**  \n",
                "Diastolic Blood Pressure shows a tight bell-shaped distribution centered around 72 mm Hg, which corresponds to normal diastolic resting pressure."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Graph 6: Correlation Heatmap"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 7))\n",
                "correlation_matrix = df_cleaned.corr()\n",
                "sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='Blues', linewidths=0.5)\n",
                "plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Explanation for Panelist:**  \n",
                "The correlation heatmap measures linear associations between features and the target `Outcome`:\n",
                "- **Glucose** shows the strongest positive correlation with `Outcome` (~0.49).\n",
                "- **BMI** (~0.31) and **Age** (~0.24) also show moderate positive correlations with `Outcome`.\n",
                "- **Pregnancies** correlates strongly with **Age** (~0.54), as expected."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 13. Feature Relationships with Diabetes Outcome\n",
                "Now we analyze how key physiological measurements differ between Low Risk and High Risk patient groups."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Graph 7: Glucose Level vs Outcome"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(7, 4.5))\n",
                "sns.boxplot(x='Outcome', y='Glucose', data=df_cleaned, palette=['#2ecc71', '#e74c3c'])\n",
                "plt.title('Glucose Level Comparison by Outcome', fontsize=14, fontweight='bold')\n",
                "plt.xlabel('Outcome (0 = Low Risk, 1 = High Risk)', fontsize=11)\n",
                "plt.ylabel('Glucose Level (mg/dL)', fontsize=11)\n",
                "plt.xticks([0, 1], ['Low Risk (0)', 'High Risk (1)'])\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Explanation for Panelist:**  \n",
                "Patients in the High Risk group (`Outcome = 1`) exhibit a noticeably higher median glucose level (~140 mg/dL) compared to the Low Risk group (~107 mg/dL). In this dataset, higher glucose concentrations are strongly associated with positive diabetes outcome labels."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Graph 8: BMI vs Outcome"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(7, 4.5))\n",
                "sns.boxplot(x='Outcome', y='BMI', data=df_cleaned, palette=['#2ecc71', '#e74c3c'])\n",
                "plt.title('BMI Comparison by Outcome', fontsize=14, fontweight='bold')\n",
                "plt.xlabel('Outcome (0 = Low Risk, 1 = High Risk)', fontsize=11)\n",
                "plt.ylabel('BMI (kg/m²)', fontsize=11)\n",
                "plt.xticks([0, 1], ['Low Risk (0)', 'High Risk (1)'])\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Explanation for Panelist:**  \n",
                "Patients labeled as High Risk (`Outcome = 1`) tend to have higher median BMI values (~34.2 kg/m²) than patients in the Low Risk group (~30.1 kg/m²). Higher BMI values are associated with a greater proportion of positive diabetes outcomes in this dataset."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Graph 9: Age vs Outcome"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(7, 4.5))\n",
                "sns.boxplot(x='Outcome', y='Age', data=df_cleaned, palette=['#2ecc71', '#e74c3c'])\n",
                "plt.title('Age Comparison by Outcome', fontsize=14, fontweight='bold')\n",
                "plt.xlabel('Outcome (0 = Low Risk, 1 = High Risk)', fontsize=11)\n",
                "plt.ylabel('Age (Years)', fontsize=11)\n",
                "plt.xticks([0, 1], ['Low Risk (0)', 'High Risk (1)'])\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Explanation for Panelist:**  \n",
                "The median age for High Risk patients (~36 years) is higher than for Low Risk patients (~27 years). In this dataset, older age is associated with a higher proportion of positive diabetes outcomes."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 14. Important Observations & Academic Summary\n",
                "\n",
                "### Key Observations for Exam Presentation:\n",
                "1. **Primary Risk Feature (Glucose):**  \n",
                "   `Glucose` is the single most prominent predictor feature. Higher glucose values are strongly associated with a greater proportion of positive diabetes outcomes (`Outcome = 1`) in this dataset.\n",
                "\n",
                "2. **Secondary Risk Features (BMI & Age):**  \n",
                "   Both `BMI` and `Age` show significant positive associations with diabetes risk. Higher BMI and older age correlate with higher risk categories in this patient population.\n",
                "\n",
                "3. **Biologically Invalid Zero Values Cleaned:**  \n",
                "   Physiological zeroes in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` were successfully identified as missing data and replaced using median imputation.\n",
                "\n",
                "4. **Dataset Exported for Model Training:**  \n",
                "   The clean dataset has been saved to `data/cleaned_diabetes.csv` and is ready for Phase 3 Model Training."
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

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/diabetes_analysis.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Updated notebooks/diabetes_analysis.ipynb successfully!")
