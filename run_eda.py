import os
import pandas as pd
import numpy as np

print("Running Phase 2 EDA & Data Cleaning Pipeline...")

# Load dataset
df = pd.read_csv('data/diabetes.csv')
print(f"Original shape: {df.shape}")

# Identify missing zeroes
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\nZero values before cleaning:")
print((df[zero_columns] == 0).sum())

df_cleaned = df.copy()
df_cleaned[zero_columns] = df_cleaned[zero_columns].replace(0, np.nan)

# Median imputation
for col in zero_columns:
    median_val = df_cleaned[col].median()
    df_cleaned[col].fillna(median_val, inplace=True)

print("\nMissing values after median imputation:")
print(df_cleaned.isnull().sum())

# Save cleaned dataset
df_cleaned.to_csv('data/cleaned_diabetes.csv', index=False)
print("\nCleaned dataset successfully written to data/cleaned_diabetes.csv")
