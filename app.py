# ==============================================================================
# Project: Diabetes Risk Prediction Using Machine Learning
# File: app.py
# Description: Streamlit Web Application for Diabetes Risk Prediction.
# ==============================================================================

import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for visual excellence
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F9FAFB;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #E5E7EB;
        margin-bottom: 15px;
    }
    .result-box-high {
        background-color: #FEE2E2;
        border-left: 6px solid #EF4444;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }
    .result-box-low {
        background-color: #D1FAE5;
        border-left: 6px solid #10B981;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# APP HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🩺 Diabetes Risk Prediction Using Machine Learning</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">This application is an educational screening-support demonstration using machine learning. '
    'It is not a medical diagnosis tool.</div>',
    unsafe_allow_html=True
)

st.write("---")

# -----------------------------------------------------------------------------
# SIDEBAR — MODEL SELECTION & INFO
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select Machine Learning Model:",
    [
        "Random Forest",
        "Gradient Boosting",
        "K-Nearest Neighbors (KNN)",
        "Logistic Regression",
        "Decision Tree"
    ]
)

st.sidebar.write("---")
st.sidebar.subheader("📌 Project Information")
st.sidebar.info("""
**Dataset:** Pima Indians Diabetes Dataset  
**Target:** Binary Risk Classification  
• `0`: Low Diabetes Risk  
• `1`: High Diabetes Risk  

**Educational Scope:**  
Designed for academic demonstration and screening support.
""")

st.sidebar.write("---")
st.sidebar.caption("B.Tech CSE Machine Learning Project")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS TO LOAD MODELS & PREPROCESS INPUT
# -----------------------------------------------------------------------------
MODELS_DIR = "models" if os.path.exists("models") else "saved_models"

MODEL_FILE_MAPPING = {
    "Logistic Regression": ("logistic_regression.pkl", True),
    "K-Nearest Neighbors (KNN)": ("knn.pkl", True),
    "Decision Tree": ("decision_tree.pkl", False),
    "Random Forest": ("random_forest.pkl", False),
    "Gradient Boosting": ("gradient_boosting.pkl", False)
}

# Default dataset medians used during training imputation
MEDIANS = {
    'Glucose': 121.0,
    'BloodPressure': 72.0,
    'SkinThickness': 29.0,
    'Insulin': 125.0,
    'BMI': 32.3
}


@st.cache_resource
def load_scaler():
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None


@st.cache_resource
def load_model(filename):
    model_path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


# -----------------------------------------------------------------------------
# MAIN CONTENT — PATIENT MEDICAL INPUT FORM
# -----------------------------------------------------------------------------
st.subheader("📋 Enter Patient Clinical Details")
st.write("Please fill in the medical diagnostic parameters below:")

# Form layout in two clean columns
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies (Number of times pregnant)",
        min_value=0, max_value=20, value=1, step=1,
        help="Number of pregnancies."
    )
    
    glucose = st.number_input(
        "Glucose Level (mg/dL)",
        min_value=0.0, max_value=300.0, value=120.0, step=1.0,
        help="Plasma glucose concentration after a 2-hour oral glucose tolerance test."
    )
    
    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg)",
        min_value=0.0, max_value=200.0, value=70.0, step=1.0,
        help="Diastolic blood pressure."
    )
    
    skin_thickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0.0, max_value=100.0, value=20.0, step=1.0,
        help="Triceps skin fold thickness."
    )

with col2:
    insulin = st.number_input(
        "Insulin Level (mu U/ml)",
        min_value=0.0, max_value=900.0, value=79.0, step=1.0,
        help="2-Hour serum insulin level."
    )
    
    bmi = st.number_input(
        "BMI — Body Mass Index (kg/m²)",
        min_value=0.0, max_value=70.0, value=25.4, step=0.1,
        help="Body mass index = weight in kg / (height in m)²."
    )
    
    pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0, max_value=3.0, value=0.47, step=0.01,
        help="Genetic score reflecting diabetes family history."
    )
    
    age = st.number_input(
        "Age (Years)",
        min_value=1, max_value=120, value=33, step=1,
        help="Patient age in years."
    )

st.write("")

# -----------------------------------------------------------------------------
# PREDICTION ACTION & LOGIC
# -----------------------------------------------------------------------------
if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):
    
    # 1. INPUT VALIDATION
    errors = []
    if pregnancies < 0:
        errors.append("Pregnancies cannot be negative.")
    if glucose <= 0:
        errors.append("Glucose level must be greater than 0 mg/dL.")
    if bmi <= 0:
        errors.append("BMI must be greater than 0 kg/m².")
    if age <= 0 or age > 120:
        errors.append("Please enter a valid age between 1 and 120 years.")
    if blood_pressure < 0 or skin_thickness < 0 or insulin < 0 or pedigree < 0:
        errors.append("Clinical parameters cannot be negative values.")

    if errors:
        for err in errors:
            st.error(f"⚠️ Validation Error: {err}")
    else:
        # 2. PREPARE FEATURE DATAFRAME IN EXACT TRAINED COLUMN ORDER
        raw_input = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': pedigree,
            'Age': age
        }
        
        input_df = pd.DataFrame([raw_input])

        # 3. APPLY SAME MEDIAN IMPUTATION FOR ZERO VALUES IF ENTERED
        cleaned_input_df = input_df.copy()
        for col, med_val in MEDIANS.items():
            if cleaned_input_df[col].iloc[0] == 0:
                cleaned_input_df[col] = med_val

        # 4. LOAD SELECTED MODEL & SCALER
        filename, requires_scaling = MODEL_FILE_MAPPING[model_choice]
        model = load_model(filename)
        scaler = load_scaler()

        if model is None:
            st.error(f"Error: Trained model file '{filename}' not found in '{MODELS_DIR}/'. Please run train_models.py first.")
        else:
            # Scale features if required by model (Logistic Regression & KNN)
            if requires_scaling:
                if scaler is not None:
                    final_input = scaler.transform(cleaned_input_df)
                else:
                    st.error("Error: Scaler file 'scaler.pkl' not found.")
                    final_input = cleaned_input_df
            else:
                final_input = cleaned_input_df

            # 5. GENERATE PREDICTION & PROBABILITIES
            prediction = model.predict(final_input)[0]
            
            prob_high_risk = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(final_input)[0]
                prob_high_risk = probabilities[1] * 100
                prob_low_risk = probabilities[0] * 100

            # 6. DISPLAY RESULTS
            st.write("---")
            st.subheader("🎯 Prediction Output")

            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                st.metric(label="Selected Model", value=model_choice)

            with res_col2:
                if prediction == 1:
                    st.metric(label="Predicted Risk Level", value="HIGH RISK ⚠️")
                else:
                    st.metric(label="Predicted Risk Level", value="LOW RISK ✅")

            with res_col3:
                if prob_high_risk is not None:
                    confidence = prob_high_risk if prediction == 1 else prob_low_risk
                    st.metric(label="Prediction Confidence", value=f"{confidence:.1f}%")
                else:
                    st.metric(label="Prediction Confidence", value="N/A")

            st.write("")

            # Visual Alert Card
            if prediction == 1:
                st.warning(f"""
                ### ⚠️ Diabetes Risk: High
                **Selected Model:** {model_choice}  
                **Confidence Probability:** {prob_high_risk:.1f}% High Risk Probability  
                
                The machine learning model classifies these clinical characteristics in the **High Diabetes Risk** category.
                """)
            else:
                st.success(f"""
                ### ✅ Diabetes Risk: Low
                **Selected Model:** {model_choice}  
                **Confidence Probability:** {prob_low_risk:.1f}% Low Risk Probability  
                
                The machine learning model classifies these clinical characteristics in the **Low Diabetes Risk** category.
                """)

            # Medical Disclaimer Note
            st.info("ℹ️ **Medical Disclaimer:** Please consult a qualified healthcare professional for medical evaluation.")

# -----------------------------------------------------------------------------
# FOOTER & DISCLAIMER
# -----------------------------------------------------------------------------
st.write("---")
st.caption("""
**Disclaimer & Project Scope:**  
This application was developed for educational and academic screening-support demonstration purposes as part of a B.Tech Computer Science exam project.  
It is **not** a clinical diagnostic tool. All predictions are generated using statistical patterns trained on the Pima Indians Diabetes Dataset.
""")
