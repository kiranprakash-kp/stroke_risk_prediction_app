import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('stroke_prediction_logreg.pkl')
scaler = joblib.load('scaler.pkl')

# Title with description
st.markdown(
    "<h1 style='color: black;'>🧠 Stroke Risk Prediction App</h1>"
    "<p>Input patient details to assess the risk of stroke using a machine learning model trained on healthcare data.</p>",
    unsafe_allow_html=True
)

# Sidebar Inputs
st.sidebar.header('📝 Patient Information')

age = st.sidebar.slider('Age', 0, 100, 50)
glucose = st.sidebar.number_input('Average Glucose Level', min_value=50.0, max_value=300.0, value=100.0)
bmi = st.sidebar.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
hypertension = st.sidebar.radio('Hypertension', [0, 1], index=0)
heart_disease = st.sidebar.radio('Heart Disease', [0, 1], index=0)
gender = st.sidebar.selectbox('Gender', ['Female', 'Male', 'Other'])
ever_married = st.sidebar.radio('Ever Married', ['No', 'Yes'])
work_type = st.sidebar.selectbox('Work Type', ['Never_worked', 'Govt_job', 'Private', 'Self-employed', 'children'])
residence_type = st.sidebar.radio('Residence Type', ['Urban', 'Rural'])
smoking_status = st.sidebar.selectbox('Smoking Status', ['Unknown', 'formerly smoked', 'never smoked', 'smokes'])

# Sidebar Disclaimer
st.sidebar.markdown("---")
st.sidebar.info("⚠️ This model provides close stroke risk predictions based on patient data but is not a substitute for professional medical devices or diagnosis. Further improvements could enhance its alignment with clinical standards.")

# Predict button
if st.sidebar.button('🔍 Predict Stroke Risk'):
    # Numerical Features
    comorbidity = hypertension + heart_disease
    interaction = age * glucose
    to_scale = np.array([[age, glucose, bmi, comorbidity, interaction]])
    scaled = scaler.transform(to_scale)[0]  # Scale 5 features
    continuous_scaled = np.concatenate([scaled, np.array([hypertension, heart_disease])])  # Add binary features

    # Gender Encoding
    gender_encoded = [1 if gender == 'Male' else 0, 1 if gender == 'Other' else 0]

    # Ever Married Encoding
    ever_married_encoded = [1 if ever_married == 'Yes' else 0]

    # Work Type Encoding
    work_type_encoded = [0, 0, 0, 0]
    if work_type == 'Never_worked':
        work_type_encoded[0] = 1
    elif work_type == 'Private':
        work_type_encoded[1] = 1
    elif work_type == 'Self-employed':
        work_type_encoded[2] = 1
    elif work_type == 'children':
        work_type_encoded[3] = 1

    # Residence Type Encoding
    residence_encoded = [1 if residence_type == 'Urban' else 0]

    # Smoking Status Encoding
    smoking_encoded = [0, 0, 0]
    if smoking_status == 'formerly smoked':
        smoking_encoded[0] = 1
    elif smoking_status == 'never smoked':
        smoking_encoded[1] = 1
    elif smoking_status == 'smokes':
        smoking_encoded[2] = 1

    # Age Binning
    age_group = [0, 0, 0, 0]
    if 18 <= age < 35:
        age_group[0] = 1
    elif 35 <= age < 50:
        age_group[1] = 1
    elif 50 <= age < 65:
        age_group[2] = 1
    elif age >= 65:
        age_group[3] = 1

    # BMI Binning
    bmi_group = [0, 0, 0]
    if 18.5 <= bmi < 25:
        bmi_group[0] = 1
    elif 25 <= bmi < 30:
        bmi_group[1] = 1
    elif bmi >= 30:
        bmi_group[2] = 1

    # Glucose Binning
    glucose_group = [0, 0, 0, 0]
    if glucose < 100:
        glucose_group[0] = 1
    elif glucose < 125:
        glucose_group[1] = 1
    elif glucose < 200:
        glucose_group[2] = 1
    else:
        glucose_group[3] = 1

    # Final Input Vector (29 features)
    final_input = np.concatenate([
        continuous_scaled, gender_encoded, ever_married_encoded,
        work_type_encoded, residence_encoded, smoking_encoded,
        age_group, bmi_group, glucose_group
    ]).reshape(1, -1)

    # Predict
    prediction = model.predict(final_input)
    probability = model.predict_proba(final_input)[0][1]

    # Output Section
    st.subheader("🔎 Prediction Result:")

    # Color-coded result
    if prediction[0] == 1:
        st.markdown(f"<h3 style='color: red;'>⚠️ High Risk of Stroke Detected!</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: green;'>✅ Low Risk of Stroke</h3>", unsafe_allow_html=True)

    # Probability Bar
    st.progress(probability)
    st.write(f"Probability of Stroke: **{probability:.2%}**")

# Footer (Project Details)
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Stroke Risk Prediction App | Powered by Logistic Regression | Includes feature engineering & scaling</p>",
    unsafe_allow_html=True
)
