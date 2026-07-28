import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Background */
.stApp{
background: linear-gradient(135deg,#eef2ff,#dbeafe,#ecfeff);
}

/* Hide Streamlit Menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main Title */
.title{
    text-align:center;
    font-size:46px;
    font-weight:700;
    color:#000000 !important;
    margin-bottom:8px;
    text-shadow:1px 1px 2px rgba(0,0,0,0.15);
}

/* Subtitle */
.subtitle{
    text-align:center;
    font-size:18px;
    color:#333333 !important;
    font-weight:500;
    margin-bottom:30px;
}
.subtitle{
text-align:center;
font-size:18px;
color:#475569;
margin-bottom:30px;
}

/* Card */
.card{
background:white;
padding:30px;
border-radius:18px;
box-shadow:0px 10px 25px rgba(0,0,0,0.08);
margin-bottom:20px;
}

/* Prediction Result */
.success-box{
background:#dcfce7;
padding:20px;
border-radius:12px;
font-size:22px;
font-weight:600;
color:#166534;
text-align:center;
}

.danger-box{
background:#fee2e2;
padding:20px;
border-radius:12px;
font-size:22px;
font-weight:600;
color:#991b1b;
text-align:center;
}

/* Sidebar */
section[data-testid="stSidebar"]{
background:#0f172a;
color:white;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p{
color:white;
}

/* Button */
.stButton>button{
background:linear-gradient(90deg,#2563eb,#06b6d4);
color:white;
font-size:18px;
font-weight:bold;
border:none;
border-radius:12px;
padding:12px 35px;
transition:0.3s;
width:100%;
}

.stButton>button:hover{
background:linear-gradient(90deg,#1d4ed8,#0891b2);
transform:scale(1.02);
}

/* Inputs */
.stSelectbox, .stNumberInput{
border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🩺 Stroke Prediction")

st.sidebar.info("""
### About

This AI application predicts the probability of stroke based on patient health parameters.

**Machine Learning Model**
- Random Forest Classifier

**Developer**
Your Name
""")

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="title">🩺 Stroke Prediction System</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">Early Stroke Risk Detection using Machine Learning</div>', unsafe_allow_html=True)

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male","Female","Other"]
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0,1]
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0,1]
    )

    ever_married = st.selectbox(
        "Ever Married",
        ["Yes","No"]
    )

with col2:

    work_type = st.selectbox(
        "Work Type",
        ["Private","Self-employed","Govt_job","children","Never_worked"]
    )

    residence = st.selectbox(
        "Residence Type",
        ["Urban","Rural"]
    )

    glucose = st.number_input(
        "Average Glucose Level",
        value=100.0
    )

    bmi = st.number_input(
        "BMI",
        value=25.0
    )

    smoking = st.selectbox(
        "Smoking Status",
        ["formerly smoked","never smoked","smokes","Unknown"]
    )

# -----------------------------
# Encoding
# -----------------------------
gender_map={
    "Female":0,
    "Male":1,
    "Other":2
}

married_map={
    "No":0,
    "Yes":1
}

work_map={
    "Govt_job":0,
    "Never_worked":1,
    "Private":2,
    "Self-employed":3,
    "children":4
}

residence_map={
    "Rural":0,
    "Urban":1
}

smoking_map={
    "Unknown":0,
    "formerly smoked":1,
    "never smoked":2,
    "smokes":3
}

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Stroke Risk"):

    features=np.array([[
        gender_map[gender],
        age,
        hypertension,
        heart_disease,
        married_map[ever_married],
        work_map[work_type],
        residence_map[residence],
        glucose,
        bmi,
        smoking_map[smoking]
    ]])

    features=scaler.transform(features)

    prediction=model.predict(features)[0]

    probability=model.predict_proba(features)[0][1]

    st.write("")

    if prediction==1:

        st.markdown(f"""
        <div class="danger-box">
        ⚠️ High Risk of Stroke<br><br>
        Probability : {probability*100:.2f}%
        </div>
        """,unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="success-box">
        ✅ Low Risk of Stroke<br><br>
        Probability : {(1-probability)*100:.2f}%
        </div>
        """,unsafe_allow_html=True)

st.write("")
st.write("---")
st.caption("© 2026 Stroke Prediction System | Powered by Machine Learning & Streamlit")