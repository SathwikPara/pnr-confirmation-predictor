import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
import base64
from io import BytesIO

# ---------- CONFIG ----------
st.set_page_config(page_title="PNR Confirmation Predictor", page_icon="🚆", layout="centered")
import base64

# ---------- CONFIG ----------
st.set_page_config(page_title="PNR Confirmation Predictor", page_icon="🚆", layout="centered")

# ---------- PAGE BACKGROUND ----------
# ===== Background Image Styling with Overlay Fix =====
import base64
from PIL import Image
from io import BytesIO

def set_bg_from_local(img_path):
    with open(img_path, "rb") as f:
        img_bytes = f.read()
        img_base64 = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        
        .main {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: #b30000;
            text-shadow: 0px 0px 2px white;
        }}

        .stTextInput > label, .stMarkdown, .stAlert {{
            font-size: 16px !important;
            color: #111 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_from_local("trainlhb.jpg")



# Call the function early in your code
set_bg_from_local("trainlhb.jpg")

# ---------- FUNCTIONS ----------
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@st.cache_data
def generate_dataset(n=100000):
    np.random.seed(42)
    return pd.DataFrame({
        'PNR_digit_sum': np.random.randint(10, 90, n),
        'Weekday': np.random.randint(0, 7, n),
        'Month': np.random.randint(1, 13, n),
        'Confirmation_Chance': np.random.uniform(50, 100, n)
    })

@st.cache_resource
def train_model(data):
    X = data[['PNR_digit_sum', 'Weekday', 'Month']]
    y = data['Confirmation_Chance']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

def extract_features(pnr_str):
    digits = [int(d) for d in pnr_str]
    return pd.DataFrame([{
        'PNR_digit_sum': sum(digits),
        'Weekday': (digits[2] + digits[4]) % 7,
        'Month': (digits[1] + digits[6]) % 12 + 1,
    }])

# ---------- LOAD DATA & MODEL ----------
df = generate_dataset()
model = train_model(df)

# ---------- LOAD LOGO ----------
logo = Image.open(r"C:\Users\sathwik para\Downloads\ML project\Pnr Prediction\Indian_Railways.png")  # Your logo file
logo_base64 = image_to_base64(logo)

# ---------- HEADER UI ----------
st.markdown(f"""
<div style="
    background-color: #f5f5f5;
    padding: 15px 20px;
    border-radius: 15px;
    border: 1px solid #ccc;
    margin-bottom: 25px;
">
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo_base64}" width="80" style="margin-right: 20px;">
        <div>
            <h1 style="color:#b30000; margin: 0; font-size: 28px;">Indian Railways PNR Confirmation</h1>
            <h2 style="color:#b30000; margin: 0; font-size: 20px;">Chances Predictor</h2>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- PNR INPUT ----------
pnr_input = st.text_input("🔢 Enter your 10-digit PNR number:", max_chars=10)

# ---------- PREDICTION ----------
if pnr_input:
    if not pnr_input.isdigit() or len(pnr_input) != 10:
        st.error("❌ Please enter a valid 10-digit numeric PNR.")
    else:
        with st.spinner("🚆 Predicting your ticket confirmation chance..."):
            features = extract_features(pnr_input)
            prediction = model.predict(features)[0]
            st.success(f"✅ Predicted Confirmation Chance: **{prediction:.2f}%**")

            if prediction >= 80:
                st.markdown("🎉 **High** chance of confirmation. Sit back and relax!")
            elif prediction >= 50:
                st.markdown("⚠️ **Medium** chance of confirmation. Keep checking status.")
            else:
                st.markdown("❌ **Low** chance. Consider alternative plans or Tatkal.")

# ---------- FOOTER ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<small style='color:gray;'>🔒 This is a demo ML app. Predictions are based on mock data and not real-time IRCTC status.</small>",
    unsafe_allow_html=True
)
st.markdown("<p style='text-align:center; color:gray;'>👨‍💻 Made by <b>Sathwik Para</b></p>", unsafe_allow_html=True)
