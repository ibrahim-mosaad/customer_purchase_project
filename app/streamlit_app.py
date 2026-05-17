import streamlit as st
import joblib
import pandas as pd

# Load model
model, features = joblib.load("models/dt_model.pkl")

st.title("🛒 Customer Purchase Prediction System")

st.sidebar.header("Customer Input")

page_values = st.slider("Page Values", 0.0, 100.0, 10.0)
bounce_rate = st.slider("Bounce Rate", 0.0, 1.0, 0.2)
exit_rate = st.slider("Exit Rate", 0.0, 1.0, 0.3)

input_df = pd.DataFrame([{
    "PageValues": page_values,
    "BounceRates": bounce_rate,
    "ExitRates": exit_rate
}])

# Align features
input_df = input_df.reindex(columns=features, fill_value=0)

if st.button("Predict Purchase"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.success(f"✅ Will Purchase (Confidence: {proba:.2f})")
    else:
        st.error(f"❌ Will NOT Purchase (Confidence: {proba:.2f})")