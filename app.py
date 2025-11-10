import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

st.set_page_config(page_title="Population Density Predictor", page_icon="🌍")

st.title("🌍 Population Density Predictor")
st.write("Predict whether a country is **densely populated** or not based on population data, land area, and other factors.")

# === Load model ===
with st.spinner("🔄 Loading model..."):
    # Load the trained model with the correct filename
    try:
        model = joblib.load("random_forest_classifier_model.joblib")
        st.success("✅ Model loaded successfully!")
    except FileNotFoundError:
        st.error("❌ Model file (random_forest_classifier_model.joblib) not found. Please ensure the file is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ An error occurred while loading the model: {e}")
        st.stop()


# === Input data from user ===
st.header("Enter Country Data")

# Organize inputs into columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    population = st.number_input("Population (2020)", min_value=0, value=10000000)
    land_area = st.number_input("Land Area (Km²)", min_value=0, value=50000)
    net_change = st.number_input("Net Change", value=10000)

with col2:
    yearly_change = st.slider("Yearly Change (%)", min_value=-5.0, max_value=10.0, value=0.5, step=0.1, format="%.2f")
    migrants_net = st.number_input("Migrants (net)", value=0)
    fert_rate = st.slider("Fert. Rate", min_value=0.0, max_value=8.0, value=2.0, step=0.1, format="%.1f")

with col3:
    med_age = st.slider("Med. Age", min_value=10.0, max_value=60.0, value=30.0, step=0.1, format="%.1f")
    urban_pop_percent = st.slider("Urban Pop (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1, format="%.1f")
    world_share = st.slider("World Share (%)", min_value=0.0, max_value=20.0, value=0.1, step=0.1, format="%.2f")


if st.button("🚀 Predict Now!"):
    with st.spinner("🤖 Analyzing data... Please wait..."):
        time.sleep(1) # Add a small delay for dramatic effect
        
        # --- Data Preprocessing ---
        # Convert percentages to decimals (as they were likely trained)
        yearly_change_processed = yearly_change / 100.0
        urban_pop_percent_processed = urban_pop_percent / 100.0
        world_share_processed = world_share / 100.0
        
        # Create DataFrame with all features in the correct order
        input_data = pd.DataFrame({
            'Population (2020)': [population],
            'Land Area (Km²)': [land_area],
            'Yearly Change': [yearly_change_processed],
            'Net Change': [net_change],
            'Migrants (net)': [migrants_net],
            'Fert. Rate': [fert_rate],
            'Med. Age': [med_age],
            'Urban Pop %': [urban_pop_percent_processed],
            'World Share': [world_share_processed]
        })

        try:
            # Make prediction using the trained model
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]

            # Interpret the prediction (0 for not crowded, 1 for crowded)
            if prediction == 1:
                st.markdown("<h2 style='text-align: center; color: #FF4B4B;'>🌆 This country is predicted to be DENSELY POPULATED.</h2>", unsafe_allow_html=True)
                confidence = prediction_proba[1]
            else:
                st.markdown("<h2 style='text-align: center; color: #09AB3B;'>🌄 This country is predicted to be NOT DENSELY POPULATED.</h2>", unsafe_allow_html=True)
                confidence = prediction_proba[0]

            # Center the metric
            _, col_metric, _ = st.columns([1, 1, 1])
            with col_metric:
                st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
            
            # Fun interaction!
            st.balloons()

            st.write("---")
            st.write("🔍 Data sent to the model (after processing):")
            st.dataframe(input_data)

        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")

st.markdown("---")
st.caption("MLOps Midterm - Felix Desandy.")
