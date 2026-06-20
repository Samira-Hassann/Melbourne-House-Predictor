import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# Page Config
# ==========================================
st.set_page_config(
    page_title="Melbourne House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# ==========================================
# Load Model
# ==========================================
model = joblib.load("catboost_house_price_pipeline.pkl")

# ==========================================
# Dropdown Lists
# (استبدليها بالقيم الحقيقية من الداتا)
# ==========================================

suburb_list = [
    "Richmond",
    "Brunswick",
    "St Kilda",
    "Footscray"
]

council_list = [
    "Yarra",
    "Moreland",
    "Boroondara",
    "Melbourne"
]

region_list = [
    "Northern Metropolitan",
    "Southern Metropolitan",
    "Western Metropolitan",
    "Eastern Metropolitan"
]

# ==========================================
# Mappings
# ==========================================

type_map = {
    "🏠 House": "h",
    "🏢 Unit": "u",
    "🏘️ Townhouse": "t"
}

method_map = {
    "🤝 Sold": "S",
    "📑 Sold Prior": "SP",
    "💼 Private Sale": "PI",
    "🔨 Vendor Bid": "VB",
    "🏆 Auction": "SA"
}

# ==========================================
# Header
# ==========================================

st.title("🏡 Melbourne House Price Predictor")

st.markdown("""
Predict Melbourne property prices using a tuned **CatBoost Machine Learning Model** 🤖

Fill in the property details below and click **Predict Price**.
""")

# ==========================================
# Form
# ==========================================

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    # =====================
    # Location
    # =====================

    with col1:

        st.subheader("📍 Location Information")

        Suburb = st.selectbox(
            "🏘️ Suburb",
            suburb_list
        )

        CouncilArea = st.selectbox(
            "🏛️ Council Area",
            council_list
        )

        Regionname = st.selectbox(
            "🌏 Region",
            region_list
        )

        Lattitude = st.number_input(
            "📌 Latitude",
            format="%.6f"
        )

        Longtitude = st.number_input(
            "📌 Longitude",
            format="%.6f"
        )

    # =====================
    # Property
    # =====================

    with col2:

        st.subheader("🏠 Property Details")

        Rooms = st.number_input(
            "🛏️ Rooms",
            min_value=1,
            value=3
        )

        Bathroom = st.number_input(
            "🛁 Bathrooms",
            min_value=1,
            value=1
        )

        Car = st.number_input(
            "🚗 Car Spaces",
            min_value=0,
            value=1
        )

        Distance = st.number_input(
            "📏 Distance From CBD (KM)",
            min_value=0.0,
            value=10.0
        )

        Propertycount = st.number_input(
            "🏘️ Property Count",
            min_value=0,
            value=5000
        )

        LogLandsize = st.number_input(
            "🌳 Log Land Size",
            min_value=0.0,
            value=5.0,
            help="Use the log-transformed land size value"
        )

    st.markdown("---")

    st.subheader("💼 Sale Information")

    Type_display = st.selectbox(
        "🏡 Property Type",
        list(type_map.keys())
    )

    Method_display = st.selectbox(
        "🤝 Sale Method",
        list(method_map.keys())
    )

    predict_btn = st.form_submit_button(
        "🚀 Predict Price"
    )

# ==========================================
# Prediction
# ==========================================

if predict_btn:

    input_df = pd.DataFrame([{
        "Suburb": Suburb,
        "Rooms": Rooms,
        "Type": type_map[Type_display],
        "Method": method_map[Method_display],
        "Distance": Distance,
        "Bathroom": Bathroom,
        "Car": Car,
        "CouncilArea": CouncilArea,
        "Lattitude": Lattitude,
        "Longtitude": Longtitude,
        "Regionname": Regionname,
        "Propertycount": Propertycount,
        "LogLandsize": LogLandsize
    }])

    try:

        log_prediction = model.predict(input_df)

        predicted_price = np.expm1(log_prediction)[0]

        st.success("✅ Prediction Generated Successfully")

        st.metric(
            label="💰 Estimated House Price",
            value=f"AUD {predicted_price:,.0f}"
        )

        st.balloons()

        st.info(
            "🤖 Prediction generated using the final tuned CatBoost model."
        )

    except Exception as e:

        st.error(f"❌ Error: {e}")
