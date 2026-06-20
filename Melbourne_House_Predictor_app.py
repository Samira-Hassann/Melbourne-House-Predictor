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
# Lists
# ==========================================

suburb_list = [
    # الصقي هنا قائمة الأحياء كاملة
    "Abbotsford",
    "Aberfeldie",
    "Airport West",
    "Albanvale",
    "Albert Park",
    # .....
    "Yarra Glen",
    "Yarraville"
]

council_list = [
    'Yarra',
    'Moonee Valley',
    'Port Phillip',
    'Darebin',
    'Hobsons Bay',
    'Stonnington',
    'Boroondara',
    'Monash',
    'Glen Eira',
    'Whitehorse',
    'Maribyrnong',
    'Bayside',
    'Moreland',
    'Manningham',
    'Banyule',
    'Melbourne',
    'Kingston',
    'Brimbank',
    'Hume',
    'Knox',
    'Maroondah',
    'Casey',
    'Melton',
    'Greater Dandenong',
    'Nillumbik',
    'Whittlesea',
    'Frankston',
    'Macedon Ranges',
    'Yarra Ranges',
    'Wyndham',
    'Cardinia',
    'Moorabool'
]

region_list = [
    'Eastern Metropolitan',
    'Eastern Victoria',
    'Northern Metropolitan',
    'Northern Victoria',
    'South-Eastern Metropolitan',
    'Southern Metropolitan',
    'Western Metropolitan',
    'Western Victoria'
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
st.markdown(
    "Predict Melbourne property prices using the final tuned CatBoost model 🤖"
)

# ==========================================
# Form
# ==========================================

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    # ------------------
    # Location
    # ------------------
    with col1:

        st.subheader("📍 Location Information")

        Suburb = st.selectbox(
            "🏘️ Suburb",
            options=sorted(suburb_list),
            index=None,
            placeholder="🔍 Search suburb..."
        )

        CouncilArea = st.selectbox(
            "🏛️ Council Area",
            options=sorted(council_list)
        )

        Regionname = st.selectbox(
            "🌏 Region",
            options=region_list
        )

        Lattitude = st.number_input(
            "📌 Latitude",
            value=-37.8136,
            format="%.6f"
        )

        Longtitude = st.number_input(
            "📌 Longitude",
            value=144.9631,
            format="%.6f"
        )

    # ------------------
    # Property Details
    # ------------------
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
            value=2
        )

        Car = st.number_input(
            "🚗 Car Spaces",
            min_value=0,
            value=1
        )

        Distance = st.number_input(
            "📏 Distance from CBD (KM)",
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
            value=5.0
        )

    st.divider()

    st.subheader("💼 Sale Information")

    Type_display = st.selectbox(
        "🏠 Property Type",
        options=list(type_map.keys())
    )

    Method_display = st.selectbox(
        "🤝 Sale Method",
        options=list(method_map.keys())
    )

    predict_btn = st.form_submit_button("🚀 Predict Price")

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

        prediction_log = model.predict(input_df)
        prediction_price = np.expm1(prediction_log)[0]

        st.success("✅ Prediction Generated Successfully")

        st.metric(
            label="💰 Estimated House Price",
            value=f"AUD {prediction_price:,.0f}"
        )

        st.info(
            "🤖 This prediction was generated using the final tuned CatBoost model."
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
