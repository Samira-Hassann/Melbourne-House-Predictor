import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# ============================================================
# تعريف الكلاس المخصص (نفس اللي في نوتبوك التدريب بالظبط)
# لازم يكون معرّف قبل joblib.load عشان الـ pickle يلاقيه
# ============================================================
class MedianTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, m=10):
        self.m = m
        self.mapping = None
        self.global_median = None

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = y.values if hasattr(y, 'values') else y
        X['target'] = y
        self.global_median = X['target'].median()
        col = X.columns[0]
        counts = X.groupby(col)['target'].count()
        medians = X.groupby(col)['target'].median()
        self.mapping = (counts * medians + self.m * self.global_median) / (counts + self.m)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        col = X.columns[0]
        encoded = X[col].map(self.mapping)
        return encoded.fillna(self.global_median).to_frame()

st.set_page_config(page_title="Melbourne House Price Predictor", page_icon="🏡", layout="wide")

model = joblib.load("catboost_house_price_pipeline.pkl")

# ... باقي الكود زي ما هو بالظبط (suburb_list, الفورم, إلخ)

suburb_list = [
    'Abbotsford','Aberfeldie','Airport West','Albanvale','Albert Park','Albion','Alphington',
    'Altona','Altona Meadows','Altona North','Ardeer','Armadale','Ascot Vale','Ashburton',
    'Ashwood','Aspendale','Aspendale Gardens','Attwood','Avondale Heights','Bacchus Marsh',
    'Balaclava','Balwyn','Balwyn North','Bayswater','Bayswater North','Beaconsfield',
    'Beaconsfield Upper','Beaumaris','Bellfield','Bentleigh','Bentleigh East','Berwick',
    'Black Rock','Blackburn','Blackburn North','Blackburn South','Bonbeach','Boronia',
    'Box Hill','Braybrook','Briar Hill','Brighton','Brighton East','Broadmeadows','Brookfield',
    'Brooklyn','Brunswick','Brunswick East','Brunswick West','Bulleen','Bullengarook',
    'Bundoora','Burnley','Burnside','Burnside Heights','Burwood','Burwood East','Cairnlea',
    'Camberwell','Campbellfield','Canterbury','Carlton','Carlton North','Carnegie',
    'Caroline Springs','Carrum','Carrum Downs','Caulfield','Caulfield East','Caulfield North',
    'Caulfield South','Chadstone','Chelsea','Cheltenham','Clayton','Clayton South',
    'Clifton Hill','Coburg','Coburg North','Collingwood','Craigieburn','Cranbourne',
    'Cremorne','Croydon','Dandenong','Dandenong North','Docklands','Doncaster','Doncaster East',
    'Donvale','Doreen','Epping','Essendon','Fitzroy','Footscray','Frankston','Glen Iris',
    'Glen Waverley','Glenroy','Greensborough','Hampton','Hawthorn','Heidelberg','Highett',
    'Hoppers Crossing','Ivanhoe','Kew','Kingsbury','Knoxfield','Lalor','Malvern',
    'Maribyrnong','Melbourne','Melton','Mentone','Mill Park','Mitcham','Moonee Ponds',
    'Moorabbin','Mount Waverley','Narre Warren','Newport','Northcote','Oakleigh','Ormond',
    'Pakenham','Point Cook','Prahran','Preston','Richmond','Ringwood','Rowville','Sandringham',
    'South Yarra','Southbank','Springvale','St Kilda','Sunbury','Sunshine','Tarneit',
    'Thornbury','Toorak','Truganina','Werribee','Wheelers Hill','Williamstown','Windsor',
    'Wyndham Vale','Yarraville'
]

council_list = [
    'Banyule','Bayside','Boroondara','Brimbank','Cardinia','Casey','Darebin',
    'Frankston','Glen Eira','Greater Dandenong','Hobsons Bay','Hume','Kingston',
    'Knox','Macedon Ranges','Manningham','Maribyrnong','Maroondah','Melbourne',
    'Melton','Monash','Moonee Valley','Moorabool','Moreland','Nillumbik',
    'Port Phillip','Stonnington','Whitehorse','Whittlesea','Wyndham','Yarra',
    'Yarra Ranges','Unavailable'
]

region_list = [
    'Eastern Metropolitan','Eastern Victoria','Northern Metropolitan','Northern Victoria',
    'South-Eastern Metropolitan','Southern Metropolitan','Western Metropolitan','Western Victoria'
]


type_map = {"🏠 House":"h","🏢 Unit":"u","🏘️ Townhouse":"t"}
method_map = {"🤝 Sold":"S","📑 Sold Prior":"SP","💼 Private Sale":"PI","🔨 Auction":"SA"}


st.title("🏡 Melbourne House Price Predictor")

with st.form("form"):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Location")

        suburb = st.selectbox("🏘️ Suburb", suburb_list, index=None)
        council = st.selectbox("🏛️ Council Area", council_list)
        region = st.selectbox("🌏 Region", region_list)

        lat = st.number_input("📌 Latitude", value=-37.81)
        lon = st.number_input("📌 Longitude", value=144.96)

    with col2:
        st.subheader("🏠 Property")

        rooms = st.number_input("🛏️ Rooms", 1, 10, 3)
        bath = st.number_input("🛁 Bathrooms", 1, 10, 2)
        car = st.number_input("🚗 Cars", 0, 5, 1)
        dist = st.number_input("📏 Distance", 0.0, 50.0, 10.0)
        prop_count = st.number_input("🏘️ Property Count", 0, 20000, 5000)
        land = st.number_input("🌳 Land Size", 0.0, 5000.0, 500.0)

    st.subheader("💼 Sale Info")

    typ = st.selectbox("🏠 Type", list(type_map.keys()))
    method = st.selectbox("🤝 Method", list(method_map.keys()))

    submit = st.form_submit_button("🚀 Predict")

if submit:

    input_df = pd.DataFrame([{
        "Suburb": suburb,
        "Rooms": rooms,
        "Type": type_map[typ],
        "Method": method_map[method],
        "Distance": dist,
        "Bathroom": bath,
        "Car": car,
        "CouncilArea": council,
        "Lattitude": lat,
        "Longtitude": lon,
        "Regionname": region,
        "Propertycount": prop_count,
        "LogLandsize": np.log1p(land)
    }])

    pred_log = model.predict(input_df)
    price = np.expm1(pred_log)[0]

    st.success("✅ Prediction Done")
    st.metric("💰 Estimated Price", f"AUD {price:,.0f}")
