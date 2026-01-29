import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Car Price Predictor",
    layout="centered"
)

st.markdown(
    """
    <style>
    header {display: none;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stHeader"] {display: none;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 720px;
    }

    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    h1 {
        text-align: center;
        font-size: 36px;
        margin-bottom: 4px;
    }

    .subtitle {
        text-align: center;
        color: #8b949e;
        margin-bottom: 28px;
        font-size: 15px;
    }

    label, p, span {
        color: #e6edf3 !important;
    }

    div[data-baseweb="select"] > div,
    input {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }

    .stSlider > div {
        color: #e6edf3;
    }

    button {
        background-color: #238636 !important;
        color: white !important;
        border-radius: 8px !important;
        height: 48px;
        font-size: 16px;
        margin-top: 10px;
    }

    button:hover {
        background-color: #2ea043 !important;
    }

    hr {
        border: none;
        border-top: 1px solid #30363d;
        margin: 24px 0;
    }

    .footer {
        text-align: center;
        color: #8b949e;
        font-size: 13px;
        margin-top: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

model = pickle.load(open("log_linearregressionmodel.pkl", "rb"))

@st.cache_data
def load_data():
    return pd.read_csv("model_training/car_clean_data.csv")

data = load_data()

companies = sorted(data["company"].unique())
fuel_types = sorted(data["fuel_type"].unique())

st.markdown("<h1>tamimystic Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Used Car Price Prediction using Linear Regression</div>",
    unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)

st.subheader("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Company", companies)
    car_names = sorted(data[data["company"] == company]["name"].unique())
    name = st.selectbox("Car Name", car_names)
    year = st.slider(
        "Manufacturing Year",
        int(data["year"].min()),
        int(data["year"].max()),
        2019
    )

with col2:
    kms_driven = st.number_input(
        "Kilometers Driven",
        min_value=100,
        step=1000,
        value=50000
    )
    fuel_type = st.selectbox("Fuel Type", fuel_types)

if "Diesel" in name:
    fuel_type = "Diesel"

st.markdown("<hr>", unsafe_allow_html=True)

if st.button("Predict Price", use_container_width=True):
    input_df = pd.DataFrame(
        [[name, company, year, kms_driven, fuel_type]],
        columns=["name", "company", "year", "kms_driven", "fuel_type"]
    )
    log_price = model.predict(input_df)[0]
    price = np.exp(log_price)
    st.success(f"Estimated Car Price: BDT {round(price, 2):,}")

st.markdown(
    "<div class='footer'>This prediction is based on a custom dataset and is not a real market valuation.</div>",
    unsafe_allow_html=True
)
