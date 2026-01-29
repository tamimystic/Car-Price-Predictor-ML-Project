import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Car Price Predictor",
    layout="centered"
)

model = pickle.load(open("log_linearregressionmodel.pkl", "rb"))

@st.cache_data
def load_data():
    return pd.read_csv("model training/car_clean_data.csv")

data = load_data()

car_names = sorted(data["name"].unique())
companies = sorted(data["company"].unique())
fuel_types = sorted(data["fuel_type"].unique())

st.markdown(
    "<h1 style='text-align:center;'>tamimystic Car Price Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:gray;'>Predict used car prices using Machine Learning (Linear Regression)</p>",
    unsafe_allow_html=True
)

st.divider()

st.subheader("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox(
        "Company",
        companies
    )

    filtered_names = sorted(
        data[data["company"] == company]["name"].unique()
    )

    name = st.selectbox(
        "Car Name",
        filtered_names
    )

    year = st.slider(
        "Manufacturing Year",
        int(data["year"].min()),
        int(data["year"].max()),
        2019
    )

with col2:
    kms_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        step=1000,
        value=50000
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        fuel_types
    )

st.divider()

if st.button("Predict Price", use_container_width=True):
    input_df = pd.DataFrame(
        [[name, company, year, kms_driven, fuel_type]],
        columns=["name", "company", "year", "kms_driven", "fuel_type"]
    )

    prediction = model.predict(input_df)[0]
    pred_price = np.exp(prediction)

    st.success(f"Estimated Car Price: BDT {round(pred_price, 2):,}")


    st.markdown(
    "<hr><p style='text-align:center; color:gray;'> This is not real, Its predict based on custom dataset</p>",
    unsafe_allow_html=True
)

