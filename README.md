# Car Price Predictor

## Project Website
- https://tamimystic-car-price-predictor.streamlit.app/

## Overview
Car Price Predictor is a Machine Learning–based web application that predicts the **price of used cars** using features such as car name, company, manufacturing year, kilometers driven, and fuel type.  
The project follows a complete ML pipeline from data cleaning and EDA to model training, evaluation, and deployment with Streamlit.     
This model predict price over the custom dataset, so take it seriously.

---

## Features
- Real-time car price prediction  
- Clean and professional dark-themed UI  
- Log-transformed regression to avoid negative price predictions  
- Same preprocessing pipeline used during training and inference  

---

## Tech Stack
- **Language:** Python  
- **Web Framework:** Streamlit  
- **Machine Learning:** Scikit-learn  
- **Data Processing:** Pandas, NumPy  
- **Data Visualization:** Matplotlib, Seaborn(i did not use it)  

---

## Model Evaluation
Multiple regression models were tested to find the most reliable performer.

| Model | R² Score |
|------|---------|
| **Linear Regression (Log Transformed)** | **84.57%** |
| Random Forest Regressor | 70.01% |
| Gradient Boosting Regressor | 70.41% |

**Final Choice:** Log-Transformed Linear Regression  
This approach was selected to maintain prediction stability and prevent negative output values.

---

## How to Run Locally

### 1. Create Virtual Environment
- python -m venv venv

### 2. Activate Environment
- venv\Scripts\activate

### 3. Clone the Repository
- git clone https://github.com/tamimystic/Car-Price-Predictor-ML-Project.git

### 4. Install Dependencies
- pip install -r requirements.txt

### 5. Run the Application
- streamlit run app.py

## Project Structure
```text
CAR-PRICE-PREDICTOR-ML-PROJECT
├── .devcontainer/
│   └── devcontainer.json
├── model_training/
│   ├── car_data.csv
│   ├── car_clean_data.csv
│   ├── car_price_predictor.ipynb
│   └── linearregressionmodel.pkl
├── retrain_env/
├── app.py
├── log_linearregressionmodel.pkl
├── README.md
├── requirements.txt
├── runtime.txt
├── LICENSE
└── .gitignore




