import streamlit as st
import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import datetime  # Import the datetime module

# Load the exported_data variable
with open("src/Asset/ML_Comp/exported_data.pkl", "rb") as f:
    exported_data = pickle.load(f)

def load_model():
    # Load the saved XGBoost model
    model_path = "src/Asset/ML_Comp/xgb_model.json"  # Replace with the actual path
    best_model= xgb.Booster()
    best_model.load_model(model_path)

    encoder = exported_data['encoder']
    scaler = exported_data['scaler']
    numeric_features = exported_data['numerical_imputer']
    categorical_features = exported_data['categorical_imputer']
    return best_model, encoder, scaler, numeric_features, categorical_features

def preprocess_data(input_df, encoder, scaler, numeric_features, categorical_features):
    df = input_df.copy()

    # Separate the categorical and numeric features
    categorical_df = df[categorical_features]
    numeric_df = df[numeric_features]

    # Encode the categorical features
    encoded_categorical = encoder.transform(categorical_df)

    # Scale the numeric features
    scaled_numeric = scaler.transform(numeric_df)

    # Concatenate the encoded categorical and scaled numeric features
    processed_df = np.concatenate((scaled_numeric, encoded_categorical), axis=1)

    # Get the names of columns after preprocessing
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(encoded_feature_names)

    # Convert the NumPy array to a pandas DataFrame
    processed_df = pd.DataFrame(processed_df, columns=all_feature_names)

    st.write("Columns in the DataFrame:", processed_df.columns.tolist())
    st.write("DataFrame Shape:", processed_df.shape)

    return processed_df

def main():
    st.set_page_config(page_title="Sales Prediction App", page_icon="📊", layout="wide")
    st.title("Sales Prediction App")
    st.markdown(
        """
<style>
    .stApp {
        background-color: #008080;  /* Teal */
        color: #000000;
    }
    .stTextInput {
        color: #000000;
        background-color: #ff00ff;  /* Magenta */
    }
    .stButton {
        background-color: #ffd700;  /* Gold */
        color: #000000;
    }
</style>
        """,
        unsafe_allow_html=True
    )

    # Load the ML model and data preprocessors
    best_model, encoder, scaler, numeric_features, categorical_features = load_model()

   # User input section
    date = st.date_input("Date")  # Use date_input

    # Extract year, month, and day from the selected date
    selected_date = datetime.datetime.strptime(str(date), '%Y-%m-%d')

    store_nbr = st.number_input("Store Number", min_value=1, step=1)
    family = st.selectbox("Select Family", ["AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS", "BREAD/BAKERY",
                                            "PET SUPPLIES", "PLAYERS AND ELECTRONICS", "POULTRY",
                                            "PREPARED FOODS", "PRODUCE", "SCHOOL AND OFFICE SUPPLIES", "SEAFOOD"])
    onpromotion = st.radio("On Promotion", ["Not on Promotion", "On Promotion"])

    # Predict button
    if st.button("Predict Sales"):
        # Prepare input data for prediction
        input_data = {
            'Year': [selected_date.year],
            'Month': [selected_date.month],
            'Day': [selected_date.day],
            'Store Number': [store_nbr],
            'family': [family],
            'onpromotion': [onpromotion],
            'sales': [0],  # Placeholder value for sales column
            'dcoilwtico': [0],  # Placeholder value for dcoilwtico column
            'transactions': [0],  # Placeholder value for transactions column
        }

        input_df = pd.DataFrame(input_data)

        # Preprocess the input data
        df = preprocess_data(input_df, encoder, scaler, numeric_features, categorical_features)

        # Make predictions using the loaded XGBoost model
        prediction = best_model.predict(df) # type: ignore

        # Display the prediction
        st.write(f"Predicted Sales: {prediction[0]:.2f}")

if __name__ == "__main__":
  main()
