import streamlit as st
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Load the exported_data variable
with open("src/Asset/ML_Comp/exported_data.pkl", "rb") as f:
    exported_data = pickle.load(f)

def load_model():
    model = exported_data['best_model']
    encoder = exported_data['encoder']
    scaler = exported_data['scaler']
    numeric_features = exported_data['numerical_imputer']
    categorical_features = exported_data['categorical_imputer']
    return model, encoder, scaler, numeric_features, categorical_features

def preprocess_data(input_df, encoder, scaler, numeric_features, categorical_features):
    df = input_df.copy()
    df.drop(columns=['Date'], inplace=True)

    # Encode categorical features using the provided encoder
    encoded_features = encoder.transform(df[categorical_features])
    df_encoded = pd.DataFrame(encoded_features, columns=encoder.get_feature_names(categorical_features))

    # Concatenate the encoded features with the original DataFrame
    df = pd.concat([df.drop(columns=categorical_features), df_encoded], axis=1)

    # Scale the numeric features using the provided scaler
    df[numeric_features] = scaler.transform(df[numeric_features])

    return df

def main():
    st.title("Sales Prediction App")

    # Load the ML model and data preprocessors
    model, encoder, scaler, numeric_features, categorical_features = load_model()

    # User input section
    date = st.date_input("Date")
    store_nbr = st.number_input("Store Number", min_value=1, step=1)
    family = st.selectbox("Select Family", ["AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS", "BREAD/BAKERY",
                                            "PET SUPPLIES", "PLAYERS AND ELECTRONICS", "POULTRY",
                                            "PREPARED FOODS", "PRODUCE", "SCHOOL AND OFFICE SUPPLIES", "SEAFOOD"])
    onpromotion = st.radio("On Promotion", ["Not on Promotion", "On Promotion"])

    # Predict button
    if st.button("Predict Sales"):
        # Prepare input data for prediction
        input_data = {
            'Date': [date],
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
        prediction = model.predict(df)

        # Display the prediction
        st.write(f"Predicted Sales: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()
