import base64

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.options.mode.copy_on_write = True

# Load the trained model
model = joblib.load("./model/xgb_loan_rate_predictor.pkl")

# Define the categorical columns
categorical_cols = [
    "term",
    "emp_title",
    "emp_length",
    "home_ownership",
    "verification_status",
    "issue_d",
    "loan_status",
    "pymnt_plan",
    "purpose",
    "title",
    "zip_code",
    "addr_state",
    "earliest_cr_line",
    "initial_list_status",
    "last_pymnt_d",
    "last_credit_pull_d",
    "application_type",
    "hardship_flag",
    "disbursement_method",
    "debt_settlement_flag",
]

# Define the features you want to use
x_cols = [
    "total_rec_int",
    "fico_range_high",
    "last_fico_range_low",
    "percent_bc_gt_75",
    "total_bc_limit",
    "num_tl_op_past_12m",
    "dti",
    "mo_sin_old_rev_tl_op",
    "mths_since_recent_inq",
    "collection_recovery_fee",
    "mths_since_recent_bc",
    "out_prncp_inv",
    "annual_inc",
]

cols_to_keep = x_cols + categorical_cols

# Initialize LabelEncoder and StandardScaler
scaler = StandardScaler()
le = LabelEncoder()


def download_link(
    object_to_download: pd.DataFrame, download_filename: str, download_link_text: str
) -> str:
    """
    Generates a link to download the given object_to_download.
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(
            index=False
        )  # Convert DataFrame to CSV
    b64 = base64.b64encode(
        object_to_download.encode()
    ).decode()  # Encode CSV data to base64
    return f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def predict(input_data: np.ndarray) -> np.ndarray:
    """
    Make a prediction using the trained model.
    
    Note: input_data is expected to be a NumPy array after preprocessing.
    """
    # Assuming input_data is already prepared as a NumPy array for prediction
    prediction = model.predict(input_data)
    return prediction


st.title("Loan Interest Rate Prediction App")

st.markdown(
    """
This app allows you to predict loan interest rates based on your loan and borrower information. 
Please upload your data as a CSV file. Ensure your data includes the necessary features that the model expects.
"""
)

# Create a file uploader for the user to upload files
uploaded_file = st.file_uploader("Upload your input CSV file", type="csv")

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    # Display the uploaded file
    st.write("Preview of uploaded file:")
    st.write(input_df.head())

    input_preprocessed_df = input_df.copy()
    input_preprocessed_df = input_preprocessed_df[cols_to_keep]

    # Apply LabelEncoder to each categorical column
    for col in categorical_cols:
        input_preprocessed_df[col] = le.fit_transform(input_preprocessed_df[col])

    # Standardize the features
    X_scaled = scaler.fit_transform(input_preprocessed_df[x_cols])

    # Add categorical features to X scaled
    input_preprocessed_df = np.concatenate(
        (X_scaled, input_preprocessed_df[categorical_cols]), axis=1
    )

    if st.button("Predict"):

        preds = predict(input_preprocessed_df)

        # Append predictions to the DataFrame
        input_df["Predicted Interest Rate"] = preds

        st.write("Predictions have been added to the DataFrame:")
        st.write(input_df.head())

        # Generate a link for downloading the results
        tmp_download_link = download_link(
            input_df, "predictions.csv", "Click here to download your predictions!"
        )
        st.markdown(tmp_download_link, unsafe_allow_html=True)

else:
    st.write("Please upload a file to begin.")
    