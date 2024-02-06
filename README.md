# Loan Interest Rate Prediction App

## Overview

This Loan Interest Rate Prediction App is designed to help users predict interest rates on loans based on borrower and loan information. It utilises a trained model to offer predictions directly within a web interface built with Streamlit.

## Features

- Predict loan interest rates using borrower's loan information.
- Upload borrower data in CSV format for predictions.
- Download the predictions appended to the uploaded data.

## How to Use

1. Navigate to the Streamlit app URL.
2. Upload your data as a CSV file by clicking on the "Upload your input CSV file" button.
3. The app will display a preview of the uploaded file.
4. Click on "Predict" to generate interest rate predictions, which will be added to the DataFrame and displayed.
5. Download the DataFrame with predictions by clicking on the provided link.

## Model Information

The prediction model is an XGBoost model trained on loan data, saved as `xgb_loan_rate_predictor.pkl`. It predicts interest rates based on various loan and borrower characteristics.

## Preprocessing Steps

Data preprocessing includes:

- Encoding categorical features using `LabelEncoder`.
- Standardizing numerical features with `StandardScaler`.
- Keeping specified features for prediction.

## Prediction and Download

After processing the uploaded data, predictions are made and appended to the uploaded DataFrame. Users can download the result as a CSV file.

## Developer Notes

- Adjust the path to the model file as needed.
- Customize categorical and numerical features based on your model's training.
- The app's functionality can be extended or modified according to specific requirements.

### Quick Start

Clone this repository, install the dependencies, and run the Streamlit app using `streamlit run app.py --server.enableXsrfProtection=false --server.port=7860`.

---
