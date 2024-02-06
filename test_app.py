import numpy as np
import pandas as pd

from app import (
    categorical_cols,
    cols_to_keep,
    download_link,
    le,
    predict,
    scaler,
    x_cols,
)

pd.options.mode.copy_on_write = True

input_df = pd.read_csv("DataScienceAssignment_test.csv")

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


def test_predict():
    """
    A function that takes a pandas DataFrame as input and returns a pandas Series. It calls the predict function and then performs checks on the returned predictions.
    """

    # Call the predict function
    predictions = predict(input_preprocessed_df)

    # Check if predictions are returned as expected
    assert isinstance(
        predictions, np.ndarray
    ), "Predictions should be returned as an array"
    assert len(predictions) > 0, "Should return a non-empty series of predictions"
    assert int(predictions[0]) == 10


preds = predict(input_preprocessed_df)

# Append predictions to the DataFrame
input_df["Predicted Interest Rate"] = preds


def test_download_link() -> None:
    """
    Call the download_link function
    Check if the link is correctly formatted
    """
    # Call the download_link function
    link = download_link(input_df, "predictions.csv", "Download Predictions")

    # Check if the link is correctly formatted
    assert "predictions.csv" in link, "The download link should be correctly formatted"
