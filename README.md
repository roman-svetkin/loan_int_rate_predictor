# TikTok CCS Category Scoring Tool

## Overview

This tool is designed to analyze and rank TikTok category, age, and gender combinations for a list of GLOBAL_IDs. It integrates multiple data sources and utilizes various ranking methodologies including raw counts, random forest, and TF-IDF.

## Features

- Parses command line arguments for configuration.
- Reads and processes data from CSV and JSON files.
- Applies different ranking methods: raw counts, random forest, and TF-IDF.
- Maps TikTok IDs to category names.
- Augments data with TikTok reach information.
- Outputs the results in a specified CSV file.

## Usage

Run the script with the desired arguments:

```bash
python tiktok_pipeline.py --reach_threshold [value] --top_n [value] --ranking_type [type] --audience_builder_csv_file_path [path] --json_tiktok_to_css_file_path [path] --output_file_path [path] --tiktok_categories_csv_path [path]
```

## Arguments

- `--reach_threshold`: The minimum reach threshold for filtering data.
- `--top_n`: Number of top attribute combinations to consider.
- `--ranking_type`: Type of ranking ('raw_counts', 'random_forest', or 'tf_idf').
- `--audience_builder_csv_file_path`: Path to the audience builder CSV file.
- `--json_tiktok_to_css_file_path`: Path to the TikTok to CCS JSON file.
- `--output_file_path`: Path for the output CSV file.
- `--tiktok_categories_csv_path`: Path to the TikTok categories CSV file.

Make sure `TIKTOK_ACCESS_TOKEN` and `TIKTOK_ADVERTISER_ID` are set in the environment.

There are different methods for ranking and scoring TikTok category, age, and gender combinations. Here's a breakdown of each method:

1. **Raw Counts**:
   - This method simply counts the occurrences of each combination of TikTok category, age, and gender in the data.
   - The `apply_combinations` function generates these counts by iterating through the DataFrame and tallying each unique combination.
   - The counts are then normalized and sorted, with the top `n` combinations (as specified by the `--top_n` argument) being selected.
   - This method is straightforward and effective for identifying the most frequent combinations but doesn't account for the uniqueness or significance of each combination beyond its frequency.

2. **Random Forest**:
   - The Random Forest method uses a machine learning approach. It is slow and may run up to 30 minutes to complete. Its results are similar to TF-IDF which runs in less than a minute.
   - It first encodes the data using one-hot encoding for categorical variables (like TikTok category, age, and gender) and label encoding for the target variable (GLOBAL_IDs).
   - The RandomForestClassifier from scikit-learn is trained on this data.
   - The importance of each feature (i.e., each TikTok category, age, and gender combination) is then extracted from the model.
   - The `calculate_random_forest_scores` function maps these importances back to the original features, calculates a composite score for each combination, and selects the top `n` scores.
   - This method considers the influence of each combination on the model's ability to classify the GLOBAL_IDs, offering a more nuanced understanding of the data's structure.

3. **TF-IDF (Term Frequency-Inverse Document Frequency)**:
   - TF-IDF is a statistical measure used to evaluate the importance of a word or combination of words within a dataset. It is used by default.
   - The `calculate_tf_idf` function first calculates the term frequency (TF) for each TikTok category, age, and gender combination.
   - It then calculates the inverse document frequency (IDF) for each combination, which diminishes the weight of combinations that occur too commonly across all documents (GLOBAL_IDs).
   - The TF and IDF scores are multiplied to obtain the TF-IDF score for each combination.
   - This method is effective in identifying combinations that are both frequent and uniquely significant across the dataset, providing a balance between raw frequency and uniqueness.

Each of these methods offers a different perspective on the data, with raw counts focusing on sheer frequency, random forest providing insights based on a machine learning model, and TF-IDF balancing frequency with uniqueness.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request with your changes.

## License

(C) 2024 Dentsu London Ltd. All rights reserved.

## Contact

For more information or support, please contact <roman.svetkin@dentsu.com>.
