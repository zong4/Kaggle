import pandas as pd


# Drop cols if not in test data
def drop_useless_cols(data, target_data):
    for col in data.columns:
        if col not in target_data.columns:
            data = data.drop([col], axis=1)
    return data


# Add missing dummy columns
def add_missing_dummy_columns(data, target_data):
    missing_cols = set(target_data.columns) - set(data.columns)
    for col in missing_cols:
        data[col] = 0
    return data


# Sort columns
def sort_columns(data, target_data):
    data = data[target_data.columns]
    return data


# Drop cols if more than threshold of data is same
def drop_cols_with_same_data(data, threshold):
    for col in data.columns:
        if data[col].value_counts(normalize=True).values[0] > threshold:
            data = data.drop([col], axis=1)
    return data


# Drop cols if more than threshold of data is na
def drop_cols_with_na(data, threshold):
    for col in data.columns:
        if data[col].isna().sum() / len(data) > threshold:
            data = data.drop([col], axis=1)
    return data


# Fill na if data type is not object
def fill_na_with_mean(data):
    for col in data.columns:
        if data[col].dtype != 'object':
            data[col] = data[col].fillna(data[col].mean())
    return data


# Normalize data if data type is not object
def normalize(data):
    for col in data.columns:
        if data[col].dtype != 'object':
            data[col] = (data[col] - data[col].mean()) / data[col].std()
    return data


# One hot encoding with na
def one_hot_encoding(data):
    data = pd.get_dummies(data, dummy_na=True)
    return data