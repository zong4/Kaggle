# Description: Predict house prices using PyTorch 
# Author: Zong
# Date: 2025-02-26
# Python version: 3.9
# Download data from: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data


# Load data
import sys
import pandas as pd

basic_path = 'house-prices-advanced-regression-techniques'
train_data = pd.read_csv(basic_path + "/train.csv")
test_data = pd.read_csv(basic_path + "/test.csv")


# Prepare environment
import functions_pytorch

functions_pytorch.set_seed(42)


# Extract target variable
train_labels = train_data['SalePrice']
train_data.drop(['SalePrice'], axis=1, inplace=True)


# Data preprocessing
import functions

train_data = functions.drop_useless_cols(train_data, test_data)

train_data.drop(['Id'], axis=1, inplace=True)
train_data = functions.drop_cols_with_same_data(train_data, 0.9)
train_data = functions.drop_cols_with_na(train_data, 0.8)
train_data = functions.fill_na_with_mean(train_data)
train_data = functions.normalize(train_data)
train_data = functions.one_hot_encoding(train_data)
print(train_data.info())
print()
print(train_data.head())
print()

test_data.drop(['Id'], axis=1, inplace=True)
test_data = functions.fill_na_with_mean(test_data)
test_data = functions.normalize(test_data)
test_data = functions.one_hot_encoding(test_data)

test_data = functions.drop_useless_cols(test_data, train_data)
test_data = functions.add_missing_dummy_columns(test_data, train_data)
test_data = functions.sort_columns(test_data, train_data)


# Decision Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

model = RandomForestRegressor()
model.fit(train_data, train_labels)

predictions = model.predict(test_data)
print("Mean Absolute Error: " + str(mean_absolute_error(train_labels, model.predict(train_data))))
print()

submission = pd.DataFrame({'Id': test_data.index + 1461, 'SalePrice': predictions})
submission.to_csv(basic_path + '/submission.csv', index=False)
print(submission.head())