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


# Set up model
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

class HousePricesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):    
        try:
            data = torch.tensor(self.data.iloc[idx].values.astype('float32'))
            label = torch.tensor(self.labels.iloc[idx].astype('float32'))
            return data, label
        except KeyError as e:
            print(f"KeyError: {e} at index {idx}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e} at index {idx}")
            raise
    
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
model = Net(train_data.shape[1])
summary(model, input_size=(train_data.shape[1],))
model.to(functions_pytorch.device)
print()


# Train model with KFold cross validation
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

k = 5
num_epochs = 100
batch_size = 32
learning_rate = 0.01

criterion = functions_pytorch.RMSLELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_kfold(train_data, train_labels, model, criterion, optimizer, k):
    kf = KFold(n_splits=k, shuffle=True)

    cnt = 1
    for train, val in kf.split(train_data):
        print(f"Fold {cnt}")

        train_data_fold = train_data.iloc[train]
        train_labels_fold = train_labels.iloc[train]
        val_data_fold = train_data.iloc[val]
        val_labels_fold = train_labels.iloc[val]

        train_dataset = HousePricesDataset(train_data_fold, train_labels_fold)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = HousePricesDataset(val_data_fold, val_labels_fold)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        train_losses, val_losses = functions_pytorch.train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs)

        # Plot losses
        plt.clf()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='val Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.savefig(basic_path + f'/loss_fold_{cnt}.png')

        cnt += 1
        print()
train_kfold(train_data, train_labels, model, criterion, optimizer, k)


# Predict test data
def predict_test_data(model, test_data):
    # Train model with all data
    print("Train model with all data")
    train_dataset = HousePricesDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    functions_pytorch.train_model(model, criterion, optimizer, train_loader, None, num_epochs)

    test_dataset = HousePricesDataset(test_data, pd.Series([0] * len(test_data)))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = []
    with torch.no_grad():
        model.eval()
        for data, _ in test_loader:
            outputs = model(data.to(functions_pytorch.device))
            predictions.append(outputs.item())
        model.train()

    print()
    return predictions
# best_model_path = 'model_fold_5.pth'
# model.load_state_dict(torch.load(best_model_path))
predictions = predict_test_data(model, test_data)

submission = pd.DataFrame({'Id': test_data.index + 1461, 'SalePrice': predictions})
submission.to_csv(basic_path + '/submission.csv', index=False)
print(submission.head())