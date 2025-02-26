import torch
import random
import numpy as np


# Set device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
print()


# Set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Logarithmic root mean squared error
class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()
    
    def forward(self, pred, actual):
        pred = torch.clamp(pred, min=0)
        actual = torch.clamp(actual, min=0)
        return torch.sqrt(torch.mean((torch.log(pred + 1) - torch.log(actual + 1)) ** 2))

# Train model
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data.to(device))
            loss = criterion(outputs, labels.to(device).view(-1, 1))
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())

        if val_loader:
            with torch.no_grad():
                model.eval()
                total_loss = 0
                for data, labels in val_loader:
                    outputs = model(data.to(device))
                    total_loss += criterion(outputs, labels.to(device).view(-1, 1)).item()
                val_losses.append(total_loss / len(val_loader))
                model.train()

        if (epoch + 1) % 10 == 0:
            if val_loader:
                print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}')
            else:
                print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]}')
    return train_losses, val_losses
