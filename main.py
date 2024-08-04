import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

import io
import os
import ssl
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch import nn, optim

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, models

class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.pool= nn.MaxPool2d(kernel_size=2, stride=2)
    self.relu = nn.ReLU()
    self.flatten = nn.Flatten()
    self.dropout = nn.Dropout(p=0.5)
    self.fc1 = nn.Linear(32*14*14, 64)
    self.bn2 = nn.BatchNorm1d(64)
    self.fc2 = nn.Linear(64,32)
    self.fc3 = nn.Linear(32,10) # MNIST Contains Numbers from 0-9
  def forward(self, x):
    x=self.conv1(x)
    x=self.bn1(x)
    x=self.relu(x)
    x=self.pool(x)
    x=self.flatten(x)
    x=self.dropout(x)
    x=self.fc1(x)
    x=self.bn2(x)
    x=self.relu(x)
    x=self.fc2(x)
    x=self.relu(x)
    x=self.fc3(x)
    return x

ssl._create_default_https_context = ssl._create_unverified_context

transform = transforms.Compose([
    transforms.ToTensor()
])

data_path = 'data'

# Check if the main file is already present
file_present = os.path.exists(os.path.join(data_path, 'train-images-idx3-ubyte.gz'))

dataset = datasets.MNIST(root='data', train=True, download=not file_present, transform=transform)

train_size = int(0.8 * len(dataset))
dev_size = int(0.1* len(dataset))
test_size = len(dataset)-train_size-dev_size
train_dataset, dev_dataset, test_dataset = random_split(dataset, [train_size, dev_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_accuracies = []
val_accuracies = []


# Training and validation loop
num_epochs = 6
for epoch in range(num_epochs):
    model.train()
    correct_train = 0
    total_train = 0
    for step, batch in enumerate(train_dataloader):
        batch_X, batch_y = batch

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += batch_y.size(0)
        correct_train += (predicted == batch_y).sum().item()


    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)

    # Validation loop
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0
    with torch.no_grad():
        for batch in dev_dataloader:
            batch_X, batch_y = batch

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

            # Calculate validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_val += batch_y.size(0)
            correct_val += (predicted == batch_y).sum().item()

    val_loss /= len(dev_dataloader)
    val_accuracy = 100 * correct_val / total_val
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')
