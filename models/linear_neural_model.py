import torch
import torch.nn as nn


class LinearNeuralModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearNeuralModel, self).__init__()
        self.lin1 = nn.Linear(input_dim, output_dim)
        self.norm1 = nn.BatchNorm1d(output_dim)
        self.lin2 = nn.Linear(output_dim, output_dim)
        self.norm2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.lin1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.sigmoid(out)
        return out


def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model
