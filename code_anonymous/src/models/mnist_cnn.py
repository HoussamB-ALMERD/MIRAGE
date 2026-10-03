import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np

class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_parameters(net: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy().copy() for val in net.state_dict().values()]

def set_parameters(net: nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = {}
    for (k, current_param), v in zip(net.state_dict().items(), parameters):
        state_dict[k] = torch.as_tensor(v, dtype=current_param.dtype)
    net.load_state_dict(state_dict, strict=True)

def train(net: nn.Module, train_loader: torch.utils.data.DataLoader, epochs: int, lr: float, device: str = "cpu", proximal_mu: float = 0.0) -> float:
    net.to(device)
    net.train()
    global_params = [p.detach().clone() for p in net.parameters()] if proximal_mu > 0 else None
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    total_loss = 0.0
    total_samples = 0
    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            if global_params is not None:
                prox = sum(((p - g) ** 2).sum() for p, g in zip(net.parameters(), global_params))
                loss = loss + (proximal_mu / 2.0) * prox
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
    return total_loss / max(1, total_samples)

def test(net: nn.Module, test_loader: torch.utils.data.DataLoader, device: str = "cpu"):
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total_loss, total_samples = 0, 0.0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            total_loss += criterion(outputs, labels).item() * len(labels)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss = total_loss / max(1, total_samples)
    accuracy = correct / max(1, total_samples)
    return loss, accuracy
