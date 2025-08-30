import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ["CNN"]

class CNN(nn.Module):
    """CNN model.

    Args:
        num_channels (int, optional): Number of input channels. Defaults to 1.
        num_classes (int, optional): Number of classes. Defaults to 3.
        input_size (tuple, optional): Input size. Defaults to (100, 100).
    """
    def __init__(
        self,
        num_channels: int = 3,
        num_classes: int = 3,
        input_size: tuple = (100, 100),
        feature_dim: int = 256,
    ):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.dropout = nn.Dropout(p=0.2)

        # Compute flattened size
        dummy_input = torch.zeros(1, num_channels, *input_size)
        with torch.no_grad():
            dummy_output = self.pool3(
                F.relu(self.bn3(self.conv3(
                    self.pool2(
                        F.relu(self.bn2(self.conv2(
                            self.pool1(
                                F.relu(self.bn1(self.conv1(dummy_input)))
                            )
                        )))
                    )
                )))
            )
        flatten_dim = dummy_output.view(1, -1).shape[1]

        # Bottleneck Layer (Fully Connected)
        self.fc1 = nn.Linear(in_features=flatten_dim, out_features=256)
        self.fc1.weight.data.normal_(0, 0.005)
        self.fc1.bias.data.fill_(0.0)
        self.layer_norm = nn.LayerNorm(256)

        # Output Layer (Fully Connected)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0.0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.layer_norm(x)
        latent_space = x

        x = self.fc2(x)

        return latent_space, x
