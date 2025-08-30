import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as escnn_nn

__all__ = ["ENN"]

class ENN(nn.Module):
    """ENN model. Can be equivariant to C_N or D_N. D_4 used for most experiments.

    Args:
        num_channels (int, optional): Number of input channels. Defaults to 1.
        num_classes (int, optional): Number of classes. Defaults to 3.
        input_size (tuple, optional): Input size. Defaults to (100, 100).
        N (int, optional): Number of rotations. Defaults to 4.
        dihedral (bool, optional): Whether to use dihedral group. Defaults to True.
    """
    def __init__(
        self,
        num_channels: int = 3,
        num_classes: int = 3,
        input_size: tuple = (100, 100),
        N=4,
        dihedral=True,
    ):
        super(ENN, self).__init__()

        if N == 1:
            self.r2_act = gspaces.trivialOnR2()  # D1 group and C1 group

        else:
            if dihedral:
                self.r2_act = gspaces.flipRot2dOnR2(
                    N=N
                )  # D4 group with 4 rotations and flip
            else:
                self.r2_act = gspaces.rot2dOnR2(
                    N=N
                )  # D4 group with 4 rotations and flip

        self.input_type = escnn_nn.FieldType(
            self.r2_act, num_channels * [self.r2_act.trivial_repr]
        )
        self.conv1 = escnn_nn.R2Conv(
            in_type=self.input_type,
            out_type=escnn_nn.FieldType(self.r2_act, 8 * [self.r2_act.regular_repr]),
            kernel_size=5,
            padding=2,
        )
        self.bn1 = escnn_nn.InnerBatchNorm(self.conv1.out_type)
        self.relu1 = escnn_nn.ReLU(self.conv1.out_type)
        self.pool1 = escnn_nn.PointwiseMaxPool2D(
            self.conv1.out_type, kernel_size=2, stride=2, padding=0
        )
        self.dropout1 = escnn_nn.PointwiseDropout(self.conv1.out_type, p=0.2)

        self.conv2 = escnn_nn.R2Conv(
            in_type=self.conv1.out_type,
            out_type=escnn_nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
            kernel_size=3,
            padding=1,
        )

        self.bn2 = escnn_nn.InnerBatchNorm(self.conv2.out_type)
        self.relu2 = escnn_nn.ReLU(self.conv2.out_type)
        self.pool2 = escnn_nn.PointwiseMaxPool2D(
            self.conv2.out_type, kernel_size=2, stride=2, padding=0
        )
        self.dropout2 = escnn_nn.PointwiseDropout(self.conv2.out_type, p=0.2)

        self.conv3 = escnn_nn.R2Conv(
            in_type=self.conv2.out_type,
            out_type=escnn_nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
            kernel_size=3,
            padding=1,
        )
        self.bn3 = escnn_nn.InnerBatchNorm(self.conv3.out_type)
        self.relu3 = escnn_nn.ReLU(self.conv3.out_type)
        self.pool3 = escnn_nn.PointwiseMaxPool2D(
            self.conv3.out_type, kernel_size=2, stride=2, padding=0
        )
        self.dropout3 = escnn_nn.PointwiseDropout(self.conv3.out_type, p=0.2)

        self.gpool = escnn_nn.GroupPooling(self.pool3.out_type)

        c = self.gpool.out_type.size
        dummy_input = torch.zeros(1, num_channels, *input_size)
        dummy_input = escnn_nn.GeometricTensor(dummy_input, self.input_type)
        with torch.no_grad():
            dummy_output = self.gpool(
                self.pool3(
                    self.relu3(
                        self.bn3(
                            self.conv3(
                                self.pool2(
                                    self.relu2(
                                        self.bn2(
                                            self.conv2(
                                                self.pool1(
                                                    self.relu1(
                                                        self.bn1(
                                                            self.conv1(dummy_input)
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        flatten_dim = dummy_output.tensor.view(1, -1).shape[1]

        self.fc1 = nn.Linear(in_features=flatten_dim, out_features=256)
        self.fc1.weight.data.normal_(0, 0.005)
        self.fc1.bias.data.fill_(0.0)
        self.layer_norm = nn.LayerNorm(256)

        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0.0)

    def forward(self, x):
        x = escnn_nn.GeometricTensor(x, self.input_type)

        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.dropout3(x)

        x = self.gpool(x)

        x = x.tensor.view(x.tensor.size(0), -1)
        x = self.fc1(x)
        x = self.layer_norm(x)
        latent_space = x

        x = self.fc2(x)

        return latent_space, x
