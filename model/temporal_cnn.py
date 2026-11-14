import torch
import torch.nn as nn


class TemporalCNN(nn.Module):
    """
    Temporal convolutional neural network for EEG classification.

    Input:
        (batch_size, 64, 480)

    Output:
        (batch_size, 2)

    Classes:
        0 = Left fist
        1 = Right fist
    """

    def __init__(
        self,
        n_channels=64,
        n_classes=2,
        dropout=0.5,
    ):
        super().__init__()

        self.features = nn.Sequential(

            # First convolutional block
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=32,
                kernel_size=15,
                padding=7,
            ),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=2),

            # Second convolutional block
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=2),

            # Third convolutional block
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(64),
            nn.ELU(),

            # Temporal feature aggregation
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":

    model = TemporalCNN()

    # Simulated EEG input
    X = torch.randn(16, 64, 480)

    model.eval()

    with torch.no_grad():
        output = model(X)

    n_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Input shape:", tuple(X.shape))
    print("Output shape:", tuple(output.shape))
    print("Trainable parameters:", n_parameters)