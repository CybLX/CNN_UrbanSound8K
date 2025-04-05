
import os
import torch
import torch.nn as nn
from torch.nn import init

# ----------------------------
# Audio Classification Model
# ----------------------------
class AudioClassifier (nn.Module):
    """
    Audio classification model using convolutional neural networks (CNNs).

    This model consists of four convolutional blocks followed by an adaptive 
    average pooling layer and a fully connected classification layer.

    Parameters
    ----------
    None

    Returns
    -------
    AudioClassifier
        A CNN-based model for audio classification.

    Notes
    -----
    The model uses Kaiming initialization for better convergence.
    """

    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self) -> None:
        """Initialize the CNN model architecture."""
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8) 
        
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # terceira Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Quarta Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]


        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor containing class scores.
        """
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x
    
    # ----------------------------
    # Save Checkpoint
    # ----------------------------
    def save_checkpoint(self, path: str, version: str) -> None:
        """
        Save the model checkpoint.

        Parameters
        ----------
        path : str
            Directory path where the checkpoint will be saved.
        version : str
            Version identifier for the checkpoint.

        Returns
        -------
        None
        """
        model_filepath = os.path.join(path, f"{version}")
        torch.save({f'{version}': self.state_dict()}, model_filepath)
    
    # ----------------------------
    # Load Checkpoint
    # ----------------------------
    @staticmethod
    def load_checkpoint(path: str, device: torch.device) -> 'AudioClassifier':
        """
        Load a model checkpoint.

        Parameters
        ----------
        path : str
            Path to the saved model checkpoint.
        device : torch.device
            Device to use for training (CPU/GPU).
        Returns
        -------
        AudioClassifier
            The loaded model.
        """

        checkpoint = torch.load(path, map_location=device)
        model = AudioClassifier()
        model.load_state_dict(checkpoint[next(iter(checkpoint))])  # Ajuste para pegar a chave correta
        return model.to(device)