import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CCNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # Match the CNN structure from the table
        self.cnn = nn.Sequential(
            # First conv layer: Input -> 17,13,16
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            
            # Second conv layer: 17,13,16 -> 15,11,32
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.1),  # First dropout after second conv
            
            # Third conv layer: 15,11,32 -> 13,9,32
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.1)  # Second dropout after third conv
        )

        # Dynamically compute flattened output size
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = torch.flatten(self.cnn(sample), 1).shape[1]

        # Three fully connected layers as per the paper (64-256 neurons)
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        x = torch.flatten(x, 1)
        return self.fc(x)
