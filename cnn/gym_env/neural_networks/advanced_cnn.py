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
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Second conv layer: 17,13,16 -> 15,11,32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),  # First dropout after second conv
            
            # Third conv layer: 15,11,32 -> 13,9,32
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Use Global Average Pooling
        # It will output a tensor of shape (batch_size, 32)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # The input to the first fully connected layer is now the number of channels
        # from the last conv layer (which is 32).
        n_flatten = 32
                                                                                            
        # Fully connected layers
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
        # Pass through CNN
        cnn_features = self.cnn(observations)
        # Apply Global Average Pooling
        pooled_features = self.global_avg_pool(cnn_features)
        # Flatten the pooled features
        flattened_features = torch.flatten(pooled_features, 1)
        # Pass through fully connected layers
        return self.fc(flattened_features)
