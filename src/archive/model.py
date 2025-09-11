import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.autograd.profiler as profiler


class SpatialEncoderResNet(nn.Module):
    def __init__(self, input_channels=7):
        super(SpatialEncoderResNet, self).__init__()

        # Load pre-trained ResNet model (excluding the fully connected layers)
        self.resnet = models.resnet50(pretrained=True)

        # Modify the first convolution layer to accept the custom number of input channels
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Remove the fully connected layers
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove FC layers

        # Adaptive pooling to reduce the output feature map to a single value per channel
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)  # Pool to (1, 1)

    def forward(self, x):
        """
        x: Tensor of shape (B, L, C, H, W)
        B: Batch size
        L: Sequence length (e.g., 120 days)
        C: Number of input channels (e.g., 7 climate variables)
        H: Height of the watershed grid
        W: Width of the watershed grid
        """
        B, L, C, H, W = x.shape

        # Reshape input to (B * L, C, H, W) to process the entire sequence in parallel
        x_reshaped = x.view(B * L, C, H, W)  # (B * L, C, H, W)

        # Pass the reshaped tensor through ResNet (no loop needed)
        feature_map = self.resnet(x_reshaped)  # (B * L, 2048, H', W')

        # Apply adaptive pooling to reduce the feature map to (B * L, 2048, 1, 1)
        pooled_features = self.adaptive_pool(feature_map)  # (B * L, 2048, 1, 1)

        # Reshape back to (B, L, 2048, 1, 1)
        pooled_features = pooled_features.view(B, L, 2048, 1, 1)

        return pooled_features

class TemporalStreamflowPredictor(nn.Module):
    def __init__(self, input_dim, num_leads=4):
        super(TemporalStreamflowPredictor, self).__init__()

        # Temporal Fusion Transformer (TFT) or any transformer-based model for sequential data
        self.tft = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=256),
            num_layers=6
        )

        # Fully connected layer for lead-day prediction (K)
        self.fc = nn.Linear(input_dim, num_leads)

    def forward(self, spatial_features):
        """
        spatial_features: (B, L, C, 1, 1)
        """
        # Flatten spatial features into a sequence for the transformer
        B, L, C, H, W = spatial_features.shape
        spatial_features = spatial_features.view(B, L, C * H * W)  # Flatten spatial dimensions

        # Apply the transformer to capture temporal dependencies
        temporal_output = self.tft(spatial_features)  # (B, L, input_dim)

        # Output streamflow prediction for each lead
        output = self.fc(temporal_output[:, -1, :])  # Use the last timestep for prediction
        return output  # (B, num_leads)



class SpatioTemporalStreamflowModel(nn.Module):
    def __init__(self, seq_len=120, input_channels=7, num_leads=4):
        super(SpatioTemporalStreamflowModel, self).__init__()

        # Spatial encoder with pre-trained ResNet
        self.spatial_encoder = SpatialEncoderResNet(input_channels)

        # Temporal feature extraction (TFT or any transformer)
        self.temporal_predictor = TemporalStreamflowPredictor(input_dim=2048, num_leads=num_leads)

    # def forward(self, X):
    #     """
    #     X: Tensor of shape (B, L, C, H, W)
    #     """
    #     with profiler.profile(use_device='cuda') as prof:
    #         # Extract spatial features from the sequence using ResNet
    #         spatial_features = self.spatial_encoder(X)  # (B, L, 2048, 1, 1)
    #     print("Spatial Encoder profiling:")
    #     print(prof.key_averages().table(sort_by="cpu_time_total"))

    #     with profiler.profile(use_device='cuda') as prof:
    #         # Use the temporal predictor to make streamflow predictions
    #         output = self.temporal_predictor(spatial_features)
    #     print("Temporal Predictor profiling:")
    #     print(prof.key_averages().table(sort_by="cpu_time_total"))

    #     return output

    def forward(self, X):
        """
        X: Tensor of shape (B, L, C, H, W)
        """
        # Extract spatial features from the sequence using ResNet
        spatial_features = self.spatial_encoder(X)  # (B, L, 2048, 1, 1)

        # Use the temporal predictor to make streamflow predictions
        output = self.temporal_predictor(spatial_features)

        return output
