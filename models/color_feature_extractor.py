import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorAwareFeatureExtractor(nn.Module):
    def __init__(self, backbone_features: int = 1024, color_features: int = 256):
        super().__init__()
        self.backbone_features = backbone_features
        self.color_features = color_features

        # Basic CNN to extract color-related features
        self.color_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, color_features),
        )

        # Skin tone head (6 classes for Fitzpatrick types)
        self.skin_tone_classifier = nn.Sequential(
            nn.Linear(color_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 6)
        )

        # Undertone head (3 classes)
        self.undertone_classifier = nn.Sequential(
            nn.Linear(color_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, image: torch.Tensor):
        """
        Forward pass

        Args:
            image (Tensor): Shape (B, 3, H, W)

        Returns:
            Dict with predictions and color features
        """
        color_features = self.color_conv(image)

        skin_logits = self.skin_tone_classifier(color_features)
        undertone_logits = self.undertone_classifier(color_features)

        return {
            "color_features": color_features,
            "skin_tone_probs": F.softmax(skin_logits, dim=1),
            "undertone_probs": F.softmax(undertone_logits, dim=1)
        }
