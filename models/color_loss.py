from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
# Import our color theory modules (assuming they're in the same package)
from color_theory_fashion import (  # SkinToneCategory,; Undertone,
    AdvancedColorAnalyzer, ColorProfile, ColorTheoryEngine,
    FashionColorRecommendationEngine)


@dataclass
class MLColorPrediction:
    """ML model prediction with color analysis"""

    garment_predictions: List[Dict]
    color_profile: Optional[ColorProfile]
    color_recommendations: Dict
    confidence_scores: Dict[str, float]


class ColorAwareFeatureExtractor(nn.Module):
    """
    Feature extractor that incorporates color theory knowledge
    """

    def __init__(self, backbone_features: int = 1024, color_features: int = 256):
        super().__init__()
        self.backbone_features = backbone_features
        self.color_features = color_features

        # Color-specific feature extractors
        self.color_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, color_features),
        )

        # Skin tone classification head
        self.skin_tone_classifier = nn.Sequential(
            nn.Linear(color_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 6),  # 6 Fitzpatrick types
        )

        # Undertone classification head
        self.undertone_classifier = nn.Sequential(
            nn.Linear(color_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),  # Warm, Cool, Neutral
        )

        # Color harmony prediction head
        self.harmony_predictor = nn.Sequential(
            nn.Linear(color_features * 2, 256),  # For color pairs
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # Complementary, Analogous, Triadic, Monochromatic
        )

    def forward(self, image: torch.Tensor, color_pairs: Optional[torch.Tensor] = None):
        """
        Forward pass with color-aware feature extraction

        Args:
            image: Input image tensor [B, C, H, W]
            color_pairs: Optional color pair tensor for harmony prediction [B, 2, 3]
        """
        batch_size = image.size(0)

        # Extract color features
        color_features = self.color_conv(image)

        # Skin tone prediction
        skin_tone_logits = self.skin_tone_classifier(color_features)
        skin_tone_probs = F.softmax(skin_tone_logits, dim=1)

        # Undertone prediction
        undertone_logits = self.undertone_classifier(color_features)
        undertone_probs = F.softmax(undertone_logits, dim=1)

        # Color harmony prediction (if color pairs provided)
        harmony_probs = None
        if color_pairs is not None:
            # Flatten color pairs and concatenate with color features
            color_pair_features = color_pairs.view(batch_size, -1)
            harmony_input = torch.cat([color_features, color_features], dim=1)
            harmony_logits = self.harmony_predictor(harmony_input)
            harmony_probs = F.softmax(harmony_logits, dim=1)

        return {
            "color_features": color_features,
            "skin_tone_probs": skin_tone_probs,
            "undertone_probs": undertone_probs,
            "harmony_probs": harmony_probs,
        }


class ColorTheoryLoss(nn.Module):
    """
    Custom loss function incorporating color theory principles
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.2, gamma: float = 0.1):
        super().__init__()
        self.alpha = alpha  # Weight for skin tone loss
        self.beta = beta  # Weight for undertone loss
        self.gamma = gamma  # Weight for harmony loss

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        Calculate multi-component color theory loss
        """
        losses = {}
        total_loss = 0

        # Skin tone classification loss
        if "skin_tone_probs" in predictions and "skin_tone_labels" in targets:
            skin_loss = self.ce_loss(
                predictions["skin_tone_probs"], targets["skin_tone_labels"]
            )
            losses["skin_tone_loss"] = skin_loss
            total_loss += self.alpha * skin_loss

        # Undertone classification loss
        if "undertone_probs" in predictions and "undertone_labels" in targets:
            undertone_loss = self.ce_loss(
                predictions["undertone_probs"], targets["undertone_labels"]
            )
            losses["undertone_loss"] = undertone_loss
            total_loss += self.beta * undertone_loss

        # Color harmony loss
        if "harmony_probs" in predictions and "harmony_labels" in targets:
            harmony_loss = self.ce_loss(
                predictions["harmony_probs"], targets["harmony_labels"]
            )
            losses["harmony_loss"] = harmony_loss
            total_loss += self.gamma * harmony_loss

        # Color consistency loss (custom)
        if "color_features" in predictions and "target_colors" in targets:
            color_consistency_loss = self._color_consistency_loss(
                predictions["color_features"], targets["target_colors"]
            )
            losses["color_consistency_loss"] = color_consistency_loss
            total_loss += 0.1 * color_consistency_loss

        losses["total_loss"] = total_loss
        return losses

    def _color_consistency_loss(
        self, color_features: torch.Tensor, target_colors: torch.Tensor
    ) -> torch.Tensor:
        """
        Custom loss to encourage color consistency based on color theory
        """
        # This would implement Delta E or other perceptual color distance metrics
        # For now, using simple MSE as placeholder
        return self.mse_loss(color_features, target_colors)


class MLColorAnalysisPipeline:
    """
    Complete ML pipeline integrating color theory with fashion analysis
    """

    def __init__(self, model_path: Optional[str] = None):
        self.color_engine = ColorTheoryEngine()
        self.advanced_analyzer = AdvancedColorAnalyzer()
        self.recommendation_engine = FashionColorRecommendationEngine()

        # ML model components
        self.feature_extractor = ColorAwareFeatureExtractor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path:
            self.load_model(model_path)

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_model(self, model_path: str):
        """Load pre-trained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint["model_state_dict"])
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for ML model"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB if needed
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0).to(self.device)

    def extract_face_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face region from image using OpenCV (placeholder)
        In practice, you'd use a more sophisticated face detector
        """
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]  # Take the first face
            return image[y : y + h, x : x + w]
        return None

    def analyze_image_complete(
        self,
        image: np.ndarray,
        existing_wardrobe: List[Dict],
        occasion: str = "casual",
        season: str = "spring",
    ) -> MLColorPrediction:
        """
        Run full analysis: image → color profile → recommendations
        """
        image_tensor = self.preprocess_image(image)
        face_region = self.extract_face_region(image)

        if face_region is None:
            raise ValueError("Face not detected in the image.")

        # Analyze skin and undertone using color theory
        skin_tone, skin_conf = self.color_engine.extract_skin_tone_from_face(
            face_region
        )
        undertone, undertone_conf = self.color_engine.detect_undertone(face_region)

        # Extract dominant outfit colors (placeholder logic)
        outfit_colors = self.color_engine.extract_dominant_colors(image, n_colors=3)

        # Build color profile
        color_profile = ColorProfile(
            skin_tone=skin_tone,
            undertone=undertone,
            dominant_colors=outfit_colors,
            undertone_confidence=undertone_conf,
            skin_tone_confidence=skin_conf,
        )

        # Generate recommendations
        recommendations = self.recommendation_engine.generate_outfit_color_suggestions(
            color_profile=self.recommendation_engine.analyze_complete_color_profile(
                face_region
            ),
            existing_garments=existing_wardrobe,
            target_occasion=occasion,
            season=season,
        )

        return MLColorPrediction(
            garment_predictions=existing_wardrobe,
            color_profile=color_profile,
            color_recommendations=recommendations,
            confidence_scores={
                "skin_tone_confidence": skin_conf,
                "undertone_confidence": undertone_conf,
            },
        )
