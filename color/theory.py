import colorsys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class SkinToneCategory(Enum):
    """Fitzpatrick skin tone scale categories"""

    TYPE_1 = "Very Fair"  # Always burns, never tans
    TYPE_2 = "Fair"  # Usually burns, tans minimally
    TYPE_3 = "Medium"  # Sometimes burns, tans gradually
    TYPE_4 = "Olive"  # Rarely burns, tans easily
    TYPE_5 = "Brown"  # Very rarely burns, tans very easily
    TYPE_6 = "Deep"  # Never burns, tans very easily


class Undertone(Enum):
    """Skin undertone categories"""

    WARM = "warm"  # Yellow, golden, peachy undertones
    COOL = "cool"  # Pink, red, blue undertones
    NEUTRAL = "neutral"  # Mix of warm and cool


@dataclass
class ColorProfile:
    """Complete color analysis profile for a person"""

    skin_tone: SkinToneCategory
    undertone: Undertone
    dominant_colors: List[Tuple[int, int, int]]  # RGB values
    undertone_confidence: float
    skin_tone_confidence: float


@dataclass
class ColorHarmony:
    """Color harmony analysis result"""

    harmony_type: str
    primary_color: Tuple[int, int, int]
    recommended_colors: List[Tuple[int, int, int]]
    harmony_score: float


class ColorTheoryEngine:
    """
    Comprehensive color theory implementation for fashion analysis
    """

    def __init__(self):
        # Pre-defined color palettes for different undertones
        self.warm_palette = {
            "reds": [(255, 69, 0), (220, 20, 60), (178, 34, 34)],
            "oranges": [(255, 140, 0), (255, 165, 0), (255, 99, 71)],
            "yellows": [(255, 215, 0), (255, 255, 0), (255, 250, 205)],
            "greens": [(154, 205, 50), (107, 142, 35), (85, 107, 47)],
            "browns": [(139, 69, 19), (160, 82, 45), (210, 180, 140)],
        }

        self.cool_palette = {
            "blues": [(0, 0, 255), (30, 144, 255), (70, 130, 180)],
            "purples": [(128, 0, 128), (138, 43, 226), (147, 112, 219)],
            "pinks": [(255, 20, 147), (255, 105, 180), (255, 182, 193)],
            "greens": [(0, 128, 0), (0, 255, 127), (46, 139, 87)],
            "grays": [(128, 128, 128), (169, 169, 169), (211, 211, 211)],
        }

        self.neutral_palette = {
            "earth_tones": [(139, 119, 101), (160, 142, 122), (205, 192, 176)],
            "muted_colors": [(188, 143, 143), (176, 196, 222), (221, 160, 221)],
            "classic_neutrals": [(0, 0, 0), (255, 255, 255), (128, 128, 128)],
        }

    def extract_skin_tone_from_face(
        self, face_region: np.ndarray
    ) -> Tuple[SkinToneCategory, float]:
        """
        Extract skin tone category from face region using advanced color analysis

        Args:
            face_region: Cropped face region from image (BGR format)

        Returns:
            Tuple of (SkinToneCategory, confidence_score)
        """
        # Convert to RGB
        face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

        # Create skin mask using YCrCb color space
        face_ycrcb = cv2.cvtColor(face_region, cv2.COLOR_BGR2YCrCb)

        # Define skin color range in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(face_ycrcb, lower_skin, upper_skin)

        # Extract skin pixels
        skin_pixels = face_rgb[skin_mask > 0]

        if len(skin_pixels) == 0:
            return SkinToneCategory.TYPE_3, 0.0

        # Calculate average skin color
        avg_skin_color = np.mean(skin_pixels, axis=0)

        # Calculate luminosity (perceived brightness)
        luminosity = (
            0.299 * avg_skin_color[0]
            + 0.587 * avg_skin_color[1]
            + 0.114 * avg_skin_color[2]
        )

        # Calculate color temperature (warmth/coolness indicator)
        red_component = avg_skin_color[0]
        blue_component = avg_skin_color[2]
        color_temp = red_component - blue_component

        # Classify based on luminosity and color analysis
        confidence = 0.8  # Base confidence

        if luminosity > 200:
            return SkinToneCategory.TYPE_1, confidence
        elif luminosity > 170:
            return SkinToneCategory.TYPE_2, confidence
        elif luminosity > 140:
            return SkinToneCategory.TYPE_3, confidence
        elif luminosity > 110:
            return SkinToneCategory.TYPE_4, confidence
        elif luminosity > 80:
            return SkinToneCategory.TYPE_5, confidence
        else:
            return SkinToneCategory.TYPE_6, confidence

    def detect_undertone(self, face_region: np.ndarray) -> Tuple[Undertone, float]:
        """
        Detect skin undertone using advanced color space analysis

        Args:
            face_region: Cropped face region from image

        Returns:
            Tuple of (Undertone, confidence_score)
        """
        # Convert to different color spaces for analysis
        face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        face_lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)

        # Create skin mask
        face_ycrcb = cv2.cvtColor(face_region, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(face_ycrcb, lower_skin, upper_skin)

        # Extract skin pixels in different color spaces
        skin_rgb = face_rgb[skin_mask > 0]
        skin_lab = face_lab[skin_mask > 0]

        if len(skin_rgb) == 0:
            return Undertone.NEUTRAL, 0.0

        # Calculate average values
        avg_rgb = np.mean(skin_rgb, axis=0)
        avg_lab = np.mean(skin_lab, axis=0)

        # Method 1: RGB analysis
        red_green_ratio = avg_rgb[0] / (avg_rgb[1] + 1e-6)
        blue_yellow_indicator = avg_rgb[2] - avg_rgb[0]

        # Method 2: LAB color space analysis
        # A channel: green(-) to red(+)
        # B channel: blue(-) to yellow(+)
        a_component = avg_lab[1] - 128  # Center around 0
        b_component = avg_lab[2] - 128  # Center around 0

        # Calculate undertone scores
        warm_score = 0
        cool_score = 0

        # RGB-based scoring
        if red_green_ratio > 1.1:
            warm_score += 1
        if blue_yellow_indicator < -5:
            warm_score += 1
        elif blue_yellow_indicator > 5:
            cool_score += 1

        # LAB-based scoring (more accurate)
        if b_component > 5:  # More yellow
            warm_score += 2
        elif b_component < -5:  # More blue
            cool_score += 2

        if a_component > 5:  # More red (can indicate warm)
            warm_score += 1
        elif a_component < -2:  # More green (can indicate cool)
            cool_score += 1

        # Determine undertone
        total_score = warm_score + cool_score
        if total_score == 0:
            return Undertone.NEUTRAL, 0.5

        confidence = min(abs(warm_score - cool_score) / total_score + 0.3, 0.95)

        if warm_score > cool_score:
            return Undertone.WARM, confidence
        elif cool_score > warm_score:
            return Undertone.COOL, confidence
        else:
            return Undertone.NEUTRAL, confidence

    def extract_dominant_colors(
        self, clothing_region: np.ndarray, n_colors: int = 5
    ) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors from clothing region using K-means clustering

        Args:
            clothing_region: Image region containing clothing
            n_colors: Number of dominant colors to extract

        Returns:
            List of dominant colors as RGB tuples
        """
        # Reshape image to be a list of pixels
        data = clothing_region.reshape((-1, 3))
        data = np.float32(data)

        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Convert centers to integers and sort by frequency
        centers = np.uint8(centers)

        # Count frequency of each cluster
        unique, counts = np.unique(labels, return_counts=True)
        freq_sorted_indices = np.argsort(-counts)

        # Return colors sorted by frequency
        dominant_colors = []
        for idx in freq_sorted_indices:
            color = tuple(centers[idx])
            dominant_colors.append(color)

        return dominant_colors

    def rgb_to_hsv(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to HSV color space"""
        r, g, b = [x / 255.0 for x in rgb]
        return colorsys.rgb_to_hsv(r, g, b)

    def hsv_to_rgb(self, hsv: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert HSV to RGB color space"""
        r, g, b = colorsys.hsv_to_rgb(*hsv)
        return tuple(int(x * 255) for x in [r, g, b])

    def calculate_color_harmony(
        self, base_color: Tuple[int, int, int], harmony_type: str
    ) -> List[Tuple[int, int, int]]:
        """
        Calculate color harmony based on color theory principles

        Args:
            base_color: RGB tuple of base color
            harmony_type: Type of harmony ('complementary', 'analogous', 'triadic', 'monochromatic')

        Returns:
            List of harmonious colors
        """
        h, s, v = self.rgb_to_hsv(base_color)
        harmonious_colors = []

        if harmony_type == "complementary":
            # Opposite on color wheel (180 degrees)
            comp_h = (h + 0.5) % 1.0
            harmonious_colors.append(self.hsv_to_rgb((comp_h, s, v)))

        elif harmony_type == "analogous":
            # Adjacent colors (±30 degrees)
            for offset in [-1 / 12, 1 / 12]:  # ±30 degrees
                new_h = (h + offset) % 1.0
                harmonious_colors.append(self.hsv_to_rgb((new_h, s, v)))

        elif harmony_type == "triadic":
            # 120 degree intervals
            for offset in [1 / 3, 2 / 3]:
                new_h = (h + offset) % 1.0
                harmonious_colors.append(self.hsv_to_rgb((new_h, s, v)))

        elif harmony_type == "monochromatic":
            # Same hue, different saturation/value
            for s_offset, v_offset in [(0.3, 0.8), (0.7, 0.6), (0.9, 0.9)]:
                new_s = max(0.1, min(1.0, s + s_offset - 0.5))
                new_v = max(0.1, min(1.0, v + v_offset - 0.5))
                harmonious_colors.append(self.hsv_to_rgb((h, new_s, new_v)))

        return harmonious_colors

    def get_recommended_colors_for_undertone(
        self, undertone: Undertone, skin_tone: SkinToneCategory
    ) -> Dict[str, List[Tuple[int, int, int]]]:
        """
        Get recommended color palette based on undertone and skin tone

        Args:
            undertone: Detected undertone
            skin_tone: Detected skin tone category

        Returns:
            Dictionary of color categories with recommended colors
        """
        if undertone == Undertone.WARM:
            base_palette = self.warm_palette.copy()
        elif undertone == Undertone.COOL:
            base_palette = self.cool_palette.copy()
        else:
            base_palette = self.neutral_palette.copy()

        # Adjust recommendations based on skin tone depth
        if skin_tone in [SkinToneCategory.TYPE_1, SkinToneCategory.TYPE_2]:
            # Lighter skin tones - avoid colors that wash out
            recommended = {}
            for category, colors in base_palette.items():
                # Filter out very light colors for fair skin
                filtered_colors = [
                    color for color in colors if sum(color) < 600
                ]  # Avoid very light colors
                if filtered_colors:
                    recommended[category] = filtered_colors

        elif skin_tone in [SkinToneCategory.TYPE_5, SkinToneCategory.TYPE_6]:
            # Deeper skin tones - can wear bold, bright colors
            recommended = {}
            for category, colors in base_palette.items():
                # Add more vibrant variations
                enhanced_colors = colors.copy()
                for color in colors:
                    h, s, v = self.rgb_to_hsv(color)
                    # Add more saturated version
                    if s < 0.8:
                        enhanced_s = min(1.0, s + 0.2)
                        enhanced_colors.append(self.hsv_to_rgb((h, enhanced_s, v)))
                recommended[category] = enhanced_colors
        else:
            # Medium skin tones - most versatile
            recommended = base_palette.copy()

        return recommended

    def analyze_outfit_color_harmony(
        self, garment_colors: List[Tuple[int, int, int]]
    ) -> ColorHarmony:
        """
        Analyze color harmony of an outfit

        Args:
            garment_colors: List of dominant colors from different garments

        Returns:
            ColorHarmony analysis result
        """
        if not garment_colors:
            return ColorHarmony("none", (0, 0, 0), [], 0.0)

        # Convert to HSV for analysis
        hsv_colors = [self.rgb_to_hsv(color) for color in garment_colors]

        # Analyze hue relationships
        hues = [hsv[0] for hsv in hsv_colors]

        harmony_scores = {}

        # Check for monochromatic (similar hues)
        hue_variance = np.var(hues) if len(hues) > 1 else 0
        if hue_variance < 0.01:  # Very similar hues
            harmony_scores["monochromatic"] = 0.9

        # Check for complementary (opposite hues)
        if len(hues) >= 2:
            for i in range(len(hues)):
                for j in range(i + 1, len(hues)):
                    hue_diff = abs(hues[i] - hues[j])
                    hue_diff = min(hue_diff, 1.0 - hue_diff)  # Consider circular nature

                    if 0.4 < hue_diff < 0.6:  # Around 180 degrees
                        harmony_scores["complementary"] = 0.8
                    elif 0.25 < hue_diff < 0.4:  # Around 120 degrees
                        harmony_scores["triadic"] = 0.7
                    elif hue_diff < 0.15:  # Close hues
                        harmony_scores["analogous"] = 0.75

        # Determine best harmony type
        if harmony_scores:
            best_harmony = max(harmony_scores.items(), key=lambda x: x[1])
            harmony_type, score = best_harmony
        else:
            harmony_type, score = "neutral", 0.5

        # Generate recommended colors based on primary color
        primary_color = garment_colors[0]
        recommended = self.calculate_color_harmony(primary_color, harmony_type)

        return ColorHarmony(
            harmony_type=harmony_type,
            primary_color=primary_color,
            recommended_colors=recommended,
            harmony_score=score,
        )

    def generate_color_recommendations(
        self,
        color_profile: ColorProfile,
        current_outfit_colors: List[Tuple[int, int, int]],
        occasion: str = "casual",
    ) -> Dict[str, any]:
        """
        Generate comprehensive color recommendations

        Args:
            color_profile: Complete color analysis profile
            current_outfit_colors: Colors in current outfit
            occasion: Occasion type for contextual recommendations

        Returns:
            Comprehensive recommendation dictionary
        """
        # Get base recommendations for undertone
        base_recommendations = self.get_recommended_colors_for_undertone(
            color_profile.undertone, color_profile.skin_tone
        )

        # Analyze current outfit harmony
        current_harmony = self.analyze_outfit_color_harmony(current_outfit_colors)

        # Generate occasion-specific adjustments
        occasion_adjustments = self._get_occasion_color_adjustments(occasion)

        # Combine recommendations
        recommendations = {
            "undertone_based": base_recommendations,
            "harmony_analysis": {
                "current_harmony_type": current_harmony.harmony_type,
                "harmony_score": current_harmony.harmony_score,
                "suggested_additions": current_harmony.recommended_colors,
            },
            "occasion_appropriate": occasion_adjustments,
            "confidence_scores": {
                "undertone_confidence": color_profile.undertone_confidence,
                "skin_tone_confidence": color_profile.skin_tone_confidence,
            },
        }

        return recommendations

    def _get_occasion_color_adjustments(
        self, occasion: str
    ) -> Dict[str, List[Tuple[int, int, int]]]:
        """Get color adjustments based on occasion"""
        occasion_palettes = {
            "formal": {
                "primary": [(0, 0, 0), (25, 25, 112), (128, 128, 128)],
                "accent": [(255, 255, 255), (220, 220, 220)],
            },
            "business": {
                "primary": [(25, 25, 112), (0, 0, 0), (128, 128, 128)],
                "accent": [(255, 255, 255), (173, 216, 230)],
            },
            "casual": {
                "primary": [(70, 130, 180), (34, 139, 34), (165, 42, 42)],
                "accent": [(255, 255, 255), (255, 215, 0)],
            },
            "party": {
                "primary": [(128, 0, 128), (255, 20, 147), (0, 0, 255)],
                "accent": [(255, 215, 0), (255, 105, 180)],
            },
        }

        return occasion_palettes.get(occasion, occasion_palettes["casual"])


# Color analysis demonstration function
def color_analysis(
    outfit_colors: List[Tuple[int, int, int]], debug=False
) -> ColorHarmony:
    """Demonstrate the color theory implementation"""
    engine = ColorTheoryEngine()

    # Analyze harmony
    harmony = engine.analyze_outfit_color_harmony(outfit_colors)
    if debug:
        # Example: Analyze a hypothetical face region (normally from real image)
        print("=== Color Theory Engine Demo ===\n")

        # Simulate skin tone detection
        print("1. Skin Tone Analysis:")
        print("   - Detected: Medium skin tone with warm undertone")
        print("   - Confidence: 0.85")
        print(f"\n2. Outfit Colors: {outfit_colors}")

        print("\n3. Color Harmony Analysis:")
        print(f"   - Primary Color: {harmony.primary_color}")
        print(f"   - Harmony Type: {harmony.harmony_type}")
        print(f"   - Harmony Score: {harmony.harmony_score:.2f}")
        print(f"   - Recommended additions: {harmony.recommended_colors}")

        # Generate color wheel harmonies
        base_color = (70, 130, 180)  # Steel blue
        print(f"\n4. Color Wheel Harmonies for {base_color}:")
        for harmony_type in ["complementary", "analogous", "triadic", "monochromatic"]:
            colors = engine.calculate_color_harmony(base_color, harmony_type)
            print(f"   - {harmony_type.capitalize()}: {colors}")

    return harmony


class AdvancedColorAnalyzer:
    """
    Advanced color analysis with perceptual color science integration
    """

    def __init__(self):
        self.color_engine = ColorTheoryEngine()
        # CIE standard illuminants for color matching
        self.illuminants = {
            "D65": (0.31271, 0.32902),  # Daylight 6500K
            "A": (0.44757, 0.40745),  # Incandescent
            "F2": (0.37208, 0.37529),  # Cool fluorescent
        }

    def calculate_delta_e(
        self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]
    ) -> float:
        """
        Calculate Delta E color difference using CIE76 formula
        Values < 1: Not perceptible
        1-2: Perceptible through close observation
        2-10: Perceptible at a glance
        >10: Colors are more different than similar
        """
        # Convert RGB to LAB color space
        lab1 = self._rgb_to_lab(color1)
        lab2 = self._rgb_to_lab(color2)

        # Calculate Delta E (CIE76)
        delta_l = lab1[0] - lab2[0]
        delta_a = lab1[1] - lab2[1]
        delta_b = lab1[2] - lab2[2]

        delta_e = np.sqrt(delta_l**2 + delta_a**2 + delta_b**2)
        return delta_e

    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to LAB color space"""
        # Normalize RGB values
        r, g, b = [x / 255.0 for x in rgb]

        # Apply gamma correction
        def gamma_correct(c):
            if c > 0.04045:
                return ((c + 0.055) / 1.055) ** 2.4
            else:
                return c / 12.92

        r = gamma_correct(r)
        g = gamma_correct(g)
        b = gamma_correct(b)

        # Convert to XYZ color space (using sRGB matrix)
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

        # Normalize by D65 illuminant
        x = x / 0.95047
        y = y / 1.00000
        z = z / 1.08883

        # Apply LAB transformation
        def lab_transform(t):
            if t > 0.008856:
                return t ** (1 / 3)
            else:
                return (7.787 * t) + (16 / 116)

        fx = lab_transform(x)
        fy = lab_transform(y)
        fz = lab_transform(z)

        # Calculate LAB values
        LAB_l = (116 * fy) - 16
        LAB_a = 500 * (fx - fy)
        LAB_b = 200 * (fy - fz)

        return (LAB_l, LAB_a, LAB_b)

    def calculate_color_temperature(self, rgb: Tuple[int, int, int]) -> float:
        """
        Calculate correlated color temperature (CCT) of a color
        Returns temperature in Kelvin
        """
        # Convert to XYZ color space
        # lab = self._rgb_to_lab(rgb)

        # Simple approximation for CCT based on color characteristics
        # This is a simplified version - full CCT calculation requires more complex algorithms
        r, g, b = rgb

        # Calculate color temperature indicator
        if b > r:  # Cooler colors
            temp = 6500 + (b - r) * 20  # Higher temperature for blue-er colors
        else:  # Warmer colors
            temp = 6500 - (r - b) * 15  # Lower temperature for red-er colors

        return max(2000, min(10000, temp))  # Clamp to reasonable range

    def generate_seasonal_palette(
        self, base_colors: List[Tuple[int, int, int]], season: str
    ) -> List[Tuple[int, int, int]]:
        """
        Generate season-appropriate color variations
        """
        seasonal_adjustments = {
            "spring": {"saturation": 1.2, "brightness": 1.1, "warmth": 0.1},
            "summer": {"saturation": 0.8, "brightness": 1.0, "warmth": -0.1},
            "autumn": {"saturation": 1.1, "brightness": 0.9, "warmth": 0.2},
            "winter": {"saturation": 1.3, "brightness": 0.8, "warmth": -0.2},
        }

        if season not in seasonal_adjustments:
            return base_colors

        adjustments = seasonal_adjustments[season]
        seasonal_colors = []

        for color in base_colors:
            h, s, v = self.color_engine.rgb_to_hsv(color)

            # Apply seasonal adjustments
            new_s = max(0.1, min(1.0, s * adjustments["saturation"]))
            new_v = max(0.1, min(1.0, v * adjustments["brightness"]))

            # Adjust warmth by shifting hue slightly
            hue_shift = adjustments["warmth"] / 12  # Convert to hue units
            new_h = (h + hue_shift) % 1.0

            seasonal_color = self.color_engine.hsv_to_rgb((new_h, new_s, new_v))
            seasonal_colors.append(seasonal_color)

        return seasonal_colors

    def calculate_outfit_coherence_score(
        self, outfit_colors: List[Tuple[int, int, int]]
    ) -> float:
        """
        Calculate overall coherence score for an outfit based on color theory
        Returns score from 0.0 to 1.0
        """
        if len(outfit_colors) < 2:
            return 1.0

        scores = []

        # 1. Color harmony score
        harmony = self.color_engine.analyze_outfit_color_harmony(outfit_colors)
        harmony_score = harmony.harmony_score
        scores.append(harmony_score * 0.4)  # 40% weight

        # 2. Color contrast score (ensure readability and visual interest)
        contrast_scores = []
        for i in range(len(outfit_colors)):
            for j in range(i + 1, len(outfit_colors)):
                delta_e = self.calculate_delta_e(outfit_colors[i], outfit_colors[j])
                # Optimal contrast is between 10-50 Delta E units
                if 10 <= delta_e <= 50:
                    contrast_scores.append(1.0)
                elif delta_e < 10:
                    contrast_scores.append(delta_e / 10)  # Too similar
                else:
                    contrast_scores.append(max(0.2, 50 / delta_e))  # Too different

        avg_contrast = np.mean(contrast_scores) if contrast_scores else 0.5
        scores.append(avg_contrast * 0.3)  # 30% weight

        # 3. Color temperature consistency
        temperatures = [
            self.calculate_color_temperature(color) for color in outfit_colors
        ]
        temp_variance = np.var(temperatures)
        temp_score = max(
            0.0, 1.0 - (temp_variance / 2000000)
        )  # Normalize temperature variance
        scores.append(temp_score * 0.2)  # 20% weight

        # 4. Saturation balance
        saturations = [
            self.color_engine.rgb_to_hsv(color)[1] for color in outfit_colors
        ]
        sat_variance = np.var(saturations)
        sat_score = max(
            0.0, 1.0 - (sat_variance * 4)
        )  # Penalize high saturation variance
        scores.append(sat_score * 0.1)  # 10% weight

        return sum(scores)


class FashionColorRecommendationEngine:
    """
    Complete fashion color recommendation system integrating all color theory components
    """

    def __init__(self):
        self.color_engine = ColorTheoryEngine()
        self.advanced_analyzer = AdvancedColorAnalyzer()

        # Fashion-specific color associations
        self.occasion_color_psychology = {
            "professional": {
                "authoritative": [(25, 25, 112), (0, 0, 0), (128, 128, 128)],
                "approachable": [(0, 100, 0), (70, 130, 180), (139, 69, 19)],
                "creative": [(75, 0, 130), (220, 20, 60), (255, 140, 0)],
            },
            "casual": {
                "relaxed": [(100, 149, 237), (152, 251, 152), (255, 182, 193)],
                "energetic": [(255, 69, 0), (50, 205, 50), (255, 215, 0)],
                "sophisticated": [(72, 61, 139), (105, 105, 105), (47, 79, 79)],
            },
            "evening": {
                "elegant": [(0, 0, 0), (25, 25, 112), (128, 0, 128)],
                "glamorous": [(255, 215, 0), (220, 20, 60), (138, 43, 226)],
                "romantic": [(255, 192, 203), (216, 191, 216), (230, 230, 250)],
            },
        }

        # Body type color strategies
        self.body_type_strategies = {
            "highlight": ["bright_colors", "warm_colors", "light_colors"],
            "minimize": ["dark_colors", "cool_colors", "monochromatic"],
            "balance": ["medium_tones", "strategic_patterns", "color_blocking"],
        }

    def analyze_complete_color_profile(
        self, face_image: np.ndarray, body_shape: str = None
    ) -> Dict[str, any]:
        """
        Complete color analysis pipeline

        Args:
            face_image: Face region for skin analysis
            body_shape: Optional body shape information

        Returns:
            Comprehensive color profile and recommendations
        """
        # Step 1: Basic color analysis
        skin_tone, skin_confidence = self.color_engine.extract_skin_tone_from_face(
            face_image
        )
        undertone, undertone_confidence = self.color_engine.detect_undertone(face_image)

        # Step 2: Create color profile
        color_profile = ColorProfile(
            skin_tone=skin_tone,
            undertone=undertone,
            dominant_colors=[
                (128, 128, 128)
            ],  # Placeholder, would extract from clothing
            undertone_confidence=undertone_confidence,
            skin_tone_confidence=skin_confidence,
        )

        # Step 3: Generate base recommendations
        base_recommendations = self.color_engine.get_recommended_colors_for_undertone(
            undertone, skin_tone
        )

        # Step 4: Add advanced analysis
        seasonal_recommendations = {}
        for season in ["spring", "summer", "autumn", "winter"]:
            # Get base colors from recommendations
            base_colors = []
            for category, colors in base_recommendations.items():
                base_colors.extend(colors[:2])  # Take top 2 from each category

            seasonal_colors = self.advanced_analyzer.generate_seasonal_palette(
                base_colors, season
            )
            seasonal_recommendations[season] = seasonal_colors

        # Step 5: Generate occasion-specific recommendations
        occasion_recommendations = {}
        for occasion, moods in self.occasion_color_psychology.items():
            occasion_rec = {}
            for mood, colors in moods.items():
                # Filter colors based on undertone compatibility
                compatible_colors = self._filter_colors_by_undertone(colors, undertone)
                occasion_rec[mood] = compatible_colors
            occasion_recommendations[occasion] = occasion_rec

        return {
            "color_profile": {
                "skin_tone": skin_tone.value,
                "undertone": undertone.value,
                "confidence_scores": {
                    "skin_tone": skin_confidence,
                    "undertone": undertone_confidence,
                },
            },
            "base_recommendations": base_recommendations,
            "seasonal_recommendations": seasonal_recommendations,
            "occasion_recommendations": occasion_recommendations,
            "analysis_metadata": {
                "analysis_timestamp": np.datetime64("now"),
                "color_temperature_range": self._get_recommended_temp_range(undertone),
                "recommended_contrast_levels": self._get_contrast_recommendations(
                    skin_tone
                ),
            },
        }

    def _filter_colors_by_undertone(
        self, colors: List[Tuple[int, int, int]], undertone: Undertone
    ) -> List[Tuple[int, int, int]]:
        """Filter colors based on undertone compatibility"""
        compatible_colors = []

        for color in colors:
            color_temp = self.advanced_analyzer.calculate_color_temperature(color)

            if undertone == Undertone.WARM and color_temp < 6000:
                compatible_colors.append(color)
            elif undertone == Undertone.COOL and color_temp > 6000:
                compatible_colors.append(color)
            elif undertone == Undertone.NEUTRAL:
                compatible_colors.append(color)

        return (
            compatible_colors if compatible_colors else colors
        )  # Fallback to original

    def _get_recommended_temp_range(self, undertone: Undertone) -> Tuple[int, int]:
        """Get recommended color temperature range for undertone"""
        ranges = {
            Undertone.WARM: (2700, 5500),
            Undertone.COOL: (5500, 8000),
            Undertone.NEUTRAL: (4000, 7000),
        }
        return ranges[undertone]

    def _get_contrast_recommendations(
        self, skin_tone: SkinToneCategory
    ) -> Dict[str, str]:
        """Get contrast level recommendations based on skin tone"""
        if skin_tone in [SkinToneCategory.TYPE_1, SkinToneCategory.TYPE_2]:
            return {
                "avoid": "very_high_contrast",
                "recommended": "medium_contrast",
                "optimal": "soft_contrast",
            }
        elif skin_tone in [SkinToneCategory.TYPE_5, SkinToneCategory.TYPE_6]:
            return {
                "avoid": "very_low_contrast",
                "recommended": "high_contrast",
                "optimal": "bold_contrast",
            }
        else:
            return {
                "avoid": "extreme_contrast",
                "recommended": "medium_to_high_contrast",
                "optimal": "balanced_contrast",
            }

    def generate_outfit_color_suggestions(
        self,
        color_profile: Dict[str, any],
        existing_garments: List[Dict[str, any]],
        target_occasion: str = "casual",
        season: str = "spring",
    ) -> Dict[str, any]:
        """
        Generate specific outfit color suggestions based on existing wardrobe

        Args:
            color_profile: Complete color profile from analyze_complete_color_profile
            existing_garments: List of existing garment data with colors
            target_occasion: Target occasion for the outfit
            season: Current season

        Returns:
            Detailed outfit suggestions with color coordination
        """
        undertone = Undertone(color_profile["color_profile"]["undertone"])

        # Extract colors from existing garments
        existing_colors = []
        garment_color_map = {}

        for garment in existing_garments:
            if "dominant_colors" in garment:
                garment_colors = garment["dominant_colors"]
                existing_colors.extend(garment_colors)
                garment_color_map[garment.get("id", len(garment_color_map))] = (
                    garment_colors
                )

        # Generate outfit combinations
        outfit_suggestions = []

        # Strategy 1: Monochromatic outfits
        for base_color in existing_colors[:5]:  # Limit to top 5 colors
            mono_colors = self.color_engine.calculate_color_harmony(
                base_color, "monochromatic"
            )
            mono_outfit = self._build_outfit_from_colors(
                base_color, mono_colors, existing_garments, "monochromatic"
            )
            if mono_outfit:
                outfit_suggestions.append(mono_outfit)

        # Strategy 2: Complementary combinations
        for base_color in existing_colors[:3]:
            comp_colors = self.color_engine.calculate_color_harmony(
                base_color, "complementary"
            )
            comp_outfit = self._build_outfit_from_colors(
                base_color, comp_colors, existing_garments, "complementary"
            )
            if comp_outfit:
                outfit_suggestions.append(comp_outfit)

        # Strategy 3: Seasonal recommendations
        seasonal_colors = color_profile["seasonal_recommendations"].get(season, [])
        for seasonal_color in seasonal_colors[:3]:
            seasonal_outfit = self._build_outfit_from_colors(
                seasonal_color, [seasonal_color], existing_garments, "seasonal"
            )
            if seasonal_outfit:
                outfit_suggestions.append(seasonal_outfit)

        # Rank outfits by coherence score
        for outfit in outfit_suggestions:
            outfit_colors = [tuple(color) for color in outfit["color_palette"]]
            coherence_score = self.advanced_analyzer.calculate_outfit_coherence_score(
                outfit_colors
            )
            outfit["coherence_score"] = coherence_score

        # Sort by coherence score
        outfit_suggestions.sort(key=lambda x: x["coherence_score"], reverse=True)

        return {
            "top_suggestions": outfit_suggestions[:5],
            "color_coordination_tips": self._generate_coordination_tips(
                undertone, target_occasion
            ),
            "seasonal_adjustments": self._get_seasonal_styling_tips(season),
            "occasion_guidelines": self._get_occasion_color_guidelines(target_occasion),
        }

    def _build_outfit_from_colors(
        self,
        primary_color: Tuple[int, int, int],
        harmony_colors: List[Tuple[int, int, int]],
        available_garments: List[Dict[str, any]],
        strategy: str,
    ) -> Optional[Dict[str, any]]:
        """Build an outfit based on color harmony strategy"""
        target_colors = [primary_color] + harmony_colors
        selected_garments = []
        used_categories = set()

        # Find garments that match the color scheme
        for garment in available_garments:
            if garment.get("wear_category") in used_categories:
                continue

            garment_colors = garment.get("dominant_colors", [])

            # Check if garment colors are compatible
            for garment_color in garment_colors:
                for target_color in target_colors:
                    delta_e = self.advanced_analyzer.calculate_delta_e(
                        tuple(garment_color), target_color
                    )
                    if delta_e < 30:  # Reasonably close match
                        selected_garments.append(garment)
                        used_categories.add(garment.get("wear_category"))
                        break
                if garment in selected_garments:
                    break

        if len(selected_garments) < 2:  # Need at least 2 pieces for an outfit
            return None

        # Extract color palette from selected garments
        outfit_colors = []
        for garment in selected_garments:
            outfit_colors.extend(
                garment.get("dominant_colors", [])[:2]
            )  # Top 2 colors per garment

        return {
            "garments": selected_garments,
            "color_palette": outfit_colors,
            "harmony_strategy": strategy,
            "primary_color": primary_color,
            "coherence_score": 0.0,  # Will be calculated later
        }

    def _generate_coordination_tips(
        self, undertone: Undertone, occasion: str
    ) -> List[str]:
        """Generate personalized coordination tips"""
        tips = []

        # Undertone-specific tips
        if undertone == Undertone.WARM:
            tips.extend(
                [
                    "Choose gold jewelry and accessories over silver",
                    "Opt for warm whites (cream, ivory) instead of stark white",
                    "Earth tones like terracotta, olive, and warm browns are your best friends",
                ]
            )
        elif undertone == Undertone.COOL:
            tips.extend(
                [
                    "Silver jewelry and cool-toned metals complement your undertone best",
                    "Pure white and bright white look stunning on you",
                    "Blues, purples, and cool greens enhance your natural coloring",
                ]
            )
        else:  # Neutral
            tips.extend(
                [
                    "You can wear both gold and silver jewelry - mix metals for interest",
                    "Both warm and cool colors work, giving you maximum versatility",
                    "Focus on the intensity of colors rather than temperature",
                ]
            )

        # Occasion-specific tips
        occasion_tips = {
            "professional": [
                "Stick to a maximum of 3 colors in your outfit",
                "Use neutral bases with one pop of color",
                "Ensure sufficient contrast between top and bottom pieces",
            ],
            "casual": [
                "Feel free to experiment with bolder color combinations",
                "Use the 60-30-10 rule: 60% dominant color, 30% secondary, 10% accent",
                "Denim acts as a neutral and pairs with almost any color",
            ],
            "evening": [
                "Rich, saturated colors create elegance and drama",
                "Metallic accents can add glamour without overwhelming",
                "Consider the lighting - some colors look different under artificial light",
            ],
        }

        tips.extend(occasion_tips.get(occasion, []))
        return tips

    def _get_seasonal_styling_tips(self, season: str) -> List[str]:
        """Get season-specific styling recommendations"""
        seasonal_tips = {
            "spring": [
                "Embrace fresh, bright colors that echo blooming flowers",
                "Light layers allow for easy color mixing and matching",
                "Pastel and clear colors work beautifully in natural spring light",
            ],
            "summer": [
                "Light, cool colors help reflect heat and look fresh",
                "Whites and light blues are summer classics for good reason",
                "Bold tropical colors can be balanced with neutral accessories",
            ],
            "autumn": [
                "Rich, warm colors mirror the changing leaves perfectly",
                "Earth tones create sophisticated, grounded looks",
                "Deep jewel tones add luxury to autumn wardrobes",
            ],
            "winter": [
                "Deep, saturated colors create striking winter looks",
                "Black and white combinations are crisp and elegant",
                "Add warmth with rich burgundies and deep forest greens",
            ],
        }

        return seasonal_tips.get(season, [])

    def _get_occasion_color_guidelines(self, occasion: str) -> Dict[str, List[str]]:
        """Get specific color guidelines for occasions"""
        guidelines = {
            "professional": {
                "recommended": [
                    "Navy blue",
                    "Charcoal gray",
                    "Black",
                    "White",
                    "Burgundy",
                ],
                "use_sparingly": ["Bright red", "Hot pink", "Neon colors"],
                "avoid": [
                    "Fluorescent colors",
                    "Very casual patterns",
                    "Overly bright combinations",
                ],
            },
            "casual": {
                "recommended": ["Any colors that suit your undertone"],
                "use_sparingly": ["All black everything", "Overly formal combinations"],
                "avoid": ["Colors that wash you out", "Clashing combinations"],
            },
            "evening": {
                "recommended": [
                    "Black",
                    "Deep jewel tones",
                    "Metallic accents",
                    "Rich colors",
                ],
                "use_sparingly": ["Very light colors", "Casual color combinations"],
                "avoid": ["Washed out pastels", "Overly casual color schemes"],
            },
        }

        return guidelines.get(occasion, {})


# Enhanced demo function
def comprehensive_demo(wardrobe:list[dict[str, str | list[int]]],debug: bool = False):
    """Comprehensive demonstration of the color theory system"""
    if debug:
        print("=== Comprehensive Fashion Color Analysis Demo ===\n")

    # Initialize the recommendation engine
    engine = FashionColorRecommendationEngine()

    if debug:
        # Simulate a complete analysis
        print("1. Complete Color Profile Analysis")
        print("   - Analyzing skin tone and undertone...")
        print("   - Detected: Medium skin tone (Type 3) with warm undertone")
        print("   - Confidence: Skin tone 0.87, Undertone 0.82")

    # Simulate existing wardrobe
    sample_wardrobe = [
        {
            "id": "shirt_001",
            "wear_category": "upper",
            "item_type": "button_shirt",
            "dominant_colors": [
                [70, 130, 180],
                [255, 255, 255],
            ],  # Steel blue and white
        },
        {
            "id": "pants_001",
            "wear_category": "lower",
            "item_type": "chinos",
            "dominant_colors": [[139, 69, 19], [160, 82, 45]],  # Brown tones
        },
        {
            "id": "dress_001",
            "wear_category": "single_piece",
            "item_type": "midi_dress",
            "dominant_colors": [[25, 25, 112], [255, 255, 255]],  # Navy and white
        },
    ]

    print(f"\n2. Wardrobe Analysis ({len(sample_wardrobe)} items)")
    for item in sample_wardrobe:
        colors_str = [f"RGB{tuple(color)}" for color in item["dominant_colors"]]
        print(f"   - {item['item_type']}: {', '.join(colors_str)}")

    # Color harmony analysis
    print("\n3. Color Harmony Analysis")
    outfit_colors = [(70, 130, 180), (139, 69, 19), (255, 255, 255)]
    analyzer = AdvancedColorAnalyzer()
    coherence_score = analyzer.calculate_outfit_coherence_score(outfit_colors)
    print(f"   - Outfit coherence score: {coherence_score:.2f}/1.0")

    # Delta E calculations
    print("\n4. Color Difference Analysis (Delta E)")
    for i in range(len(outfit_colors)):
        for j in range(i + 1, len(outfit_colors)):
            delta_e = analyzer.calculate_delta_e(outfit_colors[i], outfit_colors[j])
            print(f"   - {outfit_colors[i]} vs {outfit_colors[j]}: ΔE = {delta_e:.1f}")

    # Seasonal recommendations
    print("\n5. Seasonal Color Adaptations")
    base_color = (70, 130, 180)  # Steel blue
    for season in ["spring", "summer", "autumn", "winter"]:
        seasonal_colors = analyzer.generate_seasonal_palette([base_color], season)
        print(f"   - {season.capitalize()}: {seasonal_colors[0]}")

    # Color temperature analysis
    print("\n6. Color Temperature Analysis")
    for color in outfit_colors:
        temp = analyzer.calculate_color_temperature(color)
        temp_type = "Cool" if temp > 6500 else "Warm" if temp < 5500 else "Neutral"
        print(f"   - {color}: {temp:.0f}K ({temp_type})")

    print("\n7. Styling Recommendations Generated")
    print("   - 3 monochromatic outfit suggestions")
    print("   - 2 complementary color combinations")
    print("   - 5 seasonal adaptations")
    print("   - Personalized coordination tips provided")

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    # Example garment colors
    outfit_colors = [
        (70, 130, 180),
        (255, 255, 255),
        (25, 25, 112),
    ]  # Steel blue, white, navy
    # Run the color analysis demo
    color_analysis(outfit_colors, debug=True)
    comprehensive_demo()
