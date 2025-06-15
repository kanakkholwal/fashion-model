from dataclasses import dataclass
import json
import urllib.request

import cv2
import numpy as np

from color_ml_integration_pipeline import MLColorAnalysisPipeline

@dataclass
class FashionAnalysisArg:
    "Fashion Color Analysis arguments"
    image_path = str
    prediction_path = str
    wardrobe_path = str
    occasion = str
    occasion = str
    
    


def load_image(image_path: str):
    if image_path.startswith('http://') or image_path.startswith('https://'):
        resp = urllib.request.urlopen(image_path.replace("https://","http://"))
        image_data = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found or could not be loaded: {image_path}")
    return image

def run_fashion_analysis_pipeline(image_path: str,prediction_path:str, wardrobe_path: str, occasion: str = 'casual', season: str = 'spring'):
    # Load image (local or remote)
    image = load_image(image_path)

    # Load wardrobe JSON
    with open(wardrobe_path, 'r') as f:
        wardrobe = json.load(f)

    # Initialize pipeline
    pipeline = MLColorAnalysisPipeline()

    # Run full analysis
    prediction = pipeline.analyze_image_complete(
        image=image,
        existing_wardrobe=wardrobe,
        occasion=occasion,
        season=season
    )

    # Display results
    print("=== Fashion Color Analysis Result ===")
    print("\n--- Color Profile ---")
    print(f"Skin Tone: {prediction.color_profile.skin_tone.value}")
    print(f"Undertone: {prediction.color_profile.undertone.value}")
    print("Dominant Colors:", prediction.color_profile.dominant_colors)
    print("Confidence Scores:", prediction.confidence_scores)

    print("\n--- Recommendations ---")
    print(json.dumps(prediction.color_recommendations, indent=2))
     # Save entire prediction object to JSON
    with open(prediction_path, 'w') as out_file:
        json.dump(prediction.__dict__, out_file, indent=2, default=lambda o:str(o))

    return prediction


if __name__ == "__main__":
    # Example usage
    # image_file = "samples/person_01.jpg"  # or URL like 'https://example.com/person_01.jpg'
    image_file = "http://assets.myntassets.com/h_720,q_90,w_540/v1/assets/images/18762622/2023/10/13/e67de1ce-cc38-449c-accf-fdc69186c8831697199664984ThesouledstoreBatmanTheBatSigilBlackOversizedT-Shirts5.jpg"
    wardrobe_file = "samples/wardrobe_test.json"

    run_fashion_analysis_pipeline(
        image_path=image_file,
        prediction_path='samples/prediction_fashion_analysis.json',
        wardrobe_path=wardrobe_file,
        occasion='professional',
        season='spring'
    )
