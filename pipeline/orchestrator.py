# fashion_system/pipeline/orchestrator.py

from models.clip_search_engine import FashionSearchEngine
from color.analyzer import MLColorAnalysisPipeline


class FashionSystem:
    def __init__(self, db_path: str, model_path: str = None):
        self.search_engine = FashionSearchEngine(db_path)
        self.color_pipeline = MLColorAnalysisPipeline(model_path)

    def analyze_image(self, image_path: str, wardrobe_path: str, occasion: str, season: str):
        color_result = self.color_pipeline.analyze_image_complete(
            image=image_path,
            existing_wardrobe=wardrobe_path,
            occasion=occasion,
            season=season
        )
        similar_items = self.search_engine.search(image_path, k=5)
        return {
            "color_analysis": color_result,
            "similar_items": similar_items
        }
