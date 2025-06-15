from dataclasses import dataclass
import color_prediction


@dataclass
class PredictionArgs:
    """Arguments for prediction pipeline"""

    fashion_analysis = color_prediction.FashionAnalysisArg


def generate_predictions(fashion_analysis: PredictionArgs.fashion_analysis):
    predictions = color_prediction.run_fashion_analysis_pipeline(
        image_path=fashion_analysis.image_path,
        prediction_path=fashion_analysis.prediction_path,
        wardrobe_path=fashion_analysis.wardrobe_path,
        occasion=fashion_analysis.occasion,
        season=fashion_analysis.season,
    )

    return {
        "predictions": predictions,
        "message": "Fashion color analysis completed successfully.",
    }


if __name__ == "__main__":
    # Example usage
    args = PredictionArgs(
        fashion_analysis=color_prediction.FashionAnalysisArg(
            image_path="https://example.com/image.jpg",
            prediction_path="predictions.json",
            wardrobe_path="wardrobe.json",
            occasion="casual",
            season="spring",
        )
    )

    result = generate_predictions(args.fashion_analysis)
    print(result)
