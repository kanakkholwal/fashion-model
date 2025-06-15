import tempfile

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile

from pipeline.orchestrator import FashionSystem

app = FastAPI()
system = FashionSystem("processed_db.json", model_path="saved_models/best_color_model.pt")


@app.post("/analyze/")
async def analyze_fashion(
    image: UploadFile = File(...),
    wardrobe: UploadFile = File(...),
    occasion: str = Form("casual"),
    season: str = Form("spring")
):
    # Save image and wardrobe file to temp paths
    with tempfile.NamedTemporaryFile(delete=False) as img_temp:
        img_temp.write(await image.read())
        image_path = img_temp.name

    with tempfile.NamedTemporaryFile(delete=False) as ward_temp:
        ward_temp.write(await wardrobe.read())
        wardrobe_path = ward_temp.name

    result = system.analyze_image(
        image_path=image_path,
        wardrobe_path=wardrobe_path,
        occasion=occasion,
        season=season
    )

    return {
        "skin_tone": result["color_analysis"].color_profile.skin_tone.value,
        "undertone": result["color_analysis"].color_profile.undertone.value,
        "dominant_colors": result["color_analysis"].color_profile.dominant_colors,
        "recommendations": result["color_analysis"].color_recommendations,
        "similar_items": result["similar_items"]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
