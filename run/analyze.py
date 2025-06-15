# run/analyze.py

from pipeline.orchestrator import FashionSystem

if __name__ == "__main__":
    system = FashionSystem(
        db_path="processed_db.json",
        model_path="saved_models/best_color_model.pt"
    )

    result = system.analyze_image(
        image_path="samples/person.jpg",
        wardrobe_path="samples/wardrobe_test.json",
        occasion="casual",
        season="spring"
    )

    print("\n--- Color Profile ---")
    print(result["color_analysis"].color_profile)

    print("\n--- Recommendations ---")
    print(result["color_analysis"].color_recommendations)

    print("\n--- Similar Items ---")
    for item in result["similar_items"]:
        print(f"{item['title']} -> {item['product_url']}")
