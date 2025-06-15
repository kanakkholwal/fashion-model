# Initialize engine
from PIL import Image

from api import FashionSearchEngine

search_engine = FashionSearchEngine("processed_db.json")

# Text search
results = search_engine.search("oversized black cotton batman tshirt for men", k=3)
print("Text search results:", results)

# Image search
img = Image.open("https://assets.myntassets.com/h_720,q_90,w_540/v1/assets/images/2024/SEPTEMBER/16/XwCgFzBY_6d29bca91827462f9e11178316b85ae9.jpg")
results = search_engine.search(img, k=5)
print("Image search results:", results)

# Find similar to existing product
similar_items = search_engine.find_similar(
    "https://myntra.com/tshirts/the+souled+store/the-souled-store-batman-the-bat-sigil-black-oversized-t-shirts/18762622/buy",
    k=4,
)
print("Similar items:", similar_items)

# Add new products
new_items = [
    {
        "product_url": "https://myntra.com/tshirts/u.s.+polo+assn.+denim+co./us-polo-assn-denim-co-men-graphic-printed-polo-collar-cotton-slim-fit-t-shirt/30925184/buy",
        "image_urls": [
            "https://assets.myntassets.com/h_720,q_90,w_540/v1/assets/images/2024/SEPTEMBER/10/ynSxaKjf_5b5aefe070bc45349511e56cc51a0a21.jpg",
            "https://assets.myntassets.com/h_720,q_90,w_540/v1/assets/images/2024/SEPTEMBER/10/z6s3OL0r_9fdaef1c0e434c2d85fc54ce1db862d6.jpg",
            "https://assets.myntassets.com/h_720,q_90,w_540/v1/assets/images/2024/SEPTEMBER/10/7h3dbcb3_7aeea803df114a89bd3a71dd981d5751.jpg",
            "https://assets.myntassets.com/h_720,q_90,w_540/v1/assets/images/2024/SEPTEMBER/10/LGoB9f5P_4b7937c0ac5f4194b17543a0136a52a9.jpg",
        ],
        "title": "U.S. Polo Assn. Denim Co.",
        "description": "Men Graphic Printed Polo Collar Cotton Slim Fit T-shirt",
        "gender": "men",
        "item_type": "t-shirt",
        "wear_type": "upper_body",
        "specifications": {
            "fabric": "cotton",
            "fit": "slim fit",
            "length": "regular",
            "main trend": "graphic print others",
            "neck": "polo collar",
        },
    }
]
search_engine.update_index(new_items)
