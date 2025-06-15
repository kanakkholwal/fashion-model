import json
import pandas as pd

# Load JSON data
with open('processed_db.json') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Create combined text description
df['text_description'] = df.apply(lambda x: 
    f"{x['title']} {x['description']} {x['gender']} {x['item_type']} "
    f"Fabric: {x['specifications'].get('fabric', '')} "
    f"Fit: {x['specifications'].get('fit', '')} "
    f"Pattern: {x['specifications'].get('pattern', '')} "
    f"Neck: {x['specifications'].get('neck', '')}",
    axis=1
)

# Keep only relevant columns
df = df[['product_url', 'image_urls', 'text_description']]