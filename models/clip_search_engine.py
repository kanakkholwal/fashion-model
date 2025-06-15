# fashion_system/models/clip_search_engine.py

import json
from io import BytesIO
from typing import Dict, List, Union

import clip
import faiss
import numpy as np
import requests
import torch
from PIL import Image


class FashionSearchEngine:
    def __init__(self, data_source: Union[str, List[Dict]], device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        self.index = faiss.IndexFlatIP(512)
        self.data = self._load_data(data_source)
        self.image_embeddings = []
        self.text_embeddings = []
        self.metadata = []
        self.valid_indices = []
        self._build_index()

    def _load_data(self, source: Union[str, List[Dict]]) -> List[Dict]:
        if isinstance(source, str):
            with open(source) as f:
                data = json.load(f)
        else:
            data = source

        return [
            {
                "product_url": item["product_url"],
                "image_url": item["image_urls"][0],
                "text_description": f"{item['title']} {item['description']}. "
                                    f"Gender: {item['gender']}, Type: {item['item_type']}. "
                                    f"Fabric: {item['specifications'].get('fabric', 'N/A')}, "
                                    f"Fit: {item['specifications'].get('fit', 'N/A')}, "
                                    f"Neck: {item['specifications'].get('neck', 'N/A')}, "
                                    f"Pattern: {item['specifications'].get('pattern', 'N/A')}",
                "metadata": item
            }
            for item in data
        ]

    def _encode_image(self, url: str) -> Union[np.ndarray, None]:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.model.encode_image(tensor)
            return emb.cpu().numpy().astype("float32")
        except Exception:
            return None

    def _encode_text(self, text: str) -> np.ndarray:
        token = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(token)
        return emb.cpu().numpy().astype("float32")

    def _build_index(self):
        for i, item in enumerate(self.data):
            img_emb = self._encode_image(item["image_url"])
            txt_emb = self._encode_text(item["text_description"])
            if img_emb is not None:
                combined = (img_emb + txt_emb) / 2
                self.image_embeddings.append(img_emb)
                self.text_embeddings.append(txt_emb)
                self.metadata.append(item["metadata"])
                self.index.add(combined)
                self.valid_indices.append(i)

    def search(self, query: Union[str, Image.Image, np.ndarray], k: int = 5) -> List[Dict]:
        if isinstance(query, str):
            query_emb = self._encode_text(query)
        elif isinstance(query, Image.Image):
            tensor = self.preprocess(query).unsqueeze(0).to(self.device)
            with torch.no_grad():
                query_emb = self.model.encode_image(tensor).cpu().numpy().astype("float32")
        elif isinstance(query, np.ndarray):
            query_emb = query
        else:
            return []

        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0: continue
            meta = self.data[self.valid_indices[idx]]["metadata"]
            results.append({
                **meta,
                "score": float(distances[0][i])
            })
        return results
