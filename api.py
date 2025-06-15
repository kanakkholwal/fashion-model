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
        """
        Initialize the fashion search engine
        954
        Args:
            data_source: Path to JSON file or list of product dictionaries
            device: Hardware device ('cuda' or 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = None, None
        self.index = None
        self.metadata = []
        self.image_embeddings = []
        self.text_embeddings = []
        
        # Load data
        self.data = self._load_data(data_source)
        
        # Initialize model and index
        self._initialize_model()
        self._build_index()

    def _load_data(self, data_source: Union[str, List[Dict]]) -> List[Dict]:
        """Load and preprocess fashion product data"""
        if isinstance(data_source, str):
            with open(data_source) as f:
                data = json.load(f)
        else:
            data = data_source
            
        processed = []
        for product in data:
            # Create enhanced text description
            specs = product.get('specifications', {})
            description = (
                f"{product['title']} {product['description']}. "
                f"Gender: {product['gender']}, Type: {product['item_type']}. "
                f"Fabric: {specs.get('fabric', 'N/A')}, Fit: {specs.get('fit', 'N/A')}, "
                f"Neck: {specs.get('neck', 'N/A')}, Pattern: {specs.get('pattern', 'N/A')}"
            )
            
            processed.append({
                'product_url': product['product_url'],
                'image_url': product['image_urls'][0],  # Use first image
                'text_description': description,
                'metadata': product  # Store all original data
            })
        return processed

    def _initialize_model(self):
        """Load CLIP model and preprocessing functions"""
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()  # Set to evaluation mode

    def _get_image_embedding(self, image: Union[str, Image.Image]) -> np.ndarray:
        """Generate embedding from image URL or PIL image"""
        if isinstance(image, str):
            try:
                response = requests.get(image)
                img = Image.open(BytesIO(response.content))
            except Exception:
                return None
        else:
            img = image
            
        try:
            image_input = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_image(image_input)
            return embedding.cpu().numpy().astype('float32')
        except Exception:
            return None

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding from text query"""
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_text(text_input)
        return embedding.cpu().numpy().astype('float32')

    def _build_index(self):
        """Generate embeddings and build FAISS index"""
        # Initialize index
        self.index = faiss.IndexFlatIP(512)  # ViT-B/32 embedding size
        
        # Generate embeddings
        valid_indices = []
        for idx, product in enumerate(self.data):
            img_emb = self._get_image_embedding(product['image_url'])
            txt_emb = self._get_text_embedding(product['text_description'])
            
            if img_emb is not None and txt_emb is not None:
                # Combined multimodal embedding
                combined_emb = (img_emb + txt_emb) / 2
                self.image_embeddings.append(img_emb)
                self.text_embeddings.append(txt_emb)
                self.metadata.append(product['metadata'])
                valid_indices.append(idx)
        
        # Create index
        if valid_indices:
            embeddings = np.vstack([
                (img + txt) / 2 
                for img, txt in zip(self.image_embeddings, self.text_embeddings)
            ])
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        
        # Map index positions to original data
        self.valid_indices = valid_indices

    def search(self, query: Union[str, Image.Image, np.ndarray], k: int = 5) -> List[Dict]:
        """
        Perform multimodal search
        
        Args:
            query: Text string, PIL Image, image URL, or existing embedding
            k: Number of results to return
            
        Returns:
            List of product metadata with similarity scores
        """
        # Handle different query types
        if isinstance(query, str):
            query_emb = self._get_text_embedding(query)
        elif isinstance(query, Image.Image):
            query_emb = self._get_image_embedding(query)
        elif isinstance(query, np.ndarray):
            query_emb = query
        else:
            raise ValueError("Unsupported query type")
        
        if query_emb is None:
            return []
        
        # Prepare for search
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb, k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:  # FAISS returns -1 for invalid indices
                continue
                
            original_idx = self.valid_indices[idx]
            results.append({
                **self.data[original_idx]['metadata'],
                'score': float(distances[0][i]),
                'product_url': self.data[original_idx]['product_url'],
                'image_url': self.data[original_idx]['image_url']
            })
        
        return results

    def find_similar(self, product_id: str, k: int = 5) -> List[Dict]:
        """Find similar items to existing product"""
        # Find product index
        for idx, product in enumerate(self.data):
            if product['metadata'].get('product_url') == product_id:
                # Use existing multimodal embedding
                emb_idx = self.valid_indices.index(idx)
                query_emb = (self.image_embeddings[emb_idx] + self.text_embeddings[emb_idx]) / 2
                return self.search(query_emb, k)
        return []

    def update_index(self, new_products: List[Dict]):
        """Add new products to the index"""
        new_data = self._load_data(new_products)
        start_idx = len(self.data)
        self.data.extend(new_data)
        
        # Add new embeddings
        new_embeddings = []
        for product in new_data:
            img_emb = self._get_image_embedding(product['image_url'])
            txt_emb = self._get_text_embedding(product['text_description'])
            
            if img_emb is not None and txt_emb is not None:
                combined_emb = (img_emb + txt_emb) / 2
                self.image_embeddings.append(img_emb)
                self.text_embeddings.append(txt_emb)
                self.metadata.append(product['metadata'])
                new_embeddings.append(combined_emb)
                self.valid_indices.append(start_idx)
                start_idx += 1
        
        # Update FAISS index
        if new_embeddings:
            new_embeddings = np.vstack(new_embeddings)
            faiss.normalize_L2(new_embeddings)
            self.index.add(new_embeddings)