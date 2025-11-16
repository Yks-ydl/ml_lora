"""
CLIP Baseline Model for Image-Text Retrieval
Implements the baseline using OpenAI's CLIP model
"""

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import open_clip
from typing import Dict, List, Tuple
import numpy as np


class CLIPBaseline:
    """
    Baseline CLIP model for cross-modal retrieval
    Supports both OpenAI CLIP and OpenCLIP variants
    """
    
    def __init__(
        self, 
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_openclip: bool = False
    ):
        """
        Args:
            model_name: HuggingFace model name or OpenCLIP model name
            device: Device to run model on
            use_openclip: Whether to use OpenCLIP implementation
        """
        self.device = device
        self.use_openclip = use_openclip
        
        if use_openclip:
            self._load_openclip(model_name)
        else:
            self._load_huggingface_clip(model_name)
    
    def _load_huggingface_clip(self, model_name: str):
        """Load CLIP from HuggingFace transformers"""
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
    def _load_openclip(self, model_name: str):
        """Load CLIP from OpenCLIP"""
        # Example: model_name = "ViT-B-32", pretrained = "laion2b_s34b_b79k"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained='laion2b_s34b_b79k'
        )
        self.model = model.to(self.device)
        self.processor = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
    
    @torch.no_grad()
    def encode_images(self, images: List) -> torch.Tensor:
        """
        Encode images into feature vectors
        
        Args:
            images: List of PIL Images or tensors
            
        Returns:
            Image features normalized to unit sphere (batch_size, feature_dim)
        """
        if self.use_openclip:
            if not isinstance(images, torch.Tensor):
                images = torch.stack([self.processor(img) for img in images])
            images = images.to(self.device)
            image_features = self.model.encode_image(images)
        else:
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts into feature vectors
        
        Args:
            texts: List of text strings
            
        Returns:
            Text features normalized to unit sphere (batch_size, feature_dim)
        """
        if self.use_openclip:
            text_tokens = self.tokenizer(texts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
        else:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            text_features = self.model.get_text_features(**inputs)
        
        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def compute_similarity(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between image and text features
        
        Args:
            image_features: (N, D) tensor
            text_features: (M, D) tensor
            
        Returns:
            Similarity matrix (N, M)
        """
        # Features are already normalized, so dot product = cosine similarity
        similarity = image_features @ text_features.T
        return similarity
    
    def retrieve_images(
        self, 
        query_text: str, 
        image_features: torch.Tensor,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k images given a text query
        
        Args:
            query_text: Text query string
            image_features: Pre-computed image features (N, D)
            top_k: Number of images to retrieve
            
        Returns:
            indices: Top-k image indices
            scores: Similarity scores
        """
        text_features = self.encode_texts([query_text])
        similarities = self.compute_similarity(image_features, text_features)
        similarities = similarities.squeeze().cpu().numpy()
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def retrieve_texts(
        self, 
        query_image, 
        text_features: torch.Tensor,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k texts given an image query
        
        Args:
            query_image: PIL Image or tensor
            text_features: Pre-computed text features (N, D)
            top_k: Number of texts to retrieve
            
        Returns:
            indices: Top-k text indices
            scores: Similarity scores
        """
        image_features = self.encode_images([query_image])
        similarities = self.compute_similarity(image_features, text_features)
        similarities = similarities.squeeze().cpu().numpy()
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores


class MultiScaleCLIP(CLIPBaseline):
    """
    Enhanced CLIP with multi-scale feature extraction
    Extracts features from multiple transformer layers for richer representations
    """
    
    def __init__(self, *args, layers_to_extract: List[int] = None, **kwargs):
        """
        Args:
            layers_to_extract: Which transformer layers to extract (e.g., [-1, -3, -6])
        """
        super().__init__(*args, **kwargs)
        self.layers_to_extract = layers_to_extract or [-1, -3, -6]
        
    @torch.no_grad()
    def encode_images_multiscale(self, images: List) -> torch.Tensor:
        """
        Extract multi-scale image features from different layers
        
        Returns:
            Concatenated features from multiple layers
        """
        if self.use_openclip:
            raise NotImplementedError("Multi-scale extraction not implemented for OpenCLIP")
        
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        # Extract from multiple layers
        outputs = self.model.vision_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get features from specified layers
        hidden_states = outputs.hidden_states
        selected_features = [hidden_states[i][:, 0] for i in self.layers_to_extract]  # CLS token
        
        # Concatenate and normalize
        multi_features = torch.cat(selected_features, dim=-1)
        multi_features = multi_features / multi_features.norm(dim=-1, keepdim=True)
        
        return multi_features


if __name__ == "__main__":
    # Example usage
    print("Testing CLIP Baseline...")
    
    # Initialize model
    model = CLIPBaseline()
    
    # Example texts
    texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a bird"
    ]
    
    # Encode texts
    text_features = model.encode_texts(texts)
    print(f"Text features shape: {text_features.shape}")
    
    # Test multi-scale
    print("\nTesting Multi-Scale CLIP...")
    multiscale_model = MultiScaleCLIP()
    print("Multi-scale model initialized successfully!")
