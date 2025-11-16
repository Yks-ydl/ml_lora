"""
LoRA Fine-tuning for Vision-Language Models
Efficient parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation)
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple
import wandb
from tqdm import tqdm


class LoRAFineTuner:
    """
    Fine-tune CLIP using LoRA for parameter-efficient training
    Particularly useful for Winoground compositional reasoning
    """
    
    def __init__(
        self,
        base_model_name: str = "openai/clip-vit-base-patch32",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: List[str] = None,
        device: str = "cuda"
    ):
        """
        Args:
            base_model_name: Base CLIP model to fine-tune
            lora_r: LoRA rank (lower = fewer parameters)
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout for LoRA layers
            target_modules: Which modules to apply LoRA (None = auto-detect)
            device: Device for training
        """
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(base_model_name)
        
        # Load base model
        self.base_model = CLIPModel.from_pretrained(base_model_name)
        
        # Configure LoRA
        if target_modules is None:
            # Default: apply to attention layers
            target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.to(device)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def contrastive_loss(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss
        
        Args:
            image_features: (batch_size, dim)
            text_features: (batch_size, dim)
            temperature: Temperature parameter for scaling
            
        Returns:
            Loss value
        """
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity matrix
        logits = (image_features @ text_features.T) / temperature
        
        # Labels: diagonal elements are positive pairs
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # Cross-entropy loss in both directions
        loss_i2t = nn.CrossEntropyLoss()(logits, labels)
        loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        return loss
    
    def hard_negative_loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        hard_negatives: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Contrastive loss with hard negative mining
        Useful for Winoground where similar descriptions exist
        
        Args:
            image_features: (batch_size, dim)
            text_features: (batch_size, dim) - positive texts
            hard_negatives: (batch_size, dim) - hard negative texts
            temperature: Temperature parameter
            
        Returns:
            Loss value
        """
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        hard_negatives = hard_negatives / hard_negatives.norm(dim=-1, keepdim=True)
        
        # Positive similarity
        pos_sim = torch.sum(image_features * text_features, dim=-1) / temperature
        
        # Hard negative similarity
        neg_sim = torch.sum(image_features * hard_negatives, dim=-1) / temperature
        
        # InfoNCE loss with hard negatives
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
        return loss.mean()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        use_hard_negatives: bool = False
    ) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader with (images, texts, [hard_negative_texts])
            optimizer: Optimizer instance
            use_hard_negatives: Whether to use hard negative mining
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            if use_hard_negatives:
                images, texts, hard_neg_texts = batch
            else:
                images, texts = batch
            
            # Process inputs
            image_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            text_inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            
            # Forward pass
            image_features = self.model.get_image_features(**image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
            
            # Compute loss
            if use_hard_negatives:
                hard_neg_inputs = self.processor(text=hard_neg_texts, return_tensors="pt", padding=True).to(self.device)
                hard_neg_features = self.model.get_text_features(**hard_neg_inputs)
                loss = self.hard_negative_loss(image_features, text_features, hard_neg_features)
            else:
                loss = self.contrastive_loss(image_features, text_features)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        metric: str = "recall@5"
    ) -> Dict[str, float]:
        """
        Evaluate on validation set
        
        Args:
            val_loader: Validation DataLoader
            metric: Evaluation metric
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        all_image_features = []
        all_text_features = []
        
        for images, texts in tqdm(val_loader, desc="Evaluating"):
            image_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            text_inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            
            image_features = self.model.get_image_features(**image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
            
            all_image_features.append(image_features)
            all_text_features.append(text_features)
        
        # Concatenate all features
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
        
        # Normalize
        all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)
        all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity matrix
        similarity = all_image_features @ all_text_features.T
        
        # Compute metrics
        metrics = self._compute_retrieval_metrics(similarity.cpu().numpy())
        
        return metrics
    
    def _compute_retrieval_metrics(self, similarity: torch.Tensor) -> Dict[str, float]:
        """Compute recall@k metrics"""
        import numpy as np
        
        n = similarity.shape[0]
        
        # Image-to-text retrieval
        i2t_ranks = []
        for i in range(n):
            rank = np.where(np.argsort(-similarity[i]) == i)[0][0] + 1
            i2t_ranks.append(rank)
        
        # Text-to-image retrieval
        t2i_ranks = []
        for i in range(n):
            rank = np.where(np.argsort(-similarity[:, i]) == i)[0][0] + 1
            t2i_ranks.append(rank)
        
        i2t_ranks = np.array(i2t_ranks)
        t2i_ranks = np.array(t2i_ranks)
        
        metrics = {
            "i2t_recall@1": (i2t_ranks <= 1).mean() * 100,
            "i2t_recall@5": (i2t_ranks <= 5).mean() * 100,
            "i2t_recall@10": (i2t_ranks <= 10).mean() * 100,
            "t2i_recall@1": (t2i_ranks <= 1).mean() * 100,
            "t2i_recall@5": (t2i_ranks <= 5).mean() * 100,
            "t2i_recall@10": (t2i_ranks <= 10).mean() * 100,
        }
        
        return metrics
    
    def save_model(self, save_path: str):
        """Save LoRA weights"""
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
    
    def load_model(self, load_path: str):
        """Load LoRA weights"""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.base_model, load_path)
        self.model.to(self.device)


if __name__ == "__main__":
    print("Testing LoRA Fine-tuner...")
    
    # Initialize
    finetuner = LoRAFineTuner()
    
    print("LoRA model initialized successfully!")
    print(f"Trainable parameters: {sum(p.numel() for p in finetuner.model.parameters() if p.requires_grad)}")
