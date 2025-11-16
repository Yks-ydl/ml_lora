"""
Main training script for multi-modal image retrieval
Supports multiple models, datasets, and training strategies
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent))

from models.clip_baseline import CLIPBaseline, MultiScaleCLIP
from models.lora_finetuner import LoRAFineTuner
from utils.data_loader import get_dataloaders
from utils.metrics import compute_all_metrics, compute_winoground_score, print_metrics
from utils.logger import ExperimentLogger


class Trainer:
    """
    Main trainer class for vision-language models
    """
    
    def __init__(self, config: dict):
        """
        Initialize trainer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize logger
        log_dir = Path(config.get('save_dir', 'logs')) / 'logs'
        experiment_name = config.get('experiment_name', 'experiment')
        self.logger = ExperimentLogger(str(log_dir), experiment_name)
        
        # Setup model
        self.setup_model()
        
        # Setup data
        self.setup_data()
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Best metrics tracking
        self.best_recall = 0.0
        self.best_epoch = 0
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping_patience', None)
        self.monitor_metric = config.get('monitor_metric', 'avg_recall@5')
        self.epochs_without_improvement = 0
    
    def setup_model(self):
        """Initialize model based on config"""
        model_type = self.config.get('model_type', 'clip')
        
        if model_type == 'clip':
            self.model = CLIPBaseline(
                model_name=self.config.get('model_name', 'openai/clip-vit-base-patch32'),
                device=self.device
            )
            self.trainable = False  # Baseline CLIP is not trained
            
        elif model_type == 'multiscale_clip':
            self.model = MultiScaleCLIP(
                model_name=self.config.get('model_name', 'openai/clip-vit-base-patch32'),
                layers_to_extract=self.config.get('layers', [-1, -3, -6]),
                device=self.device
            )
            self.trainable = False
            
        elif model_type == 'lora':
            self.model = LoRAFineTuner(
                base_model_name=self.config.get('model_name', 'openai/clip-vit-base-patch32'),
                lora_r=self.config.get('lora_r', 8),
                lora_alpha=self.config.get('lora_alpha', 16),
                lora_dropout=self.config.get('lora_dropout', 0.1),
                device=self.device
            )
            self.trainable = True
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Initialized {model_type} model on {self.device}")
    
    def setup_data(self):
        """Setup dataloaders"""
        dataset_name = self.config.get('dataset', 'winoground')
        batch_size = self.config.get('batch_size', 32)
        num_workers = self.config.get('num_workers', 4)
        use_hard_negatives = self.config.get('use_hard_negatives', True)
        local_data_path = self.config.get('local_data_path', None)
        auth_token = self.config.get('auth_token', None)
        
        self.dataloaders = get_dataloaders(
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_workers=num_workers,
            use_hard_negatives=use_hard_negatives,
            local_data_path=local_data_path,
            auth_token=auth_token
        )
        
        print(f"Loaded {dataset_name} dataset")
        print(f"  Train size: {len(self.dataloaders['train'].dataset)}")
        print(f"  Val size: {len(self.dataloaders['val'].dataset)}")
        print(f"  Test size: {len(self.dataloaders['test'].dataset)}")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        if not self.trainable:
            return
        
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        # Ensure lr and weight_decay are floats (in case YAML parses them as strings)
        lr = float(lr)
        weight_decay = float(weight_decay)
        
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Cosine annealing scheduler
        num_epochs = self.config.get('num_epochs', 10)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        if not self.trainable:
            print("Model is not trainable, skipping training")
            return 0.0
        
        use_hard_negatives = self.config.get('use_hard_negatives', True)
        loss = self.model.train_epoch(
            self.dataloaders['train'],
            self.optimizer,
            use_hard_negatives=use_hard_negatives
        )
        
        self.scheduler.step()
        
        return loss
    
    @torch.no_grad()
    def evaluate(self, split: str = 'val', compute_loss: bool = False):
        """Evaluate on validation or test set"""
        dataloader = self.dataloaders[split]
        
        # Extract all features
        all_image_features = []
        all_text_features = []
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Evaluating on {split}"):
            if len(batch) == 3:  # With hard negatives
                images, texts, _ = batch
            else:
                images, texts = batch
            
            # Encode
            if self.trainable:
                image_inputs = self.model.processor(images=images, return_tensors="pt").to(self.device)
                text_inputs = self.model.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
                
                image_features = self.model.model.get_image_features(**image_inputs)
                text_features = self.model.model.get_text_features(**text_inputs)
                
                # Compute loss if requested
                if compute_loss:
                    # Normalize features
                    image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Compute similarity matrix
                    logits = image_features_norm @ text_features_norm.T * self.model.model.logit_scale.exp()
                    
                    # Contrastive loss
                    labels = torch.arange(len(images)).to(self.device)
                    loss_i2t = nn.functional.cross_entropy(logits, labels)
                    loss_t2i = nn.functional.cross_entropy(logits.T, labels)
                    batch_loss = (loss_i2t + loss_t2i) / 2
                    
                    total_loss += batch_loss.item()
                    num_batches += 1
            else:
                image_features = self.model.encode_images(images)
                text_features = self.model.encode_texts(texts)
            
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
        
        # Concatenate
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
        
        # Normalize
        all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)
        all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = (all_image_features @ all_text_features.T).numpy()
        
        # Compute metrics
        metrics = compute_all_metrics(similarity)
        
        # For Winoground, also compute specific scores
        if self.config.get('dataset') == 'winoground':
            pairs = [(i, i) for i in range(len(all_image_features))]
            wino_metrics = compute_winoground_score(
                all_image_features,
                all_text_features,
                pairs
            )
            metrics.update(wino_metrics)
        
        # Add loss to metrics if computed
        if compute_loss and num_batches > 0:
            metrics['loss'] = total_loss / num_batches
        
        return metrics
    
    def train(self):
        """Main training loop"""
        num_epochs = self.config.get('num_epochs', 10)
        
        # Evaluate before training
        print("\nInitial evaluation...")
        val_metrics = self.evaluate('val')
        print_metrics(val_metrics, "Initial Validation Metrics")
        
        if not self.trainable:
            # Just evaluate on test set
            print("\nFinal evaluation on test set...")
            test_metrics = self.evaluate('test')
            print_metrics(test_metrics, "Test Metrics")
            
            # Save evaluation results for baseline models
            self.logger.log_metrics(0, {
                **{f'val_{k}': v for k, v in val_metrics.items()},
                **{f'test_{k}': v for k, v in test_metrics.items()}
            })
            self.logger.save_summary(
                config=self.config,
                final_metrics={'val': val_metrics, 'test': test_metrics}
            )
            self.logger.close()
            
            print(f"\n✓ Evaluation complete! Results saved to: {self.logger.log_dir}")
            return
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Evaluate (compute val loss for trainable models)
            val_metrics = self.evaluate('val', compute_loss=self.trainable)
            
            # Extract and display val loss separately if available
            val_loss = val_metrics.pop('loss', None)
            if val_loss is not None:
                print(f"Val Loss: {val_loss:.4f}")
            
            print_metrics(val_metrics, f"Validation Metrics (Epoch {epoch + 1})")
            
            # Log metrics
            log_entry = {
                'train_loss': train_loss,
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            if val_loss is not None:
                log_entry['val_loss'] = val_loss
            
            self.logger.log_metrics(epoch + 1, log_entry)
            
            # Plot curves after each epoch
            self.logger.plot_curves()
            
            # Save best model and check early stopping
            monitor_value = val_metrics.get(self.monitor_metric, 0)
            if monitor_value > self.best_recall:
                self.best_recall = monitor_value
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model')
                print(f"✓ New best model saved! {self.monitor_metric}: {monitor_value:.2f}%")
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement} epoch(s)")
            
            # Early stopping check
            if self.early_stopping_patience and self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered!")
                print(f"No improvement in {self.monitor_metric} for {self.early_stopping_patience} epochs")
                print(f"Best {self.monitor_metric}: {self.best_recall:.2f}% at epoch {self.best_epoch + 1}")
                break
            
            # Save periodic checkpoints
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}')
        
        # Final evaluation on test set
        print("\nFinal evaluation on test set...")
        self.load_checkpoint('best_model')
        test_metrics = self.evaluate('test')
        print_metrics(test_metrics, "Final Test Metrics")
        
        # Save final summary
        self.logger.save_summary(
            config=self.config,
            final_metrics={'test': test_metrics}
        )
        self.logger.close()
        
        print(f"\n✓ Experiment complete! Logs saved to: {self.logger.log_dir}")
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        save_dir = Path(self.config.get('save_dir', 'checkpoints'))
        save_dir.mkdir(exist_ok=True, parents=True)
        
        save_path = save_dir / name
        
        if self.trainable:
            self.model.save_model(str(save_path))
            print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, name: str):
        """Load model checkpoint"""
        load_dir = Path(self.config.get('save_dir', 'checkpoints'))
        load_path = load_dir / name
        
        if self.trainable and load_path.exists():
            self.model.load_model(str(load_path))
            print(f"Checkpoint loaded from {load_path}")


def main():
    parser = argparse.ArgumentParser(description='Train multimodal retrieval model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, choices=['clip', 'multiscale_clip', 'lora'],
                       help='Model type (overrides config)')
    parser.add_argument('--dataset', type=str, choices=['winoground', 'flickr30k', 'mscoco'],
                       help='Dataset (overrides config)')
    parser.add_argument('--no-train', action='store_true',
                       help='Only evaluate, do not train')
    
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            'model_type': 'clip',
            'model_name': 'openai/clip-vit-base-patch32',
            'dataset': 'winoground',
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'use_hard_negatives': True,
            'use_wandb': False,
        }
    
    # Override with command line args
    if args.model:
        config['model_type'] = args.model
    if args.dataset:
        config['dataset'] = args.dataset
    if args.no_train:
        config['num_epochs'] = 0
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Train or evaluate
    trainer.train()


if __name__ == "__main__":
    main()
