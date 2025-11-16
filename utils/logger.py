"""
Simple logger for tracking experiments without wandb
Saves metrics to JSON and CSV files, and plots learning curves
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from datetime import datetime


class ExperimentLogger:
    """
    Logger for tracking training metrics and plotting curves
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initialize logger
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
        # Storage for metrics
        self.metrics_history = []
        
        # File paths
        self.json_path = self.log_dir / f"{experiment_name}_metrics.json"
        self.csv_path = self.log_dir / f"{experiment_name}_metrics.csv"
        self.plot_path = self.log_dir / f"{experiment_name}_curves.png"
        
        # Initialize CSV
        self.csv_file = None
        self.csv_writer = None
        
        print(f"📊 Logging to: {self.log_dir.absolute()}")
    
    def log_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """
        Log metrics for an epoch
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics to log
        """
        # Add timestamp and epoch
        log_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.metrics_history.append(log_entry)
        
        # Save to JSON (overwrite with full history)
        with open(self.json_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Append to CSV
        if self.csv_file is None:
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=log_entry.keys())
            self.csv_writer.writeheader()
        
        self.csv_writer.writerow(log_entry)
        self.csv_file.flush()
    
    def plot_curves(self, save_path: str = None):
        """
        Plot learning curves from logged metrics
        
        Args:
            save_path: Path to save the plot (optional, defaults to log_dir)
        """
        if not self.metrics_history:
            print("No metrics to plot yet")
            return
        
        if save_path is None:
            save_path = self.plot_path
        
        # Extract data
        epochs = [m['epoch'] for m in self.metrics_history]
        
        # Determine what metrics we have
        metric_keys = set()
        for m in self.metrics_history:
            metric_keys.update(m.keys())
        
        metric_keys.discard('epoch')
        metric_keys.discard('timestamp')
        
        # Separate train and val metrics
        train_loss = [k for k in metric_keys if k == 'train_loss']
        val_loss = [k for k in metric_keys if k == 'val_loss']
        
        # Key validation metrics (最多4个)
        key_val_metrics = []
        priority_metrics = ['val_avg_recall@5', 'val_avg_recall@1', 'val_group_score', 'val_avg_mrr']
        for metric in priority_metrics:
            if metric in metric_keys:
                key_val_metrics.append(metric)
                if len(key_val_metrics) >= 4:
                    break
        
        # If not enough priority metrics, add others
        if len(key_val_metrics) < 4:
            for k in sorted(metric_keys):
                if k.startswith('val_') and k not in key_val_metrics and k != 'val_loss':
                    key_val_metrics.append(k)
                    if len(key_val_metrics) >= 4:
                        break
        
        # Create 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Loss (train and val)
        ax = axes[0]
        has_loss_data = False
        
        if train_loss:
            values = [m.get('train_loss', None) for m in self.metrics_history]
            if any(v is not None for v in values):
                ax.plot(epochs, values, marker='o', label='Train Loss', linewidth=2, color='#e74c3c')
                has_loss_data = True
        
        if val_loss:
            values = [m.get('val_loss', None) for m in self.metrics_history]
            if any(v is not None for v in values):
                ax.plot(epochs, values, marker='s', label='Val Loss', linewidth=2, color='#3498db')
                has_loss_data = True
        
        if has_loss_data:
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Loss', fontsize=14, fontweight='bold')
        
        # Plot 2: Key validation metrics (max 4)
        ax = axes[1]
        colors = ['#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
        
        for idx, metric in enumerate(key_val_metrics[:4]):
            values = [m.get(metric, None) for m in self.metrics_history]
            if any(v is not None for v in values):
                label = metric.replace('val_', '').replace('avg_', '').replace('_', ' ').title()
                ax.plot(epochs, values, marker='o', label=label, linewidth=2, 
                       color=colors[idx % len(colors)])
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Key Validation Metrics', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Experiment: {self.experiment_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Learning curves saved to: {save_path}")
    
    def save_summary(self, config: Dict = None, final_metrics: Dict = None):
        """
        Save a summary of the experiment
        
        Args:
            config: Experiment configuration
            final_metrics: Final test/validation metrics
        """
        summary_path = self.log_dir / f"{self.experiment_name}_summary.json"
        
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'config': config,
            'final_metrics': final_metrics,
            'total_epochs': len(self.metrics_history)
        }
        
        # Find best epoch
        if self.metrics_history:
            # Try to find best validation recall
            val_recalls = []
            for m in self.metrics_history:
                for key in m:
                    if 'recall' in key.lower() and 'val' in key.lower():
                        val_recalls.append((m['epoch'], key, m[key]))
            
            if val_recalls:
                best_epoch, best_metric, best_value = max(val_recalls, key=lambda x: x[2])
                summary['best_epoch'] = best_epoch
                summary['best_metric'] = best_metric
                summary['best_value'] = best_value
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Summary saved to: {summary_path}")
    
    def close(self):
        """Close open file handles"""
        if self.csv_file:
            self.csv_file.close()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()


class SimpleProgressBar:
    """Simple progress indicator for epochs"""
    
    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.desc = desc
        self.current = 0
    
    def update(self, metrics: Dict = None):
        """Update progress"""
        self.current += 1
        progress = self.current / self.total
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        metrics_str = ""
        if metrics:
            metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                      for k, v in metrics.items()])
        
        print(f"\r{self.desc} [{bar}] {self.current}/{self.total} | {metrics_str}", end='', flush=True)
        
        if self.current == self.total:
            print()  # New line when complete
