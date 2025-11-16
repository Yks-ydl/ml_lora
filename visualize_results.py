"""
Generate detailed comparison visualizations for all experiments
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


def load_experiment(summary_path):
    """Load experiment summary"""
    with open(summary_path, 'r') as f:
        return json.load(f)


def plot_comparison():
    """Generate comparison plots"""
    
    # Load all experiments
    experiments = {
        'Baseline CLIP': load_experiment('checkpoints/exp1_baseline/logs/exp1_clip_baseline_summary.json'),
        'Multi-Scale CLIP': load_experiment('checkpoints/exp2_multiscale/logs/exp2_multiscale_clip_summary.json'),
        'LoRA Fine-tuned': load_experiment('checkpoints/exp3_improved/logs/exp3_lora_improved_early_stop_summary.json'),
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Recall@K comparison
    ax1 = plt.subplot(2, 3, 1)
    metrics_to_plot = ['avg_recall@1', 'avg_recall@5', 'avg_recall@10']
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    for idx, (name, exp) in enumerate(experiments.items()):
        test_metrics = exp['final_metrics']['test']
        values = [test_metrics.get(m, 0) for m in metrics_to_plot]
        ax1.bar(x + idx * width, values, width, label=name, alpha=0.8)
    
    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Recall (%)')
    ax1.set_title('Average Recall@K Comparison', fontweight='bold', fontsize=12)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(['R@1', 'R@5', 'R@10'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Image-to-Text vs Text-to-Image
    ax2 = plt.subplot(2, 3, 2)
    metrics_i2t = ['i2t_recall@1', 'i2t_recall@5', 'i2t_recall@10']
    metrics_t2i = ['t2i_recall@1', 't2i_recall@5', 't2i_recall@10']
    
    for idx, (name, exp) in enumerate(experiments.items()):
        test_metrics = exp['final_metrics']['test']
        i2t_values = [test_metrics.get(m, 0) for m in metrics_i2t]
        t2i_values = [test_metrics.get(m, 0) for m in metrics_t2i]
        
        x_pos = np.arange(3) * 2 + idx * 0.5
        ax2.plot(x_pos, i2t_values, marker='o', label=f'{name} (I2T)', linewidth=2)
        ax2.plot(x_pos, t2i_values, marker='s', linestyle='--', label=f'{name} (T2I)', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Recall@K')
    ax2.set_ylabel('Recall (%)')
    ax2.set_title('Image-to-Text vs Text-to-Image', fontweight='bold', fontsize=12)
    ax2.set_xticks(np.arange(3) * 2 + 0.5)
    ax2.set_xticklabels(['K=1', 'K=5', 'K=10'])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Winoground Specific Scores
    ax3 = plt.subplot(2, 3, 3)
    wino_metrics = ['text_score', 'image_score', 'group_score']
    x = np.arange(len(wino_metrics))
    
    for idx, (name, exp) in enumerate(experiments.items()):
        test_metrics = exp['final_metrics']['test']
        values = [test_metrics.get(m, 0) for m in wino_metrics]
        ax3.bar(x + idx * width, values, width, label=name, alpha=0.8)
    
    ax3.set_xlabel('Metric')
    ax3.set_ylabel('Score (%)')
    ax3.set_title('Winoground Compositional Scores', fontweight='bold', fontsize=12)
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(['Text', 'Image', 'Group'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. MRR and mAP
    ax4 = plt.subplot(2, 3, 4)
    other_metrics = ['avg_mrr', 'avg_map']
    x = np.arange(len(other_metrics))
    
    for idx, (name, exp) in enumerate(experiments.items()):
        test_metrics = exp['final_metrics']['test']
        values = [test_metrics.get(m, 0) for m in other_metrics]
        ax4.bar(x + idx * width, values, width, label=name, alpha=0.8)
    
    ax4.set_xlabel('Metric')
    ax4.set_ylabel('Score')
    ax4.set_title('Mean Reciprocal Rank & Mean Average Precision', fontweight='bold', fontsize=12)
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(['MRR', 'mAP'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 0.7])
    
    # 5. Training time comparison
    ax5 = plt.subplot(2, 3, 5)
    names = list(experiments.keys())
    times = [exp['duration_minutes'] for exp in experiments.values()]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax5.barh(names, times, color=colors, alpha=0.8)
    ax5.set_xlabel('Time (minutes)')
    ax5.set_title('Training/Evaluation Time', fontweight='bold', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, time in zip(bars, times):
        ax5.text(time + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{time:.1f} min', va='center')
    
    # 6. Overall performance heatmap
    ax6 = plt.subplot(2, 3, 6)
    
    metrics_for_heatmap = [
        'avg_recall@1', 'avg_recall@5', 'avg_recall@10',
        'text_score', 'image_score', 'group_score',
        'avg_mrr', 'avg_map'
    ]
    
    heatmap_data = []
    for name, exp in experiments.items():
        test_metrics = exp['final_metrics']['test']
        row = [test_metrics.get(m, 0) for m in metrics_for_heatmap]
        heatmap_data.append(row)
    
    # Normalize each metric to 0-1 for better visualization
    heatmap_array = np.array(heatmap_data)
    normalized = (heatmap_array - heatmap_array.min(axis=0)) / (heatmap_array.max(axis=0) - heatmap_array.min(axis=0) + 1e-10)
    
    im = ax6.imshow(normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax6.set_xticks(np.arange(len(metrics_for_heatmap)))
    ax6.set_yticks(np.arange(len(experiments)))
    ax6.set_xticklabels([m.replace('avg_', '').replace('_', '\n') for m in metrics_for_heatmap], 
                        fontsize=8, rotation=45, ha='right')
    ax6.set_yticklabels(list(experiments.keys()))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax6)
    cbar.set_label('Normalized Score', rotation=270, labelpad=15)
    
    ax6.set_title('Performance Heatmap (Normalized)', fontweight='bold', fontsize=12)
    
    # Add text annotations
    for i in range(len(experiments)):
        for j in range(len(metrics_for_heatmap)):
            text = ax6.text(j, i, f'{heatmap_array[i, j]:.1f}',
                           ha="center", va="center", color="black", fontsize=7)
    
    # Overall title
    plt.suptitle('Winoground Experiments Comparison - All Models', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save
    output_path = 'experiment_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 对比图表已保存: {output_path}\n")
    
    plt.close()


def plot_lora_training_curves():
    """Plot LoRA training curves"""
    
    # Load LoRA metrics
    import pandas as pd
    
    try:
        df = pd.read_csv('checkpoints/exp3_lora/logs/exp3_lora_winoground_metrics.csv')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Training Loss
        ax = axes[0, 0]
        ax.plot(df['epoch'], df['train_loss'], marker='o', linewidth=2, color='#e74c3c')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss over Epochs', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 2. Validation Recall
        ax = axes[0, 1]
        recall_cols = [c for c in df.columns if 'val_avg_recall' in c]
        for col in recall_cols:
            label = col.replace('val_avg_recall@', 'Recall@')
            ax.plot(df['epoch'], df[col], marker='o', linewidth=2, label=label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall (%)')
        ax.set_title('Validation Recall@K', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Winoground Scores
        ax = axes[1, 0]
        wino_cols = ['val_text_score', 'val_image_score', 'val_group_score']
        colors = ['#3498db', '#2ecc71', '#f39c12']
        for col, color in zip(wino_cols, colors):
            if col in df.columns:
                label = col.replace('val_', '').replace('_', ' ').title()
                ax.plot(df['epoch'], df[col], marker='s', linewidth=2, label=label, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score (%)')
        ax.set_title('Winoground Compositional Scores', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. MRR and mAP
        ax = axes[1, 1]
        if 'val_avg_mrr' in df.columns:
            ax.plot(df['epoch'], df['val_avg_mrr'], marker='o', linewidth=2, 
                   label='MRR', color='#9b59b6')
        if 'val_avg_map' in df.columns:
            ax.plot(df['epoch'], df['val_avg_map'], marker='s', linewidth=2, 
                   label='mAP', color='#1abc9c')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('MRR & mAP over Epochs', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('LoRA Fine-tuning Training Progress', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = 'lora_training_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ LoRA 训练曲线已保存: {output_path}\n")
        plt.close()
        
    except FileNotFoundError:
        print("⚠️  未找到 LoRA 训练日志文件")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  生成实验对比可视化")
    print("="*70 + "\n")
    
    plot_comparison()
    plot_lora_training_curves()
    
    print("✅ 所有图表生成完成！\n")
    print("生成的文件:")
    print("  - experiment_comparison.png  (三个模型的全面对比)")
    print("  - lora_training_curves.png   (LoRA训练过程曲线)")
    print("\n")
