"""
Compare multiple experiment results
"""

import json
from pathlib import Path
from typing import List, Dict
import sys


def load_summary(path: str) -> Dict:
    """Load summary file"""
    with open(path, 'r') as f:
        return json.load(f)


def format_metric(value: float, is_percent: bool = False) -> str:
    """Format metric value"""
    if is_percent:
        return f"{value:.2f}%"
    else:
        return f"{value:.4f}"


def compare_experiments(summary_paths: List[str]):
    """Compare multiple experiments"""
    
    # Load all summaries
    experiments = []
    for path in summary_paths:
        if Path(path).exists():
            exp = load_summary(path)
            exp['_path'] = path
            experiments.append(exp)
        else:
            print(f"⚠️  File not found: {path}")
    
    if not experiments:
        print("❌ No valid experiments to compare")
        return
    
    # Header
    print("\n" + "=" * 100)
    print("  EXPERIMENT COMPARISON")
    print("=" * 100)
    
    # Basic info table
    print("\n📋 实验概览:")
    print("-" * 100)
    print(f"{'实验名称':<30} {'模型类型':<20} {'数据集':<15} {'耗时(分钟)':<12}")
    print("-" * 100)
    for exp in experiments:
        name = exp['experiment_name'][:28]
        model = exp['config'].get('model_type', 'N/A')[:18]
        dataset = exp['config'].get('dataset', 'N/A')[:13]
        duration = exp.get('duration_minutes', 0)
        print(f"{name:<30} {model:<20} {dataset:<15} {duration:<12.1f}")
    print("-" * 100)
    
    # Compare key metrics on test set
    if all('final_metrics' in exp and 'test' in exp['final_metrics'] for exp in experiments):
        print("\n🎯 测试集关键指标对比:")
        print("-" * 100)
        
        # Get common metrics
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp['final_metrics']['test'].keys())
        
        # Priority metrics to show
        priority_metrics = [
            'avg_recall@1', 'avg_recall@5', 'avg_recall@10',
            'i2t_recall@1', 'i2t_recall@5', 't2i_recall@1', 't2i_recall@5',
            'group_score', 'image_score', 'text_score',
            'avg_mrr', 'avg_map'
        ]
        
        # Show priority metrics
        metrics_to_show = [m for m in priority_metrics if m in all_metrics]
        
        # Header
        header = f"{'指标':<25}"
        for exp in experiments:
            exp_name = exp['experiment_name'][:20]
            header += f" {exp_name:>20}"
        header += "  最佳"
        print(header)
        print("-" * 100)
        
        # Show each metric
        for metric in metrics_to_show:
            row = f"{metric:<25}"
            values = []
            
            for exp in experiments:
                value = exp['final_metrics']['test'].get(metric, 0)
                values.append(value)
                
                # Format based on metric type
                is_percent = 'recall' in metric or 'score' in metric
                formatted = format_metric(value, is_percent)
                row += f" {formatted:>20}"
            
            # Mark best
            best_idx = values.index(max(values))
            best_name = experiments[best_idx]['experiment_name'][:15]
            row += f"  ✓ {best_name}"
            
            print(row)
        
        print("-" * 100)
        
        # Overall winner
        print("\n🏆 综合评价:")
        
        # Count wins
        wins = {exp['experiment_name']: 0 for exp in experiments}
        
        for metric in metrics_to_show:
            values = []
            for exp in experiments:
                value = exp['final_metrics']['test'].get(metric, 0)
                values.append(value)
            
            best_idx = values.index(max(values))
            wins[experiments[best_idx]['experiment_name']] += 1
        
        # Sort by wins
        sorted_wins = sorted(wins.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (name, count) in enumerate(sorted_wins, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"  {medal} {name:<40} 胜出指标: {count}/{len(metrics_to_show)}")
    
    print("\n" + "=" * 100 + "\n")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        # Default comparison
        summary_paths = [
            "checkpoints/exp1_baseline/logs/exp1_clip_baseline_summary.json",
            "checkpoints/exp2_multiscale/logs/exp2_multiscale_clip_summary.json",
            "checkpoints/exp3_improved/logs/exp3_lora_improved_early_stop_summary.json",
        ]
        print("Usage: python compare_experiments.py <summary1.json> <summary2.json> ...")
        print(f"Using default experiments:\n")
        for path in summary_paths:
            print(f"  - {path}")
    else:
        summary_paths = sys.argv[1:]
    
    compare_experiments(summary_paths)


if __name__ == "__main__":
    main()
