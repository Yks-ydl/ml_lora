"""
Evaluation metrics for image-text retrieval
Implements standard metrics: Recall@K, MRR, mAP
"""

import numpy as np
from typing import Dict, List, Tuple
import torch


def compute_recall_at_k(similarity_matrix: np.ndarray, k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute Recall@K for both image-to-text and text-to-image retrieval
    
    Args:
        similarity_matrix: (N, M) matrix where N=num_images, M=num_texts
                          Entry [i,j] is similarity between image i and text j
        k_values: List of K values to compute recall for
        
    Returns:
        Dictionary with recall metrics
    """
    n_images = similarity_matrix.shape[0]
    n_texts = similarity_matrix.shape[1]
    
    metrics = {}
    
    # Image-to-Text Retrieval
    i2t_ranks = []
    for i in range(n_images):
        # For each image, rank all texts by similarity
        sorted_indices = np.argsort(-similarity_matrix[i])  # Descending order
        # Find where the correct text (index i) is ranked
        rank = np.where(sorted_indices == i)[0][0] + 1  # +1 because rank starts at 1
        i2t_ranks.append(rank)
    
    i2t_ranks = np.array(i2t_ranks)
    
    for k in k_values:
        metrics[f'i2t_recall@{k}'] = (i2t_ranks <= k).mean() * 100
    
    # Text-to-Image Retrieval
    t2i_ranks = []
    for j in range(n_texts):
        # For each text, rank all images by similarity
        sorted_indices = np.argsort(-similarity_matrix[:, j])  # Descending order
        # Find where the correct image (index j) is ranked
        rank = np.where(sorted_indices == j)[0][0] + 1
        t2i_ranks.append(rank)
    
    t2i_ranks = np.array(t2i_ranks)
    
    for k in k_values:
        metrics[f't2i_recall@{k}'] = (t2i_ranks <= k).mean() * 100
    
    # Average metrics
    for k in k_values:
        metrics[f'avg_recall@{k}'] = (metrics[f'i2t_recall@{k}'] + metrics[f't2i_recall@{k}']) / 2
    
    # Median rank
    metrics['i2t_median_rank'] = np.median(i2t_ranks)
    metrics['t2i_median_rank'] = np.median(t2i_ranks)
    
    return metrics


def compute_mean_reciprocal_rank(similarity_matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute Mean Reciprocal Rank (MRR)
    MRR = average of (1 / rank_of_correct_item)
    
    Args:
        similarity_matrix: (N, M) similarity matrix
        
    Returns:
        Dictionary with MRR scores
    """
    n_images = similarity_matrix.shape[0]
    n_texts = similarity_matrix.shape[1]
    
    # Image-to-Text MRR
    i2t_reciprocal_ranks = []
    for i in range(n_images):
        sorted_indices = np.argsort(-similarity_matrix[i])
        rank = np.where(sorted_indices == i)[0][0] + 1
        i2t_reciprocal_ranks.append(1.0 / rank)
    
    # Text-to-Image MRR
    t2i_reciprocal_ranks = []
    for j in range(n_texts):
        sorted_indices = np.argsort(-similarity_matrix[:, j])
        rank = np.where(sorted_indices == j)[0][0] + 1
        t2i_reciprocal_ranks.append(1.0 / rank)
    
    metrics = {
        'i2t_mrr': np.mean(i2t_reciprocal_ranks),
        't2i_mrr': np.mean(t2i_reciprocal_ranks),
        'avg_mrr': (np.mean(i2t_reciprocal_ranks) + np.mean(t2i_reciprocal_ranks)) / 2
    }
    
    return metrics


def compute_mean_average_precision(similarity_matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute Mean Average Precision (mAP)
    Useful when there are multiple correct matches per query
    
    Args:
        similarity_matrix: (N, M) similarity matrix
        
    Returns:
        Dictionary with mAP scores
    """
    # For standard 1-to-1 matching, mAP = average precision at rank of correct item
    # This is equivalent to reciprocal rank for single correct match
    
    n_images = similarity_matrix.shape[0]
    
    # Image-to-Text mAP
    i2t_avg_precisions = []
    for i in range(n_images):
        sorted_indices = np.argsort(-similarity_matrix[i])
        rank = np.where(sorted_indices == i)[0][0] + 1
        # Average Precision for single relevant item = 1 / rank
        i2t_avg_precisions.append(1.0 / rank)
    
    # Text-to-Image mAP
    t2i_avg_precisions = []
    for j in range(n_images):
        sorted_indices = np.argsort(-similarity_matrix[:, j])
        rank = np.where(sorted_indices == j)[0][0] + 1
        t2i_avg_precisions.append(1.0 / rank)
    
    metrics = {
        'i2t_map': np.mean(i2t_avg_precisions),
        't2i_map': np.mean(t2i_avg_precisions),
        'avg_map': (np.mean(i2t_avg_precisions) + np.mean(t2i_avg_precisions)) / 2
    }
    
    return metrics


def compute_winoground_score(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    pairs: List[Tuple[int, int]]
) -> Dict[str, float]:
    """
    Compute Winoground-specific evaluation metrics
    
    Winoground requires:
    - Text score: Both captions correctly identify their images
    - Image score: Both images correctly identify their captions
    - Group score: All 4 matches are correct
    
    Args:
        image_features: (N, D) tensor of image features
        text_features: (N, D) tensor of text features
        pairs: List of (image_idx, text_idx) correct pairs
        
    Returns:
        Dictionary with Winoground scores
    """
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Compute similarity matrix
    similarity = (image_features @ text_features.T).cpu().numpy()
    
    # Winoground examples come in pairs
    n_examples = len(pairs) // 2
    
    text_correct = 0
    image_correct = 0
    group_correct = 0
    
    for i in range(0, len(pairs), 2):
        # Get indices for the pair
        img0_idx, txt0_idx = pairs[i]
        img1_idx, txt1_idx = pairs[i + 1]
        
        # Text score: Each caption should rank its image highest
        txt0_ranks_img0_higher = similarity[img0_idx, txt0_idx] > similarity[img1_idx, txt0_idx]
        txt1_ranks_img1_higher = similarity[img1_idx, txt1_idx] > similarity[img0_idx, txt1_idx]
        
        if txt0_ranks_img0_higher and txt1_ranks_img1_higher:
            text_correct += 1
        
        # Image score: Each image should rank its caption highest
        img0_ranks_txt0_higher = similarity[img0_idx, txt0_idx] > similarity[img0_idx, txt1_idx]
        img1_ranks_txt1_higher = similarity[img1_idx, txt1_idx] > similarity[img1_idx, txt0_idx]
        
        if img0_ranks_txt0_higher and img1_ranks_txt1_higher:
            image_correct += 1
        
        # Group score: All 4 conditions must be satisfied
        if (txt0_ranks_img0_higher and txt1_ranks_img1_higher and 
            img0_ranks_txt0_higher and img1_ranks_txt1_higher):
            group_correct += 1
    
    metrics = {
        'text_score': text_correct / n_examples * 100,
        'image_score': image_correct / n_examples * 100,
        'group_score': group_correct / n_examples * 100
    }
    
    return metrics


def compute_all_metrics(similarity_matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics
    
    Args:
        similarity_matrix: (N, M) similarity matrix
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Recall@K
    recall_metrics = compute_recall_at_k(similarity_matrix)
    metrics.update(recall_metrics)
    
    # MRR
    mrr_metrics = compute_mean_reciprocal_rank(similarity_matrix)
    metrics.update(mrr_metrics)
    
    # mAP
    map_metrics = compute_mean_average_precision(similarity_matrix)
    metrics.update(map_metrics)
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """Pretty print evaluation metrics"""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}\n")
    
    # Group metrics
    recall_metrics = {k: v for k, v in metrics.items() if 'recall' in k}
    rank_metrics = {k: v for k, v in metrics.items() if 'rank' in k}
    other_metrics = {k: v for k, v in metrics.items() if k not in recall_metrics and k not in rank_metrics}
    
    # Print Recall@K
    if recall_metrics:
        print("Recall@K Metrics:")
        for k, v in sorted(recall_metrics.items()):
            print(f"  {k:20s}: {v:6.2f}%")
    
    # Print Rank Metrics
    if rank_metrics:
        print("\nRank Metrics:")
        for k, v in sorted(rank_metrics.items()):
            print(f"  {k:20s}: {v:6.2f}")
    
    # Print Other Metrics
    if other_metrics:
        print("\nOther Metrics:")
        for k, v in sorted(other_metrics.items()):
            if isinstance(v, float):
                print(f"  {k:20s}: {v:6.4f}")
            else:
                print(f"  {k:20s}: {v}")
    
    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    print("Testing evaluation metrics...")
    
    # Create a sample similarity matrix
    n = 100
    similarity = np.random.rand(n, n)
    
    # Make diagonal dominant (correct matches have high similarity)
    similarity += np.eye(n) * 2
    
    # Compute metrics
    metrics = compute_all_metrics(similarity)
    print_metrics(metrics)
    
    print("\nTesting Winoground metrics...")
    # Create sample features for 4 images and 4 texts (2 Winoground examples)
    image_features = torch.randn(4, 512)
    text_features = torch.randn(4, 512)
    
    # Define correct pairs
    pairs = [(0, 0), (1, 1), (2, 2), (3, 3)]
    
    wino_metrics = compute_winoground_score(image_features, text_features, pairs)
    print_metrics(wino_metrics, "Winoground Scores")
