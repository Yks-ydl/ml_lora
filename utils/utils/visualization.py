"""
Visualization utilities for multimodal retrieval
Includes t-SNE, UMAP, attention maps, and retrieval results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import torch
from typing import List, Tuple, Optional
from PIL import Image
import io


def plot_embedding_space(
    image_features: np.ndarray,
    text_features: np.ndarray,
    method: str = 'tsne',
    n_samples: int = 500,
    save_path: Optional[str] = None
):
    """
    Visualize image and text embeddings in 2D space
    
    Args:
        image_features: (N, D) array of image features
        text_features: (N, D) array of text features
        method: 'tsne', 'umap', or 'pca'
        n_samples: Number of samples to plot (for efficiency)
        save_path: Path to save figure
    """
    # Subsample if needed
    if len(image_features) > n_samples:
        indices = np.random.choice(len(image_features), n_samples, replace=False)
        image_features = image_features[indices]
        text_features = text_features[indices]
    
    # Combine features
    all_features = np.vstack([image_features, text_features])
    labels = ['Image'] * len(image_features) + ['Text'] * len(text_features)
    
    # Apply dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = reducer.fit_transform(all_features)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(all_features)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(all_features)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Separate image and text points
    img_points = embeddings_2d[:len(image_features)]
    txt_points = embeddings_2d[len(image_features):]
    
    ax.scatter(img_points[:, 0], img_points[:, 1], 
              c='blue', alpha=0.6, s=50, label='Images', marker='o')
    ax.scatter(txt_points[:, 0], txt_points[:, 1], 
              c='red', alpha=0.6, s=50, label='Texts', marker='^')
    
    # Draw lines connecting matched pairs
    for i in range(min(50, len(image_features))):  # Only show 50 connections for clarity
        ax.plot([img_points[i, 0], txt_points[i, 0]], 
               [img_points[i, 1], txt_points[i, 1]], 
               'gray', alpha=0.2, linewidth=0.5)
    
    ax.set_title(f'Vision-Language Embedding Space ({method.upper()})', fontsize=16)
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_similarity_matrix(
    similarity: np.ndarray,
    top_k: int = 50,
    save_path: Optional[str] = None
):
    """
    Visualize similarity matrix as heatmap
    
    Args:
        similarity: (N, M) similarity matrix
        top_k: Only show top_k x top_k samples
        save_path: Path to save figure
    """
    # Subsample if large
    if similarity.shape[0] > top_k:
        similarity = similarity[:top_k, :top_k]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(similarity, cmap='RdYlBu_r', center=0, 
                square=True, linewidths=0, cbar_kws={"shrink": 0.8},
                xticklabels=False, yticklabels=False, ax=ax)
    
    ax.set_title('Image-Text Similarity Matrix', fontsize=16)
    ax.set_xlabel('Text Index', fontsize=12)
    ax.set_ylabel('Image Index', fontsize=12)
    
    # Highlight diagonal (correct matches)
    for i in range(min(similarity.shape)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, 
                                   edgecolor='green', lw=2))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved similarity matrix to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_retrieval_results(
    query: str,
    retrieved_images: List[Image.Image],
    scores: List[float],
    is_text_query: bool = True,
    save_path: Optional[str] = None
):
    """
    Visualize retrieval results
    
    Args:
        query: Query text or None if image query
        retrieved_images: List of retrieved images
        scores: Similarity scores
        is_text_query: Whether query is text (vs image)
        save_path: Path to save figure
    """
    n_results = len(retrieved_images)
    n_cols = min(5, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (img, score) in enumerate(zip(retrieved_images, scores)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        ax.imshow(img)
        ax.set_title(f'Rank {idx+1}\nScore: {score:.3f}', fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_results, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    # Add query as suptitle
    if is_text_query:
        fig.suptitle(f'Text Query: "{query}"', fontsize=14, y=0.98)
    else:
        fig.suptitle('Image Query (shown separately)', fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved retrieval results to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_metrics_comparison(
    results: dict,
    save_path: Optional[str] = None
):
    """
    Compare metrics across different models/experiments
    
    Args:
        results: Dictionary of {model_name: metrics_dict}
        save_path: Path to save figure
    """
    # Extract recall metrics
    recall_keys = ['i2t_recall@1', 'i2t_recall@5', 'i2t_recall@10',
                   't2i_recall@1', 't2i_recall@5', 't2i_recall@10']
    
    models = list(results.keys())
    
    # Prepare data
    i2t_data = []
    t2i_data = []
    
    for model in models:
        metrics = results[model]
        i2t_data.append([metrics.get(k, 0) for k in recall_keys[:3]])
        t2i_data.append([metrics.get(k, 0) for k in recall_keys[3:]])
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Image-to-Text
    x = np.arange(3)
    width = 0.8 / len(models)
    
    for i, (model, values) in enumerate(zip(models, i2t_data)):
        ax1.bar(x + i*width, values, width, label=model)
    
    ax1.set_xlabel('Metric', fontsize=12)
    ax1.set_ylabel('Recall (%)', fontsize=12)
    ax1.set_title('Image-to-Text Retrieval', fontsize=14)
    ax1.set_xticks(x + width*(len(models)-1)/2)
    ax1.set_xticklabels(['R@1', 'R@5', 'R@10'])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Text-to-Image
    for i, (model, values) in enumerate(zip(models, t2i_data)):
        ax2.bar(x + i*width, values, width, label=model)
    
    ax2.set_xlabel('Metric', fontsize=12)
    ax2.set_ylabel('Recall (%)', fontsize=12)
    ax2.set_title('Text-to-Image Retrieval', fontsize=14)
    ax2.set_xticks(x + width*(len(models)-1)/2)
    ax2.set_xticklabels(['R@1', 'R@5', 'R@10'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")
        plt.close()
    else:
        plt.show()
    plt.show()


def plot_feature_distribution(
    features: np.ndarray,
    title: str = "Feature Distribution",
    save_path: Optional[str] = None
):
    """
    Plot distribution of feature values
    
    Args:
        features: (N, D) feature matrix
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Feature norms
    norms = np.linalg.norm(features, axis=1)
    # Use robust binning strategy
    n_unique = len(np.unique(norms))
    if n_unique <= 1:
        # All values are the same - just show a single bar
        axes[0, 0].hist(norms, bins=1, edgecolor='black', alpha=0.7)
    else:
        # Use number of bins that won't exceed unique values
        n_bins_norms = min(50, max(3, min(n_unique, int(len(norms) ** 0.5))))
        try:
            axes[0, 0].hist(norms, bins=n_bins_norms, edgecolor='black', alpha=0.7)
        except ValueError:
            # Fallback to very conservative binning
            axes[0, 0].hist(norms, bins=min(3, n_unique), edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Feature Norms Distribution')
    axes[0, 0].set_xlabel('L2 Norm')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Feature values distribution
    flat_features = features.flatten()
    n_unique = len(np.unique(flat_features))
    if n_unique <= 1:
        axes[0, 1].hist(flat_features, bins=1, edgecolor='black', alpha=0.7)
    else:
        n_bins_values = min(100, max(10, min(n_unique, int(len(flat_features) ** 0.5) // 10)))
        try:
            axes[0, 1].hist(flat_features, bins=n_bins_values, edgecolor='black', alpha=0.7)
        except ValueError:
            axes[0, 1].hist(flat_features, bins=min(10, n_unique), edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Feature Values Distribution')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Variance per dimension
    variances = np.var(features, axis=0)
    axes[1, 0].plot(variances, linewidth=1.5)
    axes[1, 0].set_title('Variance per Feature Dimension')
    axes[1, 0].set_xlabel('Dimension')
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation matrix (sample of dimensions)
    n_dims = min(50, features.shape[1])
    corr = np.corrcoef(features[:, :n_dims].T)
    im = axes[1, 1].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[1, 1].set_title(f'Feature Correlation (first {n_dims} dims)')
    plt.colorbar(im, ax=axes[1, 1])
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature distribution to {save_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    print("Testing visualization utilities...")
    print("Generating sample data and saving visualizations...\n")
    
    # Create output directory
    from pathlib import Path
    output_dir = Path("test_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Generate sample data
    n_samples = 100
    dim = 512
    
    image_features = np.random.randn(n_samples, dim)
    text_features = np.random.randn(n_samples, dim)
    
    # Make them somewhat aligned (for testing)
    text_features = 0.7 * image_features + 0.3 * text_features
    
    # Normalize
    image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
    text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
    
    # 1. Embedding space
    print("1. Plotting embedding space (t-SNE)...")
    plot_embedding_space(
        image_features, text_features, 
        method='tsne', 
        n_samples=100,
        save_path=str(output_dir / "embedding_tsne.png")
    )
    
    print("2. Plotting embedding space (PCA)...")
    plot_embedding_space(
        image_features, text_features, 
        method='pca', 
        n_samples=100,
        save_path=str(output_dir / "embedding_pca.png")
    )
    
    # 2. Similarity matrix
    print("3. Plotting similarity matrix...")
    similarity = image_features @ text_features.T
    plot_similarity_matrix(
        similarity, 
        top_k=50,
        save_path=str(output_dir / "similarity_matrix.png")
    )
    
    # 3. Feature distributions
    print("4. Plotting image feature distribution...")
    plot_feature_distribution(
        image_features, 
        "Image Features",
        save_path=str(output_dir / "image_features_distribution.png")
    )
    
    print("5. Plotting text feature distribution...")
    plot_feature_distribution(
        text_features, 
        "Text Features",
        save_path=str(output_dir / "text_features_distribution.png")
    )
    
    print(f"\n✅ All visualizations saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")
