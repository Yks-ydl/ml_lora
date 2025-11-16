"""
Download Winoground dataset to local storage
Run this script once to download the dataset before training
"""

from datasets import load_dataset
from pathlib import Path
import argparse
import json
from PIL import Image
from tqdm import tqdm


def download_winoground(save_dir: str = "data/winoground", auth_token: str = None):
    """
    Download Winoground dataset and save to local directory
    
    Args:
        save_dir: Directory to save the dataset
        auth_token: HuggingFace authentication token (required)
    """
    if not auth_token:
        print("ERROR: HuggingFace auth token is required!")
        print("Get your token from: https://huggingface.co/settings/tokens")
        print("Then run: python download_winoground.py --auth_token YOUR_TOKEN")
        return
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Winoground dataset to {save_path.absolute()}...")
    
    try:
        # Load dataset from HuggingFace
        # Note: use 'token' parameter for newer versions of datasets library
        winoground = load_dataset("facebook/winoground", token=auth_token)["test"]
        
        print(f"Loaded {len(winoground)} examples from Winoground")
        
        # Create subdirectories
        images_dir = save_path / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Save metadata and images
        metadata = []
        
        for idx, example in enumerate(tqdm(winoground, desc="Saving images and metadata")):
            example_id = example["id"]
            
            # Save images (convert to RGB to handle both RGBA and RGB)
            image_0_path = images_dir / f"{example_id}_image_0.jpg"
            image_1_path = images_dir / f"{example_id}_image_1.jpg"
            
            example["image_0"].convert("RGB").save(image_0_path)
            example["image_1"].convert("RGB").save(image_1_path)
            
            # Store metadata
            metadata.append({
                "id": example_id,
                "caption_0": example["caption_0"],
                "caption_1": example["caption_1"],
                "image_0": str(image_0_path.relative_to(save_path)),
                "image_1": str(image_1_path.relative_to(save_path)),
                "tag": example.get("tag", ""),
                "secondary_tag": example.get("secondary_tag", ""),
                "num_main_preds": example.get("num_main_preds", 0),
                "collapsed_tag": example.get("collapsed_tag", "")
            })
        
        # Save metadata to JSON
        metadata_path = save_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Successfully downloaded Winoground dataset!")
        print(f"  - {len(metadata)} examples saved")
        print(f"  - Images saved to: {images_dir.absolute()}")
        print(f"  - Metadata saved to: {metadata_path.absolute()}")
        print(f"\nYou can now use this dataset in training with:")
        print(f"  python train.py --config experiments/exp3_lora_winoground.yaml")
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nMake sure you:")
        print("  1. Have accepted the dataset terms at: https://huggingface.co/datasets/facebook/winoground")
        print("  2. Have a valid auth token from: https://huggingface.co/settings/tokens")
        raise


def verify_dataset(save_dir: str = "data/winoground"):
    """Verify the downloaded dataset"""
    save_path = Path(save_dir)
    
    if not save_path.exists():
        print(f"Dataset not found at {save_path.absolute()}")
        return False
    
    metadata_path = save_path / "metadata.json"
    if not metadata_path.exists():
        print(f"Metadata file not found at {metadata_path.absolute()}")
        return False
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"Found {len(metadata)} examples in dataset")
    
    # Check if images exist
    missing_images = []
    for item in metadata:
        img_0 = save_path / item["image_0"]
        img_1 = save_path / item["image_1"]
        if not img_0.exists():
            missing_images.append(str(img_0))
        if not img_1.exists():
            missing_images.append(str(img_1))
    
    if missing_images:
        print(f"Warning: {len(missing_images)} images are missing")
        return False
    
    print("✓ Dataset verification passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Download Winoground dataset')
    parser.add_argument('--auth_token', type=str, default='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',
                       help='HuggingFace authentication token (required)')
    parser.add_argument('--save_dir', type=str, default='data/winoground',
                       help='Directory to save the dataset')
    parser.add_argument('--verify', action='store_true',
                       help='Verify existing dataset instead of downloading')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset(args.save_dir)
    else:
        download_winoground(args.save_dir, args.auth_token)


if __name__ == "__main__":
    main()
