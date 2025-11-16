"""
Winoground Dataset Loader
Handles the challenging Winoground dataset for compositional reasoning
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from typing import Dict, List, Tuple
import random
import json
from pathlib import Path


class WinogroundDataset(Dataset):
    """
    Winoground dataset for compositional visual reasoning
    Each example has 2 images and 2 captions with same words but different meanings
    """
    
    def __init__(
        self, 
        split: str = "test",
        use_hard_negatives: bool = True,
        local_data_path: str = None,
        auth_token: str = None
    ):
        """
        Args:
            split: Dataset split (Winoground only has 'test')
            use_hard_negatives: Whether to include hard negatives for training
            local_data_path: Path to locally downloaded dataset (if None, download from HuggingFace)
            auth_token: HuggingFace auth token (only needed if downloading from HuggingFace)
        """
        self.use_hard_negatives = use_hard_negatives
        
        # Load from local or HuggingFace
        if local_data_path and Path(local_data_path).exists():
            print(f"Loading Winoground from local path: {local_data_path}")
            self.load_from_local(local_data_path)
        else:
            print("Loading Winoground from HuggingFace...")
            if not auth_token:
                print("Warning: No auth_token provided. This may fail if you haven't accepted dataset terms.")
                print("Get token from: https://huggingface.co/settings/tokens")
            self.dataset = load_dataset("facebook/winoground", split=split, token=auth_token)
            self.local_mode = False
    
    def load_from_local(self, data_path: str):
        """Load dataset from local directory"""
        data_path = Path(data_path)
        metadata_path = data_path / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                f"Please download the dataset first using: python download_winoground.py --auth_token YOUR_TOKEN"
            )
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.data_path = data_path
        self.local_mode = True
        print(f"Loaded {len(self.metadata)} examples from local storage")
        
    def __len__(self):
        if self.local_mode:
            return len(self.metadata)
        else:
            return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a Winoground example with positive and negative pairs
        
        Winoground structure:
        - image_0: First image
        - image_1: Second image
        - caption_0: Caption for image_0
        - caption_1: Caption for image_1
        
        The challenge: caption_0 and caption_1 use same words but different order
        """
        if self.local_mode:
            # Load from local files
            item = self.metadata[idx]
            
            # Load images from local storage
            image_0_path = self.data_path / item['image_0']
            image_1_path = self.data_path / item['image_1']
            
            image_0 = Image.open(image_0_path).convert("RGB")
            image_1 = Image.open(image_1_path).convert("RGB")
            
            caption_0 = item['caption_0']
            caption_1 = item['caption_1']
        else:
            # Load from HuggingFace dataset
            item = self.dataset[idx]
            image_0 = item['image_0']
            image_1 = item['image_1']
            caption_0 = item['caption_0']
            caption_1 = item['caption_1']
        
        # Create positive pairs
        # We want: image_0 matches caption_0, image_1 matches caption_1
        
        if self.use_hard_negatives:
            # Hard negative: the OTHER caption (same words, wrong order)
            return {
                'image_0': image_0,
                'image_1': image_1,
                'caption_0': caption_0,
                'caption_1': caption_1,
                'hard_negative_0': caption_1,  # Hard negative for image_0
                'hard_negative_1': caption_0,  # Hard negative for image_1
            }
        else:
            return {
                'image_0': image_0,
                'image_1': image_1,
                'caption_0': caption_0,
                'caption_1': caption_1,
            }
    
    def get_evaluation_format(self, idx: int) -> Dict:
        """
        Get example in evaluation format
        Returns all 4 combinations for scoring
        """
        if self.local_mode:
            item = self.metadata[idx]
            
            # Load images from local storage
            image_0_path = self.data_path / item['image_0']
            image_1_path = self.data_path / item['image_1']
            
            image_0 = Image.open(image_0_path).convert("RGB")
            image_1 = Image.open(image_1_path).convert("RGB")
            
            return {
                'id': item['id'],
                'image_0': image_0,
                'image_1': image_1,
                'caption_0': item['caption_0'],
                'caption_1': item['caption_1'],
            }
        else:
            item = self.dataset[idx]
            
            return {
                'id': idx,
                'image_0': item['image_0'],
                'image_1': item['image_1'],
                'caption_0': item['caption_0'],
                'caption_1': item['caption_1'],
                # Ground truth: (image_0, caption_0) and (image_1, caption_1) are correct
            }


class Flickr30kDataset(Dataset):
    """
    Flickr30k dataset for image-text retrieval
    Large-scale dataset with 31k images and 5 captions each
    """
    
    def __init__(
        self,
        split: str = "train",
        augment: bool = True
    ):
        """
        Args:
            split: 'train', 'val', or 'test'
            augment: Whether to apply data augmentation
        """
        self.dataset = load_dataset("nlphuji/flickr30k", split=split)
        self.augment = augment
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns (image, caption) pair
        Each image has 5 captions, randomly select one
        """
        item = self.dataset[idx]
        image = item['image']
        
        # Randomly select one of 5 captions
        captions = item['caption']
        caption = random.choice(captions)
        
        return image, caption


class MSCOCODataset(Dataset):
    """
    MS COCO dataset for image-text retrieval
    Large-scale dataset with rich annotations
    """
    
    def __init__(
        self,
        split: str = "train",
        year: str = "2017"
    ):
        """
        Args:
            split: 'train', 'val', or 'test'
            year: COCO dataset year
        """
        # Load COCO captions
        if split == "train":
            self.dataset = load_dataset("HuggingFaceM4/COCO", split=f"train")
        else:
            self.dataset = load_dataset("HuggingFaceM4/COCO", split=f"validation")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple:
        """Returns (image, caption) pair"""
        item = self.dataset[idx]
        
        # COCO has multiple captions per image
        image = item['image']
        
        # Get first caption (or randomly select if multiple)
        if isinstance(item['sentences']['raw'], list):
            caption = random.choice(item['sentences']['raw'])
        else:
            caption = item['sentences']['raw']
        
        return image, caption


def collate_fn_winoground(batch: List[Dict]) -> Dict:
    """
    Custom collate function for Winoground dataset
    Handles the special structure with paired images and captions
    """
    # Collect all images and captions
    images = []
    captions = []
    hard_negatives = []
    
    for item in batch:
        images.extend([item['image_0'], item['image_1']])
        captions.extend([item['caption_0'], item['caption_1']])
        
        if 'hard_negative_0' in item:
            hard_negatives.extend([item['hard_negative_0'], item['hard_negative_1']])
    
    if hard_negatives:
        return images, captions, hard_negatives
    else:
        return images, captions


def collate_fn_standard(batch: List[Tuple]) -> Tuple:
    """
    Standard collate function for regular image-text datasets
    """
    images, captions = zip(*batch)
    return list(images), list(captions)


def get_dataloaders(
    dataset_name: str = "winoground",
    batch_size: int = 32,
    num_workers: int = 4,
    use_hard_negatives: bool = True,
    local_data_path: str = None,
    auth_token: str = None
) -> Dict[str, DataLoader]:
    """
    Get dataloaders for specified dataset
    
    Args:
        dataset_name: 'winoground', 'flickr30k', or 'mscoco'
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_hard_negatives: For Winoground, whether to use hard negatives
        local_data_path: Path to local dataset (for Winoground)
        auth_token: HuggingFace auth token (for downloading from HuggingFace)
        
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    
    if dataset_name == "winoground":
        # Winoground only has test set, we'll split it
        full_dataset = WinogroundDataset(
            use_hard_negatives=use_hard_negatives,
            local_data_path=local_data_path,
            auth_token=auth_token
        )
        
        # Split: 60% train, 20% val, 20% test
        train_size = int(0.6 * len(full_dataset))
        val_size = int(0.2 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        collate = collate_fn_winoground
        
    elif dataset_name == "flickr30k":
        train_dataset = Flickr30kDataset(split="train")
        val_dataset = Flickr30kDataset(split="validation")
        test_dataset = Flickr30kDataset(split="test")
        
        collate = collate_fn_standard
        
    elif dataset_name == "mscoco":
        train_dataset = MSCOCODataset(split="train")
        val_dataset = MSCOCODataset(split="val")
        test_dataset = val_dataset  # COCO test labels not public
        
        collate = collate_fn_standard
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate
        ),
    }
    
    return dataloaders


if __name__ == "__main__":
    print("Testing Winoground dataset...")
    
    dataset = WinogroundDataset()
    print(f"Dataset size: {len(dataset)}")
    
    # Get first example
    example = dataset[0]
    print(f"\nExample keys: {example.keys()}")
    print(f"Caption 0: {example['caption_0']}")
    print(f"Caption 1: {example['caption_1']}")
    print(f"Hard negative 0: {example['hard_negative_0']}")
    
    print("\nTesting DataLoader...")
    loaders = get_dataloaders("winoground", batch_size=4)
    batch = next(iter(loaders['train']))
    print(f"Batch structure: {len(batch)} elements")
    print(f"Number of images: {len(batch[0])}")
    print(f"Number of captions: {len(batch[1])}")
