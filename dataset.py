import os

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from color_theory_fashion import SkinToneCategory, Undertone


class FashionColorDataset(Dataset):
    """Dataset for skin tone and undertone classification"""
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Load dataset (assuming directory structure: root/skin_type/undertone/images)
        for skin_dir in os.listdir(root_dir):
            skin_type = SkinToneCategory[skin_dir.upper()]
            for undertone_dir in os.listdir(os.path.join(root_dir, skin_dir)):
                undertone_type = Undertone[undertone_dir.upper()]
                img_dir = os.path.join(root_dir, skin_dir, undertone_dir)
                
                for img_file in os.listdir(img_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append({
                            'path': os.path.join(img_dir, img_file),
                            'skin_tone': skin_type.value,
                            'undertone': undertone_type.value
                        })
        
        # Split dataset (80% train, 10% val, 10% test)
        np.random.shuffle(self.samples)
        split_idx = int(len(self.samples) * 0.8)
        if split == 'train':
            self.samples = self.samples[:split_idx]
        elif split == 'val':
            self.samples = self.samples[split_idx:int(split_idx*1.125)]
        else:  # test
            self.samples = self.samples[int(split_idx*1.125):]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'skin_tone': torch.tensor(sample['skin_tone'], dtype=torch.long),
            'undertone': torch.tensor(sample['undertone'], dtype=torch.long)
        }

def get_transforms(mode='train'):
    """Get image transformations for different phases"""
    if mode == 'train':
        return A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:  # validation/test
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def create_data_loaders(data_dir, batch_size=32):
    """Create data loaders for all splits"""
    train_dataset = FashionColorDataset(
        data_dir, transform=get_transforms('train'), split='train'
    )
    val_dataset = FashionColorDataset(
        data_dir, transform=get_transforms('val'), split='val'
    )
    test_dataset = FashionColorDataset(
        data_dir, transform=get_transforms('val'), split='test'
    )
    
    return {
        'train': DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2),
        'test': DataLoader(test_dataset, batch_size, shuffle=False, num_workers=2)
    }