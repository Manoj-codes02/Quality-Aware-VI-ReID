import os
# pyrefly: ignore [missing-import]
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class SYSUDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.images = []
        self.labels = []
        self.cams = []
        
        # Determine subdirectory based on mode. 
        # Standard ReID datasets usually use 'train' for training and 'test' for query/gallery
        sub_dir = 'train' if mode == 'train' else 'test'
        
        base_path = os.path.join(data_dir, sub_dir)
        if not os.path.exists(base_path):
            base_path = data_dir # fallback if unzipped differently
            
        # Modalities to load
        if mode == 'train':
            modalities = ['ir', 'visible']
        elif mode == 'query':
            modalities = ['query'] # mapped to IR logically
        elif mode == 'gallery':
            modalities = ['gallery'] # mapped to visible logically
            
        person_ids = set()
        for mod in modalities:
            mod_path = os.path.join(base_path, mod)
            if os.path.exists(mod_path):
                person_ids.update(os.listdir(mod_path))
        
        person_ids = sorted(list(person_ids))
        pid2label = {pid: i for i, pid in enumerate(person_ids)}
        
        for mod in modalities:
            mod_path = os.path.join(base_path, mod)
            if not os.path.exists(mod_path): continue
            
            for pid in os.listdir(mod_path):
                p_dir = os.path.join(mod_path, pid)
                if not os.path.isdir(p_dir): continue
                
                label = pid2label[pid]
                # Default cam assignments: visible=1, ir=3
                cam = 3 if (mod == 'ir' or mod == 'query') else 1
                
                for file in os.listdir(p_dir):
                    if file.endswith(('.jpg', '.png')):
                        self.images.append(os.path.join(p_dir, file))
                        self.labels.append(label)
                        self.cams.append(cam)
                    
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        cam = self.cams[idx]
        return img, label, cam, img_path

def get_dataloader(data_dir, batch_size=32, mode='train'):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), value='random')
        ])
        shuffle = True
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
        ])
        shuffle = False
        
    dataset = SYSUDataset(data_dir, transform, mode=mode)
    if len(dataset) == 0:
        return None
        
    # We will use random sampler for query/gallery and standard dataloader creation outside
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
