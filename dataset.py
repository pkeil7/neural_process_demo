import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class ImageDataset(Dataset):
    """
    Dataset that returns position coordinates (x) and image values (y).
    Randomly splits pixels into context and target sets.
    
    Args:
        dataset: torchvision dataset (e.g., MNIST, CIFAR10)
        context_ratio: float in [0, 1], ratio of pixels to use as context
        max_context_points: int, maximum number of context points (caps context_ratio)
        flatten: bool, whether to flatten images to 1D or keep as 2D coordinates
    """
    
    def __init__(self, dataset, context_ratio=None, max_context_points=None, flatten=False):
        self.dataset = dataset
        self.context_ratio = context_ratio
        self.max_context_points = max_context_points
        self.flatten = flatten
        image, _ = self.dataset[0]
        image = np.array(image)
        
        # Handle different image shapes
        if image.ndim == 2:  # Grayscale (H, W)
            h, w = image.shape
            c = 1
            image = image.reshape(h, w, 1)
        else:  # Color (C, H, W)
            c, h, w = image.shape

        print("Image dimensions (H, W, C):", (h, w, c))
        self.h = h
        self.w = w
        self.c = c

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        image, _ = self.dataset[idx]
        image = np.array(image)
        h, w, c = self.h, self.w, self.c
        # Normalize image values to [0, 1]
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
        
        # Create coordinate grid
        if self.flatten:
            # Flatten image to 1D
            num_pixels = h * w * c
            x = np.arange(num_pixels).reshape(-1, 1).astype(np.float32)
            y = image.flatten().astype(np.float32)
        else:
            # Use 2D spatial coordinates for each pixel
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            x = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
            # Normalize coordinates to [0, 1]
            x[:, 0] /= w
            x[:, 1] /= h
            y = image.reshape(-1, c).astype(np.float32)
        
        num_pixels = len(x)
        if self.context_ratio is not None:
            assert 0 < self.context_ratio < 1, "context_ratio must be in (0, 1)"
            # Randomly split into context and target based on context_ratio
            num_context = int(num_pixels * self.context_ratio)
        else:
            # Default to random split
            if self.max_context_points is not None:
                assert self.max_context_points <= num_pixels, "max_context_points must be less than total pixels"
                num_context = np.random.randint(1, self.max_context_points)
            else:
                num_context = np.random.randint(1, num_pixels)
        

        
        indices = np.random.permutation(num_pixels)
        context_indices = indices[:num_context]
        target_indices = indices[num_context:]
        
        x_context = torch.from_numpy(x[context_indices])
        y_context = torch.from_numpy(y[context_indices])
        x_target = torch.from_numpy(x[target_indices])
        y_target = torch.from_numpy(y[target_indices])
        
        return {
            'x_context': x_context,
            'y_context': y_context,
            'x_target': x_target,
            'y_target': y_target
        }


def neural_process_collate_fn(batch):
    """
    Custom collate function to handle variable-length context and target sets.
    Pads sequences to the maximum length in the batch and creates attention masks.
    
    Args:
        batch: List of dictionaries from ImageDataset.__getitem__
        
    Returns:
        Dictionary with padded tensors and masks:
            - x_context: [batch, max_context, x_dim]
            - y_context: [batch, max_context, y_dim]
            - x_target: [batch, max_target, x_dim]
            - y_target: [batch, max_target, y_dim]
            - context_mask: [batch, max_context] - 1 for real data, 0 for padding
            - target_mask: [batch, max_target] - 1 for real data, 0 for padding
    """
    # Find max lengths in this batch
    max_context = max(item['x_context'].size(0) for item in batch)
    max_target = max(item['x_target'].size(0) for item in batch)
    
    batch_size = len(batch)
    x_dim = batch[0]['x_context'].size(-1)
    y_dim = batch[0]['y_context'].size(-1)
    
    # Initialize padded tensors
    x_context_padded = torch.zeros(batch_size, max_context, x_dim)
    y_context_padded = torch.zeros(batch_size, max_context, y_dim)
    x_target_padded = torch.zeros(batch_size, max_target, x_dim)
    y_target_padded = torch.zeros(batch_size, max_target, y_dim)
    
    # Initialize masks (1 for real data, 0 for padding)
    context_mask = torch.zeros(batch_size, max_context)
    target_mask = torch.zeros(batch_size, max_target)
    
    # Fill in the data
    for i, item in enumerate(batch):
        n_context = item['x_context'].size(0)
        n_target = item['x_target'].size(0)
        
        x_context_padded[i, :n_context] = item['x_context']
        y_context_padded[i, :n_context] = item['y_context']
        x_target_padded[i, :n_target] = item['x_target']
        y_target_padded[i, :n_target] = item['y_target']
        
        context_mask[i, :n_context] = 1
        target_mask[i, :n_target] = 1
    
    return {
        'x_context': x_context_padded,
        'y_context': y_context_padded,
        'x_target': x_target_padded,
        'y_target': y_target_padded,
        'context_mask': context_mask,
        'target_mask': target_mask
    }


def get_image_dataloader(dataset_name='mnist', context_ratio=0.5, max_context_points=None,
                         batch_size=32, num_workers=0, train=True, flatten=False, **kwargs):
    """
    Create a DataLoader for image datasets.
    
    Args:
        dataset_name: str, 'mnist' or 'cifar10'
        context_ratio: float in [0, 1], ratio of context pixels to total pixels
        max_context_points: int, maximum number of context points (caps context_ratio)
                            For MNIST (28x28=784), recommend ~392 (50% of pixels)
        batch_size: int, batch size
        num_workers: int, number of workers for DataLoader
        train: bool, whether to use training or test set
        flatten: bool, whether to flatten images to 1D
        **kwargs: additional arguments for the dataset
    
    Returns:
        DataLoader object
    """
    if dataset_name.lower() == 'mnist':
        dataset = MNIST(root='./data', train=train, download=True, 
                       transform=transforms.ToTensor())
    elif dataset_name.lower() == 'cifar10':
        dataset = CIFAR10(root='./data', train=train, download=True,
                         transform=transforms.ToTensor())
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    image_dataset = ImageDataset(dataset, context_ratio=context_ratio, 
                                 max_context_points=max_context_points, flatten=flatten)
    
    dataloader = DataLoader(image_dataset, batch_size=batch_size, 
                           num_workers=num_workers, shuffle=train,
                           collate_fn=neural_process_collate_fn)
    
    return dataloader
