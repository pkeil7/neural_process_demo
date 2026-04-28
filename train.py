"""
Training script for Neural Process on image datasets.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

from model import NP_model
from dataset import get_image_dataloader


def gaussian_nll_loss(mean, variance, target, target_mask=None):
    """
    Compute Gaussian negative log-likelihood loss.
    
    Args:
        mean: Predicted mean [batch, n_target, output_dim]
        variance: Predicted variance [batch, n_target, output_dim]
        target: Ground truth targets [batch, n_target, output_dim]
        target_mask: Optional mask for padded targets [batch, n_target]
    
    Returns:
        Scalar loss value
    """
    # Gaussian NLL: 0.5 * (log(var) + (y - mean)^2 / var + log(2π))
    # We can drop the constant log(2π) term
    # Clamp variance to prevent log(0) issues
    variance_safe = torch.clamp(variance, min=1e-6)
    nll = 0.5 * (torch.log(variance_safe) + (target - mean) ** 2 / variance_safe)
    
    if target_mask is not None:
        # Apply mask and compute mean over valid targets only
        mask_expanded = target_mask.unsqueeze(-1)  # [batch, n_target, 1]
        nll = nll * mask_expanded
        loss = nll.sum() / mask_expanded.sum()
    else:
        loss = nll.mean()
    
    return loss


def train_epoch(model, dataloader, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: Neural Process model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Move batch to device
        x_context = batch['x_context'].to(device)
        y_context = batch['y_context'].to(device)
        x_target = batch['x_target'].to(device)
        y_target = batch['y_target'].to(device)
        context_mask = batch['context_mask'].to(device)
        target_mask = batch['target_mask'].to(device)
        
        # Forward pass
        mean, variance = model(x_context, y_context, x_target, context_mask=context_mask)
        
        # Compute loss
        loss = gaussian_nll_loss(mean, variance, y_target, target_mask=target_mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: Neural Process model
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
    
    Returns:
        Average loss for the dataset
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move batch to device
            x_context = batch['x_context'].to(device)
            y_context = batch['y_context'].to(device)
            x_target = batch['x_target'].to(device)
            y_target = batch['y_target'].to(device)
            context_mask = batch['context_mask'].to(device)
            target_mask = batch['target_mask'].to(device)
            
            # Forward pass
            mean, variance = model(x_context, y_context, x_target, context_mask=context_mask)
            
            # Compute loss
            loss = gaussian_nll_loss(mean, variance, y_target, target_mask=target_mask)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    save_dir='checkpoints',
    save_every=10,
    model_name='best_model.pt'
):
    """
    Full training loop.
    
    Args:
        model: Neural Process model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs to train
        save_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
    """
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    print("-" * 60)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_dir, model_name))
            print(f"  → Saved best model (val_loss: {val_loss:.6f})")
        
        # Save periodic checkpoint
        if epoch % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    print("-" * 60)
    print(f"Training complete! Best validation loss: {best_val_loss:.6f}")
    
    return train_losses, val_losses


