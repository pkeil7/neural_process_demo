"""
Utility functions for visualizing Neural Process predictions.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import NP_model, ConvCNP


def plot_mnist_sample(batch, batch_idx=0, model=None, device='cpu'):
    """
    Plot context and target points for an MNIST sample from a batch.

    This function assumes that the image coordinates are not normalised but are real pixel values
    
    Args:
        batch: Dictionary containing batch data from dataloader
        batch_idx: Index of sample in the batch to visualize
        model: Optional NP_model to also show predictions
        device: Device to run model on
    """
    # Extract data for one sample
    x_context = batch['x_context'][batch_idx]  # [n_context, 2]
    y_context = batch['y_context'][batch_idx]  # [n_context, 1]
    x_target = batch['x_target'][batch_idx]    # [n_target, 2]
    y_target = batch['y_target'][batch_idx]    # [n_target, 1]
    context_mask = batch['context_mask'][batch_idx]  # [n_context]
    target_mask = batch['target_mask'][batch_idx]    # [n_target]
    
    # Filter out padding using masks
    n_context = int(context_mask.sum())
    n_target = int(target_mask.sum())
    
    x_context = x_context[:n_context].cpu().numpy()
    y_context = y_context[:n_context].cpu().numpy()
    x_target = x_target[:n_target].cpu().numpy()
    y_target = y_target[:n_target].cpu().numpy()
    
    # Determine image dimensions (assuming 28x28 for MNIST)
    # Coordinates are normalized to [0, 1], so scale back
    img_h, img_w = 28, 28
    
    # Create figure
    if model is not None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 1. Plot context points (sparse observations)
    context_img = np.full((img_h, img_w, 4), [0.0, 0.0, 1.0, 1.0], dtype=np.float32)  # Blue RGBA background
    for (x_coord, y_coord), val in zip(x_context, y_context):
        # Convert normalized coordinates back to pixel indices
        px = int(x_coord * img_w)
        py = int(y_coord * img_h)
        if 0 <= px < img_w and 0 <= py < img_h:
            v = val[0]
            context_img[py, px] = [v, v, v, 1.0]  # Grayscale pixel over blue background
    
    axes[0].imshow(context_img)
    axes[0].set_title(f'Context Points (n={n_context})')
    axes[0].axis('off')
    
    # 2. Plot target points (ground truth)
    target_img = np.zeros((img_h, img_w))
    for (x_coord, y_coord), val in zip(x_target, y_target):
        px = int(x_coord * img_w)
        py = int(y_coord * img_h)
        if 0 <= px < img_w and 0 <= py < img_h:
            target_img[py, px] = val[0]
    
    axes[1].imshow(target_img, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Target Ground Truth (n={n_target})')
    axes[1].axis('off')
    
    # 3. Plot full image (context + target)
    full_img = np.zeros((img_h, img_w))
    for (x_coord, y_coord), val in zip(x_context, y_context):
        px, py = int(x_coord * img_w), int(y_coord * img_h)
        if 0 <= px < img_w and 0 <= py < img_h:
            full_img[py, px] = val[0]
    for (x_coord, y_coord), val in zip(x_target, y_target):
        px, py = int(x_coord * img_w), int(y_coord * img_h)
        if 0 <= px < img_w and 0 <= py < img_h:
            full_img[py, px] = val[0]
    axes[2].imshow(full_img, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Complete Image')
    axes[2].axis('off')
    
    # 4. Plot predictions if model provided
    if model is not None:
        model.eval()
        with torch.no_grad():
            # Prepare input for model
            x_ctx = batch['x_context'][batch_idx:batch_idx+1].to(device)
            y_ctx = batch['y_context'][batch_idx:batch_idx+1].to(device)
            x_tgt = batch['x_target'][batch_idx:batch_idx+1].to(device)
            ctx_mask = batch['context_mask'][batch_idx:batch_idx+1].to(device)
            
            # Get predictions
            mean, variance = model(x_ctx, y_ctx, x_tgt, context_mask=ctx_mask)
            
            # Extract and apply target mask
            mean = mean[0, :n_target].cpu().numpy()
        
        # Plot predictions: start with context pixels in grayscale
        pred_img = np.zeros((img_h, img_w))
        for (x_coord, y_coord), val in zip(x_context, y_context):
            px, py = int(x_coord * img_w), int(y_coord * img_h)
            if 0 <= px < img_w and 0 <= py < img_h:
                pred_img[py, px] = val[0]
        for (x_coord, y_coord), val in zip(x_target, mean):
            px = int(x_coord * img_w)
            py = int(y_coord * img_h)
            if 0 <= px < img_w and 0 <= py < img_h:
                pred_img[py, px] = val[0]
        
        axes[3].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
        axes[3].set_title('Model Prediction')
        axes[3].axis('off')
    
    plt.tight_layout()
    return fig


def plot_prediction_comparison(batch, batch_idx=0, model=None, device='cpu', save_path=None):
    """
    Detailed comparison plot showing context, target, and prediction side by side.
    
    This function assumes that the image coordinates are not normalised but are real pixel values

    Args:
        batch: Dictionary containing batch data from dataloader
        batch_idx: Index of sample in the batch to visualize
        model: NP_model to generate predictions
        device: Device to run model on
        save_path: Optional path to save figure
    """
    if model is None:
        raise ValueError("Model must be provided for prediction comparison")
    
    # Extract data for one sample
    x_context = batch['x_context'][batch_idx]
    y_context = batch['y_context'][batch_idx]
    x_target = batch['x_target'][batch_idx]
    y_target = batch['y_target'][batch_idx]
    context_mask = batch['context_mask'][batch_idx]
    target_mask = batch['target_mask'][batch_idx]
    
    # Filter out padding
    n_context = int(context_mask.sum())
    n_target = int(target_mask.sum())
    
    x_context = x_context[:n_context].cpu().numpy()
    y_context = y_context[:n_context].cpu().numpy()
    x_target = x_target[:n_target].cpu().numpy()
    y_target = y_target[:n_target].cpu().numpy()
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        x_ctx = batch['x_context'][batch_idx:batch_idx+1].to(device)
        y_ctx = batch['y_context'][batch_idx:batch_idx+1].to(device)
        x_tgt = batch['x_target'][batch_idx:batch_idx+1].to(device)
        ctx_mask = batch['context_mask'][batch_idx:batch_idx+1].to(device)
        
        mean, variance = model(x_ctx, y_ctx, x_tgt, context_mask=ctx_mask)
        mean = mean[0, :n_target].cpu().numpy()
        variance = variance[0, :n_target].cpu().numpy()
    
    # Reconstruct images
    img_h, img_w = 28, 28
    
    # Context only (blue background with grayscale context pixels)
    context_img = np.full((img_h, img_w, 4), [0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    for (x_coord, y_coord), val in zip(x_context, y_context):
        px, py = int(x_coord * img_w), int(y_coord * img_h)
        if 0 <= px < img_w and 0 <= py < img_h:
            v = val[0]
            context_img[py, px] = [v, v, v, 1.0]
    
    # Ground truth (context + target)
    gt_img = np.zeros((img_h, img_w))
    for (x_coord, y_coord), val in zip(x_context, y_context):
        px, py = int(x_coord * img_w), int(y_coord * img_h)
        if 0 <= px < img_w and 0 <= py < img_h:
            gt_img[py, px] = val[0]
    for (x_coord, y_coord), val in zip(x_target, y_target):
        px, py = int(x_coord * img_w), int(y_coord * img_h)
        if 0 <= px < img_w and 0 <= py < img_h:
            gt_img[py, px] = val[0]
    
    # Prediction (context + predicted target)
    pred_img = np.zeros((img_h, img_w))
    for (x_coord, y_coord), val in zip(x_context, y_context):
        px, py = int(x_coord * img_w), int(y_coord * img_h)
        if 0 <= px < img_w and 0 <= py < img_h:
            pred_img[py, px] = val[0]
    for (x_coord, y_coord), val in zip(x_target, mean):
        px, py = int(x_coord * img_w), int(y_coord * img_h)
        if 0 <= px < img_w and 0 <= py < img_h:
            pred_img[py, px] = val[0]
    
    # Uncertainty map
    uncertainty_img = np.zeros((img_h, img_w))
    for (x_coord, y_coord), var in zip(x_target, variance):
        px, py = int(x_coord * img_w), int(y_coord * img_h)
        if 0 <= px < img_w and 0 <= py < img_h:
            uncertainty_img[py, px] = np.sqrt(var[0])  # standard deviation
    
    # Create plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    

    axes[0].imshow(context_img)
    axes[0].set_title(f'Context ({n_context} pixels, {n_context/(img_h*img_w)*100:.1f}%)')
    axes[0].axis('off')
    
    axes[1].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Model Prediction')
    axes[2].axis('off')
    
    im = axes[3].imshow(uncertainty_img, cmap='hot', vmin=0, vmax=0.5)
    axes[3].set_title('Prediction Uncertainty (σ)')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def visualize_training_batch(dataloader, model=None, device='cpu', num_samples=4):
    """
    Visualize multiple samples from a training batch.
    
    Args:
        dataloader: DataLoader to sample from
        model: Optional NP_model for predictions
        device: Device to run model on
        num_samples: Number of samples to visualize
    """
    # Get one batch
    batch = next(iter(dataloader))
    
    # Calculate grid layout
    n_cols = 4 if model is not None else 3
    n_rows = num_samples
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    
    img_h, img_w = 28, 28
    
    for row in range(num_samples):
        # Extract sample data
        x_context = batch['x_context'][row]
        y_context = batch['y_context'][row]
        x_target = batch['x_target'][row]
        y_target = batch['y_target'][row]
        context_mask = batch['context_mask'][row]
        target_mask = batch['target_mask'][row]
        
        n_context = int(context_mask.sum())
        n_target = int(target_mask.sum())
        
        x_context = x_context[:n_context].cpu().numpy()
        y_context = y_context[:n_context].cpu().numpy()
        x_target = x_target[:n_target].cpu().numpy()
        y_target = y_target[:n_target].cpu().numpy()
        
        # Context image
        context_img = np.zeros((img_h, img_w))
        for (x_coord, y_coord), val in zip(x_context, y_context):
            px, py = int(x_coord * img_w), int(y_coord * img_h)
            if 0 <= px < img_w and 0 <= py < img_h:
                context_img[py, px] = val[0]
        
        axes[row, 0].imshow(context_img, cmap='gray', vmin=0, vmax=1)
        if row == 0:
            axes[row, 0].set_title('Context')
        axes[row, 0].axis('off')
        
        # Target ground truth
        target_img = np.zeros((img_h, img_w))
        for (x_coord, y_coord), val in zip(x_target, y_target):
            px, py = int(x_coord * img_w), int(y_coord * img_h)
            if 0 <= px < img_w and 0 <= py < img_h:
                target_img[py, px] = val[0]
        
        axes[row, 1].imshow(target_img, cmap='gray', vmin=0, vmax=1)
        if row == 0:
            axes[row, 1].set_title('Target GT')
        axes[row, 1].axis('off')
        
        # Full image
        full_img = context_img + target_img
        axes[row, 2].imshow(full_img, cmap='gray', vmin=0, vmax=1)
        if row == 0:
            axes[row, 2].set_title('Complete')
        axes[row, 2].axis('off')
        
        # Predictions
        if model is not None:
            model.eval()
            with torch.no_grad():
                x_ctx = batch['x_context'][row:row+1].to(device)
                y_ctx = batch['y_context'][row:row+1].to(device)
                x_tgt = batch['x_target'][row:row+1].to(device)
                ctx_mask = batch['context_mask'][row:row+1].to(device)
                
                mean, _ = model(x_ctx, y_ctx, x_tgt, context_mask=ctx_mask)
                mean = mean[0, :n_target].cpu().numpy()
            
            pred_img = np.copy(context_img)
            for (x_coord, y_coord), val in zip(x_target, mean):
                px, py = int(x_coord * img_w), int(y_coord * img_h)
                if 0 <= px < img_w and 0 <= py < img_h:
                    pred_img[py, px] = val[0]
            
            axes[row, 3].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
            if row == 0:
                axes[row, 3].set_title('Prediction')
            axes[row, 3].axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (NLL)', fontsize=12)
    ax.set_title('Training Progress', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add best validation epoch marker
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = val_losses[best_epoch - 1]
    ax.axvline(best_epoch, color='g', linestyle='--', alpha=0.5, 
               label=f'Best Val (Epoch {best_epoch})')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    return fig


def load_model_from_checkpoint(checkpoint_path, input_dim_x, input_dim_y, hidden_dim, output_dim, device='cpu', model_type='ConvCNP'):
    """
    Load a trained Neural Process model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file (e.g., 'checkpoints/best_model.pt')
        input_dim_x: Input x dimension (e.g., 2 for 2D coordinates)
        input_dim_y: Input y dimension (e.g., 1 for grayscale)
        hidden_dim: Hidden dimension size used during training
        output_dim: Output dimension (e.g., 1 for grayscale)
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Tuple of (model, checkpoint_info)
            - model: Loaded ConvCNP model in eval mode
            - checkpoint_info: Dictionary with epoch, losses, etc.
    """
    # Create model with same architecture

    if model_type == 'NP_model':
        model = NP_model(
            input_dim_x=input_dim_x,
            input_dim_y=input_dim_y,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )

    elif model_type == 'ConvCNP':

        model = ConvCNP(
            img_h=28,
            img_w=28,
            y_dim=input_dim_y,
            hidden_channels=hidden_dim,
            n_conv_layers=3
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Extract checkpoint info
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', None),
        'train_loss': checkpoint.get('train_loss', None),
        'val_loss': checkpoint.get('val_loss', None),
    }
    
    print(f"Loaded model from {checkpoint_path}")
    if checkpoint_info['epoch'] is not None:
        print(f"  Epoch: {checkpoint_info['epoch']}")
        print(f"  Train Loss: {checkpoint_info['train_loss']:.6f}")
        print(f"  Val Loss: {checkpoint_info['val_loss']:.6f}")
    
    return model, checkpoint_info


def load_model_info_from_checkpoint(checkpoint_path,):
    """
    Load only the training information from a checkpoint without the model weights.
    
    Args:
        checkpoint_path: Path to the checkpoint file (e.g., 'checkpoints/best_model.pt')
        device: Device to load on ('cpu' or 'cuda')
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config = checkpoint.get('config', None)
    if config is not None:
        return config
    else :
        print(f"No config found in checkpoint {checkpoint_path}")
        return None
