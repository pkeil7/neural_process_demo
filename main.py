import torch
import torch.optim as optim
import numpy as np
from model import NP_model
from dataset import get_image_dataloader



def main():
    """Main training function."""
    
    # Configuration
    config = {
        'dataset': 'mnist',           # 'mnist' or 'cifar10'
        'max_context_points': 392,  # For MNIST (28x28=784), recommend ~392 (50% of pixels)
        'batch_size': 16,
        'num_workers': 4,
        'input_dim_x': 2,             # 2D coordinates (x, y)
        'input_dim_y': 1,             # Grayscale (1 for MNIST, 3 for CIFAR10)
        'hidden_dim': 128,
        'output_dim': 1,              # Prediction dimension (1 for MNIST, 3 for CIFAR10)
        'learning_rate': 1e-3,
        'num_epochs': 50,
        'save_every': 10,
        'seed': 42,
    }
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader = get_image_dataloader(
        dataset_name=config['dataset'],
        max_context_points=config['max_context_points'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        train=True,
        flatten=False
    )
    
    val_loader = get_image_dataloader(
        dataset_name=config['dataset'],
        max_context_points=config['max_context_points'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        train=False,
        flatten=False
    )
    
    # Create model
    print("\nCreating model...")
    model = NP_model(
        input_dim_x=config['input_dim_x'],
        input_dim_y=config['input_dim_y'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim']
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Train
    print("\n" + "=" * 60)
    train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=config['num_epochs'],
        save_dir='checkpoints',
        save_every=config['save_every']
    )
    
    # Save training history
    np.save('checkpoints/train_losses.npy', train_losses)
    np.save('checkpoints/val_losses.npy', val_losses)
    print("\nTraining history saved to checkpoints/")


if __name__ == '__main__':
    main()