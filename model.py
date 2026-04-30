import torch
import torch.nn as nn
import math


class NP_model(nn.Module):
    def __init__(self, input_dim_x, input_dim_y, hidden_dim, output_dim):
        super(NP_model, self).__init__()
        
        # Encoder
        self.encoder = LinearEncoder(input_dim_x, input_dim_y, hidden_dim, latent_dim=hidden_dim)
        self.decoder = LinearDecoder(input_dim_x, latent_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    
    def encode(self, x, y, mask=None):
        # encode samples
        # x, y shapes: [batch, n_context, dim]
        r = self.encoder(x, y)  # [batch, n_context, latent_dim]
        
        # Aggregate over context points (dim=1) with optional masking
        if mask is not None:
            # mask shape: [batch, n_context]
            # Masked mean: sum over valid points, divide by count of valid points
            mask_expanded = mask.unsqueeze(-1)  # [batch, n_context, 1]
            r_masked = r * mask_expanded  # Zero out padded positions
            z = r_masked.sum(dim=1) / mask_expanded.sum(dim=1)  # [batch, latent_dim]
        else:
            # Simple mean over all context points
            z = torch.mean(r, dim=1)  # [batch, latent_dim]
        return z
    
    def decode(self, z, x_target):
        # z shape: [batch, latent_dim]
        # x_target shape: [batch, n_target, input_dim_x]
        return self.decoder(z, x_target)
    
    def forward(self, x, y, x_target, context_mask=None):
        # x, y: [batch, n_context, dim]
        # x_target: [batch, n_target, dim]
        # context_mask: [batch, n_context] (optional)
        z = self.encode(x, y, mask=context_mask)
        mean, variance = self.decode(z, x_target)
        return mean, variance


class LinearEncoder(nn.Module):
    def __init__(self, input_dim_x, input_dim_y, hidden_dim, latent_dim):
        super(LinearEncoder, self).__init__()
        self.x_enc = nn.Linear(input_dim_x, 4)
        self.fc1 = nn.Linear(4 + input_dim_y, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc4 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, y):
        # Encode x
        x_encoded = torch.relu(self.x_enc(x))
        # Concatenate encoded x with y
        xy = torch.cat([x_encoded, y], dim=-1)
        h = torch.relu(self.fc1(xy))
        h = torch.relu(self.fc2(h))
        #h = torch.relu(self.fc3(h))
        #r = self.fc4(h)
        r = h
        return r
    
class LinearDecoder(nn.Module):
    def __init__(self, input_dim_x, latent_dim, hidden_dim, output_dim):
        super(LinearDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim_x + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_var = nn.Linear(hidden_dim, output_dim)
        
        # Initialize variance head to output log(0.1) initially for reasonable starting variance
        self.fc_var.bias.data.fill_(-2.3)  # log(0.1) ≈ -2.3

    def forward(self, z, x_target):
        # z shape: [batch, latent_dim]
        # x_target shape: [batch, n_target, input_dim_x]
        
        # Expand z to match number of target points
        batch_size = x_target.size(0)
        n_target = x_target.size(1)
        z_expanded = z.unsqueeze(1).expand(batch_size, n_target, -1)  # [batch, n_target, latent_dim]
        # Concatenate z with each target point
        zx = torch.cat([z_expanded, x_target], dim=-1)  # [batch, n_target, latent_dim + input_dim_x]
        
        h = torch.relu(self.fc1(zx))
        h = torch.relu(self.fc2(h))
        #h = torch.relu(self.fc3(h))
        mean = self.fc_mean(h)  # [batch, n_target, output_dim]
        
        # Compute variance with minimum bound to prevent collapse
        log_var = self.fc_var(h)
        # Clamp log variance to prevent extreme values
        log_var = torch.clamp(log_var, min=-7, max=7)  # variance ∈ [~0.001, ~1000]
        variance = torch.exp(log_var) + 1e-3  # Add minimum variance of 0.001
        
        return mean, variance


# =============================================================================
# ConvCNP: Convolutional Conditional Neural Process (Gordon et al., 2020)
# =============================================================================

class SetConvEncoder(nn.Module):
    """
    Set Convolution encoder: maps irregular (x, y) pairs to a regular grid
    using RBF kernel weighting.
    
    Produces (y_dim + 1) channels: weighted signal channels + density channel.
    Signal and density are NOT normalized — the downstream CNN learns to
    interpret the density channel.
    """
    
    def __init__(self, y_dim, grid_h, grid_w, init_lengthscale=0.1):
        super(SetConvEncoder, self).__init__()
        self.y_dim = y_dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        
        # Learnable log-lengthscale for the RBF kernel
        self.log_lengthscale = nn.Parameter(
            torch.tensor(math.log(init_lengthscale), dtype=torch.float32)
        )
        
        # Fixed grid coordinates matching dataset convention: arange(n)/n
        yy, xx = torch.meshgrid(
            torch.arange(grid_h, dtype=torch.float32) / grid_h,
            torch.arange(grid_w, dtype=torch.float32) / grid_w,
            indexing='ij'
        )
        grid = torch.stack([xx, yy], dim=-1)  # [H, W, 2] — (col/w, row/h)
        self.register_buffer('grid', grid)
    
    def forward(self, x_context, y_context, context_mask=None):
        """
        Args:
            x_context: [B, N, 2] context coordinates
            y_context: [B, N, C] context values
            context_mask: [B, N] optional mask (1=valid, 0=padding)

        Here B is batch size, N is number of context points, C is number of signal channels (e.g. 1 for grayscale).
        
        Returns:
            grid_repr: [B, C+1, H, W] grid with signal + density channels
        """
        batch_size = x_context.size(0)
        
        # Grid points flattened: [H*W, 2]
        grid_flat = self.grid.reshape(-1, 2)
        
        # Pairwise squared distances between grid points H*W and context points N: [B, H*W, N]
        # grid_flat[None, :, None, :] → [1, H*W, 1, 2]
        # x_context[:, None, :, :]   → [B, 1,   N, 2]
        sq_dist = ((grid_flat[None, :, None, :] - x_context[:, None, :, :]) ** 2).sum(-1)
        
        lengthscale = torch.exp(self.log_lengthscale)
        weights = torch.exp(-0.5 * sq_dist / (lengthscale ** 2))  # [B, H*W, N]
        
        if context_mask is not None:
            weights = weights * context_mask.unsqueeze(1)  # mask out padded points
        
        # Signal channels: Σ w_i * y_i → [B, H*W, C]
        # matrix multiply weights [B, H*W, N] with y_context [B, N, C] → [B, H*W, C]
        # apply weights to y_context to get weighted sum of context values at each grid point
        signal = torch.bmm(weights, y_context)
        
        # Density channel: Σ w_i → [B, H*W, 1]
        # sum weights to get density at each grid point
        density = weights.sum(-1, keepdim=True)
        
        # Combine and reshape to [B, C+1, H, W]
        grid_repr = torch.cat([signal, density], dim=-1)  # [B, H*W, C+1]
        grid_repr = grid_repr.reshape(batch_size, self.grid_h, self.grid_w, -1)
        grid_repr = grid_repr.permute(0, 3, 1, 2)  # [B, C+1, H, W]
        
        return grid_repr


class ConvCNP(nn.Module):
    """
    Convolutional Conditional Neural Process (ConvCNP).
    
    SetConv encoder → CNN on grid → index predictions at target locations.
    
    For image tasks the targets lie on the pixel grid, so decoding is a
    simple grid lookup (no SetConv decoder needed).
    
    Reference: Gordon et al. (2020) "Convolutional Conditional Neural Processes"
    """
    
    def __init__(self, img_h, img_w, y_dim=1, hidden_channels=128,
                 n_conv_layers=6, init_lengthscale=0.1):
        super(ConvCNP, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.y_dim = y_dim
        
        # SetConv encoder
        self.encoder = SetConvEncoder(y_dim, img_h, img_w, init_lengthscale)
        
        # CNN: (y_dim + 1) → hidden → … → 2*y_dim  (mean + log_var)
        layers = []
        in_ch = y_dim + 1
        for i in range(n_conv_layers):
            layers.append(nn.Conv2d(in_ch, hidden_channels, kernel_size=5, padding=2))
            layers.append(nn.ReLU())
            in_ch = hidden_channels
        layers.append(nn.Conv2d(hidden_channels, 2 * y_dim, kernel_size=1))
        self.cnn = nn.Sequential(*layers)
        
        # Initialize final conv bias for log_var head to ≈ log(0.1)
        final_conv = self.cnn[-1]
        final_conv.bias.data[y_dim:].fill_(-2.3)
    
    def _grid_predictions(self, x_context, y_context, context_mask=None):
        """Internal: encode → CNN → split into mean and variance grids.
        This contains the learnable parameters of the model.
        """
        grid_repr = self.encoder(x_context, y_context, context_mask)
        out = self.cnn(grid_repr)  # [B, 2*C, H, W]
        
        mean_grid = out[:, :self.y_dim]    # [B, C, H, W]
        logvar_grid = out[:, self.y_dim:]  # [B, C, H, W]
        
        logvar_grid = torch.clamp(logvar_grid, min=-7, max=7)
        variance_grid = torch.exp(logvar_grid) + 1e-3
        
        return mean_grid, variance_grid
    
    def _index_grid(self, grid, x_target):
        """
        Extract values from a [B, C, H, W] grid at target coordinates.
        
        Dataset coords: col_idx / W, row_idx / H  →  multiply back to get index for all targets in batch Nt.

        Basically just a helper function that does grid indexing for the forward pass and does not contain any learnable parameters.
        """
        batch_size = x_target.size(0)
        
        target_col = (x_target[..., 0] * self.img_w).round().long().clamp(0, self.img_w - 1)
        target_row = (x_target[..., 1] * self.img_h).round().long().clamp(0, self.img_h - 1)
        
        flat_idx = target_row * self.img_w + target_col  # [B, N_t]
        flat_idx = flat_idx.unsqueeze(1).expand(-1, self.y_dim, -1)  # [B, C, N_t]
        
        grid_flat = grid.reshape(batch_size, self.y_dim, -1)  # [B, C, H*W]
        values = torch.gather(grid_flat, 2, flat_idx)  # [B, C, N_t]
        
        return values.permute(0, 2, 1)  # [B, N_t, C]
    
    def forward(self, x_context, y_context, x_target, context_mask=None):
        """
        Args:
            x_context:    [B, N_c, 2]
            y_context:    [B, N_c, C]
            x_target:     [B, N_t, 2]
            context_mask: [B, N_c] optional (1=valid, 0=padding)
        
        Returns:
            mean:     [B, N_t, C]
            variance: [B, N_t, C]
        """
        mean_grid, variance_grid = self._grid_predictions(
            x_context, y_context, context_mask
        )
        
        mean = self._index_grid(mean_grid, x_target)
        variance = self._index_grid(variance_grid, x_target)
        
        return mean, variance
    
    def predict_grid(self, x_context, y_context, context_mask=None):
        """
        Full-image prediction for visualization.
        
        Returns:
            mean:     [B, C, H, W]
            variance: [B, C, H, W]
        """
        return self._grid_predictions(x_context, y_context, context_mask)