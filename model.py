import torch
import torch.nn as nn


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
        self.x_enc = nn.Linear(input_dim_x, 16)
        self.fc1 = nn.Linear(16 + input_dim_y, hidden_dim)
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