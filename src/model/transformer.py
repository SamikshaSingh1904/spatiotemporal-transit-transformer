"""Spatiotemporal Transformer for Asymmetric Route Demand Forecasting."""

import torch
import torch.nn as nn
from src.model.embeddings import DecoupledSpatialEmbedding


class SpatiotemporalTransitTransformer(nn.Module):
    """End-to-end spatiotemporal forecasting architecture.
    
    Tokenizes temporal windows of asymmetric OD matrices, projects disjoint 
    directional spatial representations, and performs multi-head self-attention.
    """

    def __init__(
        self,
        num_nodes: int = 16,
        window_size: int = 12,
        horizon: int = 4,
        spatial_dim: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.horizon = horizon
        self.num_pairs = num_nodes * num_nodes

        # Spatial disjoint embedding (origin / dest)
        self.spatial_embed = DecoupledSpatialEmbedding(num_nodes=num_nodes, embed_dim=spatial_dim)

        # Tokenizer projection: maps [flow_scalar + 2*spatial_dim + 2_temporal] to d_model
        self.input_projection = nn.Linear(1 + (2 * spatial_dim) + 2, d_model)

        # Positional encodings for historical time steps
        self.pos_encoder = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)

        # Standard transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projection head to multi-step forecast horizon
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model * window_size, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon),
            nn.ReLU(),  # Demand non-negativity constraint
        )

    def forward(self, x_flow: torch.Tensor, x_temp: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x_flow: (B, W, N, N) Historical origin-destination flows
            x_temp: (B, W, 2) Historical temporal features
            
        Returns:
            torch.Tensor: (B, H, N, N) Forecasted origin-destination flows
        """
        batch_size = x_flow.shape[0]

        # 1. Get static directed spatial representations: (N, N, 2 * spatial_dim)
        spatial_repr = self.spatial_embed()
        spatial_repr = spatial_repr.view(1, 1, self.num_pairs, -1)
        spatial_repr = spatial_repr.expand(batch_size, self.window_size, -1, -1)

        # 2. Reshape flows: (B, W, N*N, 1)
        flat_flows = x_flow.view(batch_size, self.window_size, self.num_pairs, 1)

        # 3. Broadcast temporal features: (B, W, N*N, 2)
        flat_temp = x_temp.unsqueeze(2).expand(-1, -1, self.num_pairs, -1)

        # 4. Tokenize & project: (B * num_pairs, W, token_dim) -> (B * num_pairs, W, d_model)
        combined_tokens = torch.cat([flat_flows, spatial_repr, flat_temp], dim=-1)
        combined_tokens = combined_tokens.permute(0, 2, 1, 3).contiguous()
        combined_tokens = combined_tokens.view(batch_size * self.num_pairs, self.window_size, -1)

        h = self.input_projection(combined_tokens) + self.pos_encoder

        # 5. Spatiotemporal Sequence Attention
        encoded = self.transformer_encoder(h)  # (B * num_pairs, W, d_model)

        # 6. Forecast Horizon Projection
        encoded_flat = encoded.view(batch_size * self.num_pairs, -1)
        out = self.forecast_head(encoded_flat)  # (B * num_pairs, H)

        # 7. Reshape to target shape: (B, H, N, N)
        out = out.view(batch_size, self.num_nodes, self.num_nodes, self.horizon)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out
