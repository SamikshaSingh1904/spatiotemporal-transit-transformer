"""Interactive demonstration for Spatiotemporal Transit Transformer."""

import torch
from src.data.graph_generator import SyntheticTransitODDataset
from src.model.transformer import SpatiotemporalTransitTransformer


def main() -> None:
    print("=== Spatiotemporal Transit Transformer Quickstart Demo ===")
    num_nodes = 8
    window_size = 12
    horizon = 4

    print(f"1. Generating synthetic asymmetric transit network with {num_nodes} stations...")
    dataset = SyntheticTransitODDataset(
        num_nodes=num_nodes,
        num_timesteps=400,
        window_size=window_size,
        horizon=horizon,
    )
    x_flow, y_flow, x_temp, y_temp = dataset[0]
    print(f"   Historical input flow tensor shape: {x_flow.shape} (Window={window_size}, Nodes={num_nodes})")
    print(f"   Target forecast flow tensor shape:  {y_flow.shape} (Horizon={horizon}, Nodes={num_nodes})")

    print("\n2. Instantiating Spatiotemporal Transformer with decoupled spatial embeddings...")
    model = SpatiotemporalTransitTransformer(
        num_nodes=num_nodes,
        window_size=window_size,
        horizon=horizon,
        spatial_dim=16,
        d_model=32,
        nhead=2,
        num_layers=2,
    )

    print("\n3. Performing forward prediction step...")
    model.eval()
    with torch.no_grad():
        x_flow_batch = x_flow.unsqueeze(0)  # (1, W, N, N)
        x_temp_batch = x_temp.unsqueeze(0)  # (1, W, 2)
        pred = model(x_flow_batch, x_temp_batch)

    print(f"   Forecast tensor shape: {pred.shape} [Batch=1, Horizon={horizon}, N={num_nodes}, N={num_nodes}]")
    print(f"   Sample predicted OD flow (Station 0 -> Station 1): {pred[0, 0, 0, 1].item():.2f} passengers/interval")
    print(f"   Sample predicted reverse flow (Station 1 -> Station 0): {pred[0, 0, 1, 0].item():.2f} passengers/interval")
    print("\n✓ Demo completed successfully!")


if __name__ == "__main__":
    main()
