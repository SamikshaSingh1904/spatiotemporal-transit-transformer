"""Evaluation Pipeline Benchmarking Against Sequential Baselines."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.data.graph_generator import SyntheticTransitODDataset
from src.model.transformer import SpatiotemporalTransitTransformer


class BaselineGRU(nn.Module):
    """Recurrent GRU baseline for edge-level temporal forecasting."""

    def __init__(self, num_nodes: int = 16, window_size: int = 12, horizon: int = 4, hidden_dim: int = 32) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x_flow: torch.Tensor) -> torch.Tensor:
        # x_flow: (B, W, N, N) -> (B * N * N, W, 1)
        b, w, n, _ = x_flow.shape
        x_reshaped = x_flow.permute(0, 2, 3, 1).contiguous().view(b * n * n, w, 1)
        _, h_n = self.gru(x_reshaped)
        out = self.fc(h_n.squeeze(0))  # (B * N * N, H)
        out = out.view(b, n, n, self.horizon).permute(0, 3, 1, 2).contiguous()
        return torch.relu(out)


def run_benchmark() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticTransitODDataset(num_nodes=8, num_timesteps=400, window_size=8, horizon=3)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    model = SpatiotemporalTransitTransformer(num_nodes=8, window_size=8, horizon=3, d_model=32, nhead=2, num_layers=1).to(device)
    baseline = BaselineGRU(num_nodes=8, window_size=8, horizon=3, hidden_dim=32).to(device)

    opt_trans = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt_gru = torch.optim.AdamW(baseline.parameters(), lr=1e-3)
    criterion = nn.L1Loss()  # Mean Absolute Error (MAE)

    print("Training benchmark on synthetic asymmetric transit traces...")
    for epoch in range(3):
        model.train()
        baseline.train()
        for x_flow, y_flow, x_temp, _ in train_loader:
            x_flow, y_flow, x_temp = x_flow.to(device), y_flow.to(device), x_temp.to(device)

            # Transformer step
            opt_trans.zero_grad()
            loss_t = criterion(model(x_flow, x_temp), y_flow)
            loss_t.backward()
            opt_trans.step()

            # Baseline GRU step
            opt_gru.zero_grad()
            loss_g = criterion(baseline(x_flow), y_flow)
            loss_g.backward()
            opt_gru.step()

    # Evaluation
    model.eval()
    baseline.eval()
    total_mae_trans, total_mae_gru = 0.0, 0.0
    with torch.no_grad():
        for x_flow, y_flow, x_temp, _ in test_loader:
            x_flow, y_flow, x_temp = x_flow.to(device), y_flow.to(device), x_temp.to(device)
            total_mae_trans += criterion(model(x_flow, x_temp), y_flow).item() * len(x_flow)
            total_mae_gru += criterion(baseline(x_flow), y_flow).item() * len(x_flow)

    mae_trans = total_mae_trans / len(test_set)
    mae_gru = total_mae_gru / len(test_set)
    improvement = ((mae_gru - mae_trans) / mae_gru) * 100.0

    print(f"Baseline GRU MAE: {mae_gru:.4f}")
    print(f"Spatiotemporal Transformer MAE: {mae_trans:.4f}")
    print(f"Relative Improvement: {improvement:.2f}%")


if __name__ == "__main__":
    run_benchmark()
