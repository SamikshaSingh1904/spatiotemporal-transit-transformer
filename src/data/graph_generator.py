"""Synthetic Directed Transit Network & Asymmetric OD Trip Matrix Generator."""

from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticTransitODDataset(Dataset):
    """Generates synthetic origin-destination transit demand tensors over time.
    
    Produces tensors of shape:
        - od_flows: (num_timesteps, num_nodes, num_nodes)
        - temporal_features: (num_timesteps, 2) [normalized hour, day_of_week]
    """

    def __init__(
        self,
        num_nodes: int = 16,
        num_timesteps: int = 2880,  # 30 days of 15-minute intervals
        window_size: int = 12,      # 3 hours historical window
        horizon: int = 4,           # 1 hour forecast horizon
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_timesteps = num_timesteps
        self.window_size = window_size
        self.horizon = horizon

        np.random.seed(seed)
        self.od_flows, self.temporal_features = self._generate_synthetic_network()

    def _generate_synthetic_network(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Asymmetric base gravity matrix: G[i, j] != G[j, i]
        base_potentials = np.random.gamma(shape=2.0, scale=2.0, size=(self.num_nodes, self.num_nodes))
        np.fill_diagonal(base_potentials, 0.0)

        # Diurnal morning & evening rush-hour curves
        t = np.arange(self.num_timesteps)
        hour_of_day = (t % 96) / 4.0  # 96 15-minute intervals per day
        morning_peak = np.exp(-0.5 * ((hour_of_day - 8.0) / 1.5) ** 2)
        evening_peak = np.exp(-0.5 * ((hour_of_day - 17.5) / 1.8) ** 2)
        diurnal_factor = 0.2 + morning_peak + evening_peak  # (T,)

        # Modulate asymmetric directional flows (e.g., suburb -> downtown AM, reverse PM)
        asymmetry_mask = np.random.uniform(0.5, 1.5, size=(self.num_nodes, self.num_nodes))
        flows = np.zeros((self.num_timesteps, self.num_nodes, self.num_nodes), dtype=np.float32)

        for step in range(self.num_timesteps):
            poisson_lambda = (
                base_potentials * asymmetry_mask * diurnal_factor[step]
                + np.random.exponential(scale=0.1, size=(self.num_nodes, self.num_nodes))
            )
            flows[step] = np.random.poisson(poisson_lambda)

        # Temporal features: hour normalized [0, 1] and day of week normalized [0, 1]
        temporal_feats = np.zeros((self.num_timesteps, 2), dtype=np.float32)
        temporal_feats[:, 0] = hour_of_day / 24.0
        temporal_feats[:, 1] = ((t // 96) % 7) / 7.0

        return torch.from_numpy(flows), torch.from_numpy(temporal_feats)

    def __len__(self) -> int:
        return self.num_timesteps - self.window_size - self.horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_flow = self.od_flows[idx : idx + self.window_size]  # (W, N, N)
        y_flow = self.od_flows[idx + self.window_size : idx + self.window_size + self.horizon]  # (H, N, N)
        x_temp = self.temporal_features[idx : idx + self.window_size]  # (W, 2)
        y_temp = self.temporal_features[idx + self.window_size : idx + self.window_size + self.horizon]  # (H, 2)
        return x_flow, y_flow, x_temp, y_temp
