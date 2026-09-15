"""Unit & Tensor Invariance Test Suite."""

import pytest
import torch
from src.data.graph_generator import SyntheticTransitODDataset
from src.model.embeddings import DecoupledSpatialEmbedding
from src.model.transformer import SpatiotemporalTransitTransformer


def test_dataset_output_shapes() -> None:
    dataset = SyntheticTransitODDataset(num_nodes=6, num_timesteps=100, window_size=6, horizon=2)
    x_flow, y_flow, x_temp, y_temp = dataset[0]

    assert x_flow.shape == (6, 6, 6)
    assert y_flow.shape == (2, 6, 6)
    assert x_temp.shape == (6, 2)
    assert y_temp.shape == (2, 2)
    assert (x_flow >= 0).all(), "Demand flows must be non-negative"


def test_decoupled_embeddings_asymmetry() -> None:
    num_nodes = 5
    embed_dim = 8
    embed_layer = DecoupledSpatialEmbedding(num_nodes=num_nodes, embed_dim=embed_dim)
    spatial_repr = embed_layer()  # (N, N, 2 * embed_dim)

    assert spatial_repr.shape == (num_nodes, num_nodes, 16)
    # Ensure origin and destination embeddings are not symmetric
    assert not torch.allclose(spatial_repr[0, 1], spatial_repr[1, 0])


def test_transformer_forward_shape_and_gradients() -> None:
    b, w, h, n = 2, 4, 2, 4
    model = SpatiotemporalTransitTransformer(
        num_nodes=n, window_size=w, horizon=h, spatial_dim=8, d_model=16, nhead=2, num_layers=1
    )

    dummy_flow = torch.rand(b, w, n, n)
    dummy_temp = torch.rand(b, w, 2)

    pred = model(dummy_flow, dummy_temp)
    assert pred.shape == (b, h, n, n)

    loss = pred.sum()
    loss.backward()
    assert model.input_projection.weight.grad is not None
