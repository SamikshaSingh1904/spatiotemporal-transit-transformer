# Spatiotemporal Transit Demand Transformer

An end-to-end deep learning framework for forecasting origin-destination (OD) passenger route demand across asymmetric metropolitan transit networks.

## Technical Architecture

```
[ Historical Flow Tensors: (B, W, N, N) ]
│
▼
┌────────────────────────────────────────────────────────┐
│  Decoupled Spatial Embedding Engine                    │
│  - Disjoint tables: E_orig(u), E_dest(v)               │
│  - Eliminates directional gradient interference        │
└──────────────────────────┬────────────────────────────┘
▼
┌────────────────────────────────────────────────────────┐
│  Spatiotemporal Tokenization & Self-Attention          │
│  - Merged spatial representation + temporal indices    │
│  - Multi-Head GELU Attention Layers                    │
└──────────────────────────┬─────────────────────────────┘
▼
[ Forecast Horizon Prediction: (B, H, N, N) ]
```

## Mathematical Formulation
Directional trip matrix flows $A \to B \neq B \to A$ create gradient collision under symmetric graph convolution. We represent directed edges via disjoint embedding projections:

$$\mathbf{E}_{uv} = [\mathbf{e}_{\text{orig}}(u) \,\|\, \mathbf{e}_{\text{dest}}(v)]$$

$$\mathbf{z}_t^{(uv)} = \text{Linear}\left( [x_t^{(uv)} \,\|\, \mathbf{E}_{uv} \,\|\, \mathbf{\tau}_t] \right) + \mathbf{P}_t$$

$$\hat{\mathbf{Y}}_{(u,v)} = \text{ReLU}\left(\mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \cdot \text{Transformer}(\mathbf{Z}^{(uv)}))\right)$$

## Quickstart

```bash
# Clone & install dependencies
pip install -r requirements.txt
pip install -e .

# Run test suite
pytest tests/ -v

# Run comparative benchmark
python -m src.benchmark
```
