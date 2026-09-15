"""Decoupled Asymmetric Spatial Embedding Layer."""

import torch
import torch.nn as nn


class DecoupledSpatialEmbedding(nn.Module):
    """Maintains independent origin and destination embedding tables.
    
    Prevents directional gradient interference in asymmetric transit networks.
    Origin node u and destination node v project into separate metric sub-spaces.
    """

    def __init__(self, num_nodes: int, embed_dim: int) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim

        self.origin_embeddings = nn.Embedding(num_nodes, embed_dim)
        self.dest_embeddings = nn.Embedding(num_nodes, embed_dim)
        self.bilinear_bias = nn.Parameter(torch.zeros(num_nodes, num_nodes))

        # Initialization
        nn.init.xavier_uniform_(self.origin_embeddings.weight)
        nn.init.xavier_uniform_(self.dest_embeddings.weight)

    def forward(self) -> torch.Tensor:
        """Computes asymmetric static spatial context matrix.
        
        Returns:
            Tensor of shape (num_nodes, num_nodes, 2 * embed_dim) representing 
            directed edge representations (u -> v).
        """
        src_idx = torch.arange(self.num_nodes, device=self.origin_embeddings.weight.device)
        dst_idx = torch.arange(self.num_nodes, device=self.dest_embeddings.weight.device)

        e_orig = self.origin_embeddings(src_idx)  # (N, D)
        e_dest = self.dest_embeddings(dst_idx)    # (N, D)

        # Broadcast into directed pair representations (N, N, 2*D)
        e_orig_expanded = e_orig.unsqueeze(1).expand(-1, self.num_nodes, -1)
        e_dest_expanded = e_dest.unsqueeze(0).expand(self.num_nodes, -1, -1)

        edge_representation = torch.cat([e_orig_expanded, e_dest_expanded], dim=-1)
        return edge_representation
