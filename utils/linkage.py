"""Decoding utils."""

import time

import numpy as np
import torch
from tqdm import tqdm

from unionfind import unionfind
from utils.lca import hyp_lca

### Single linkage using naive union find

# @profile
def nn_merge_uf_fast_np(xs, S, partition_ratio=None, verbose=False):
    """ Uses Cython union find and numpy sorting

    partition_ratio: either None, or real number > 1
    similarities will be partitioned into buckets of geometrically increasing size
    """
    n = xs.shape[0]
    # Construct distance matrix (negative similarity; since numpy only has increasing sorting)
    xs0 = xs[None, :, :]
    xs1 = xs[:, None, :]
    dist_mat = -S(xs0, xs1)  # (n, n)
    i, j = np.meshgrid(np.arange(n, dtype=int), np.arange(n, dtype=int))

    # Keep only unique pairs (upper triangular indices)
    idx = np.tril_indices(n, -1)
    ij = np.stack([i[idx], j[idx]], axis=-1)
    dist_mat = dist_mat[idx]

    # Sort pairs
    if partition_ratio is None:
        idx = np.argsort(dist_mat, axis=0)
    else:
        k, ks = ij.shape[0], []
        while k > 0:
            k = int(k // partition_ratio)
            ks.append(k)
        ks = np.array(ks)[::-1]
        if verbose:
            print(ks)
        idx = np.argpartition(dist_mat, ks, axis=0)
    ij = ij[idx]

    # Union find merging
    uf = unionfind.UnionFind(n)
    uf.merge(ij)
    return uf.tree

# Naive version
def build_ts_from_embeddings(leaves_embeddings, samples):

    # Assuming `leaves_embeddings` is a 2D tensor of shape [num_leaves, embedding_dim]

    # Get the indices of the upper triangular part excluding the diagonal
    triu_indices = torch.triu_indices(leaves_embeddings.size(0), leaves_embeddings.size(0), offset=1)

    # Extract the embeddings corresponding to these indices
    embeddings1 = leaves_embeddings[triu_indices[0]]
    embeddings2 = leaves_embeddings[triu_indices[1]]

    # Compute dot products
    dot_products = (embeddings1 * embeddings2).sum(dim=1)

    # Use argsort to get the indices in ascending order of dot products
    tri_indices = torch.argsort(dot_products, descending=True)
    sorted_pairs = triu_indices.t()[tri_indices]

    for pair in sorted_pairs:
        pass
