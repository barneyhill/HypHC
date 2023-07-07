import torch.nn as nn

from model.hyphc import HypHC, index_to_one_hot

class HypHCLinear(HypHC):
    """ Hyperbolic linear model for hierarchical clustering. """

    def __init__(self, n_nodes=1, rank=2, temperature=0.05, init_size=1e-3, max_scale=1. - 1e-3, alphabet_size=4,
                sequence_length=128, device='cpu'):
        super().__init__(n_nodes=n_nodes, rank=rank, temperature=temperature, init_size=init_size, max_scale=max_scale)
        self.alphabet_size = alphabet_size
        self.device = device
        self.linear = nn.Linear(sequence_length*alphabet_size, rank)

    def encode(self, triple_ids=None, sequences=None):
        print((*sequences.shape[:-2], -1))
        sequences = sequences.reshape((*sequences.shape[:-2], -1))
        e = self.linear(sequences)
        return e