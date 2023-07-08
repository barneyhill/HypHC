"""Hierarchical clustering dataset."""

import logging

import numpy as np
import torch
import torch.utils.data as data

from datasets.triples import generate_all_triples, samples_triples


class HCDataset(data.Dataset):
    """Hierarchical clustering dataset."""

    def __init__(self, genotypes, samples, n_triplets):
        """Creates Hierarchical Clustering dataset with triples.

        @param labels: ground truth labels
        @type labels: np.array of shape (n_datapoints,)
        """
        self.genotypes = genotypes
        self.samples = samples
        self.n_samples, self.seq_length = genotypes.shape
        self.triplets, self.similarities = self.create_triplets(n_triplets)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triple = self.triplets[idx]
        similarities = self.similarities[idx]

        return triple, similarities

    def create_triplets(self, n_triplets):
        # Randomly generate triplets
        triplets = torch.randint(len(self.samples), (n_triplets, 3))

        # Extract all d1, d2 and d3
        d1 = self.genotypes[triplets[:, 0]]
        d2 = self.genotypes[triplets[:, 1]]
        d3 = self.genotypes[triplets[:, 2]]

        # Calculate the similarity for each pair
        similarities = torch.zeros((n_triplets, 3))
        similarities[:, 0] = torch.sum((d1 - d2)**2, dim=1) / self.seq_length
        similarities[:, 1] = torch.sum((d1 - d3)**2, dim=1) / self.seq_length
        similarities[:, 2] = torch.sum((d2 - d3)**2, dim=1) / self.seq_length

        return triplets, similarities
