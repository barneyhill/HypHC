"""Hierarchical clustering dataset."""

import logging

import numpy as np
import torch
import torch.utils.data as data

from datasets.triples import generate_all_triples, samples_triples


class HCDataset(data.Dataset):
    """Hierarchical clustering dataset."""

    def __init__(self, features, labels, similarities, num_samples):
        """Creates Hierarchical Clustering dataset with triples.

        @param labels: ground truth labels
        @type labels: np.array of shape (n_datapoints,)
        @param similarities: pairwise similarities between datapoints
        @type similarities: np.array of shape (n_datapoints, n_datapoints)
        """
        self.features = features
        self.labels = labels
        self.similarities = torch.from_numpy(similarities)
        self.n_nodes = self.similarities.shape[0]
        self.triples = torch.from_numpy(self.generate_triples(num_samples).astype("int64"))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple = self.triples[idx]
        similarities = torch.tensor([self.similarities[triple[0], triple[1]], 
                                     self.similarities[triple[0], triple[2]], 
                                     self.similarities[triple[1], triple[2]]])
        return triple, similarities

    def generate_triples(self, num_samples):
        logging.info("Generating triples.")
        if num_samples < 0:
            triples = generate_all_triples(self.n_nodes)
        else:
            triples = samples_triples(self.n_nodes, num_samples=num_samples)
        logging.info(f"Total of {triples.shape[0]} triples")
        return triples.astype("int64")
