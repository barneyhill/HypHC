# HyperTS
### Inference of a tree sequence from genetic variation data accelerated by hyperbolic geometry and GPUs.
## Summary 

This library enables the efficient approximate inference of [tskit](https://github.com/tskit-dev/tskit) tree sequences from genetic variation data. This is achieved by optimising embeddings of sequences in hyperbolic space with respect to a continuous relaxation of the Dasgupta's objective. These embeddings are then used to perform clustering to create a tskit tree sequence. This algorithm is massively parallelised on the GPU.

## Background

This library is forked from HazyResearch's HypHC implementation presented in the 2020 NeurIPS paper:
> **From Trees to Continuous Embeddings and Back: Hyperbolic Hierarchical Clustering**\
> Ines Chami, Albert Gu, Vaggos Chatziafratis and Christopher Ré\
> Stanford University\
> Paper: https://arxiv.org/abs/2010.00402 \
> Code: https://github.com/HazyResearch/HypHC

With further optimisations (direct embedding of sequences) implemented from the 2021 NeurIPS paper: 
> **Neural Distance Embeddings for Biological Sequences**\
> Gabriele Corso, Rex Ying, Michal Pándy, Petar Veličković, Jure Leskovec, Pietro Liò\
> MIT, Stanford University, University of Cambridge, DeepMind\
> Paper: https://arxiv.org/abs/2109.09740 \
> Code: https://github.com/gcorso/NeuroSEED
