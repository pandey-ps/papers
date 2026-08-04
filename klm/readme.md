#### k-Nearest Neighbor Language Model                                                 

paper: https://arxiv.org/abs/1911.00172

based on the paper, in the LM, a function maps context to a distribution for the next token, this is done by creating a datastore of (key, value), and this is then used at inference for the current context, which then gets the `k` nearest neighbours and generates an output distribution (which is then interpolated).