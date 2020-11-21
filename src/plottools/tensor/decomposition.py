from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np


def visualise_evolving_factor(evolving_factor, figsize=None, cmap='viridis', as_column=True, permutation=None):
    evolving_factor = np.array(evolving_factor)
    rank = evolving_factor.shape[-1]
    if permutation is None:
        permutation = tuple(range(rank))

    if as_column:
        fig, axes = plt.subplots(rank, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, rank, figsize=figsize)

    for r, ax in enumerate(axes):
        ax.imshow(evolving_factor[..., permutation[r]], cmap=cmap)
    
    return fig, axes


def find_permutation(evolving_factor_1, evolving_factor_2):
    evolving_factor_1 = np.asarray(evolving_factor_1)
    evolving_factor_2 = np.asarray(evolving_factor_2)
    rank = evolving_factor_1.shape[-1]
    ev1, ev2 = evolving_factor_1.reshape(-1, rank), evolving_factor_2.reshape(-1, rank)
    ev1 = ev1 / np.linalg.norm(ev1, axis=0, keepdims=True)
    ev2 = ev2 / np.linalg.norm(ev2, axis=0, keepdims=True)
    fms = ev1.T@ev2
    
    best_prod = 0
    for permutation in permutations(range(rank)):
        prod = 1
        for i, p in enumerate(permutation):
            prod *= abs(fms[i, p])
        
        if prod > best_prod:
            best_prod = prod
            best_perm = permutation
    
    assert best_prod != 0

    return best_perm


def compare_evolving_factors(
    evolving_factor_1,
    evolving_factor_2,
    figsize=None,
    cmap='viridis',
    column_per_decomposition=True,
    permutation='auto',
    flip_sign=True,
):
    evolving_factor_1 = np.array(evolving_factor_1)
    evolving_factor_2 = np.array(evolving_factor_2)

    rank = evolving_factor_1.shape[-1]

    if column_per_decomposition:
        fig, axes = plt.subplots(rank, 2, figsize=figsize)
    else:
        fig, axes = plt.subplots(2, rank, figsize=figsize)
        axes = axes.T
    
    if permutation == 'auto':
        permutation = find_permutation(evolving_factor_1, evolving_factor_2)
    for r, row in enumerate(axes):
        sign = 1
        if flip_sign:
            fv1 = evolving_factor_1[..., r].ravel()
            fv2 = evolving_factor_2[..., permutation[r]].ravel()

            sign = np.sign(fv1@fv2)

    
        row[0].imshow(evolving_factor_1[..., r], cmap=cmap)        
        row[1].imshow(sign*evolving_factor_2[..., permutation[r]], cmap=cmap)
        
    return fig, axes