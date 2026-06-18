"""Pytree mini-batching helpers needed by NeuralIL training.

Vendored from ``jaxnest.utils`` so that the remote-side NeuralIL training
payload no longer depends on the ``jaxnest`` package. Only the closure used by
training is copied: ``evaluate_batch_func_pytree_args_remainder`` and the
reshape / split helpers it relies on.

``from __future__ import annotations`` keeps the original type hints (which
referenced jaxnest's walker/array aliases) as strings, so they are never
evaluated and we do not need to vendor ``jaxnest.types``.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
from typing import Callable, Union

import jax
import jax.numpy as jnp
from jax import tree_util


def reshape_into_two_dims(
    arr: FlatArray, n_runs: int, leading_axes: int = 0
) -> GroupedArray:
    """
    Reshape an array of shape (N, ...) into (n_runs, N // n_runs, ...).
    Args:
        arr: Array to be reshaped.
        n_runs: Number of runs to reshape into.
        leading_axes: Number of leading axes that should not be affected by
            the reshaping.
    Returns:
        Reshaped array.
    """
    # Compute the new size of the second dimension, taking into account any
    # leading axes that should remain unchanged
    new_second_dim_size = arr.shape[0 + leading_axes] // n_runs
    return arr.reshape(
        arr.shape[:leading_axes]  # Keep leading axes unchanged
        + (n_runs, new_second_dim_size)  # New first two dimensions
        + arr.shape[1 + leading_axes :]  # Keep remaining dimensions unchanged
    )


@partial(jax.jit, static_argnums=(1, 2))
def reshape_walkers_to_batches(
    walkers_batch_pytree: FlatWalkers, n_runs: int, leading_axes: int = 0
) -> GroupedWalkers:
    """
    Reshape a pytree with (n_runs*n_per_run, ...) to (n_runs, n_per_run, ...),
    without changing the leading axes.
    Args:
        walkers_batch_pytree: Pytree containing the walkers in flat form.
        n_runs: Number of runs to reshape into.
        leading_axes: Number of leading axes that should not be affected by
            the reshaping.
    """
    return tree_util.tree_map(
        lambda x: reshape_into_two_dims(x, n_runs, leading_axes),
        walkers_batch_pytree,
    )


def decompose_pytree_in_groups(
    pytree: Union[OrderedDict, jnp.ndarray], n_batches: int, axis: int = 0
) -> list[Union[OrderedDict, jnp.ndarray]]:
    """
    Splits a single pytree into groups of size `n_batches`, along a specified
    axis.
    Args:
        pytree: Pytree to be split.
        n_batches: Number of groups the pytree is split into.
        axis: Axis along which the pytree is split.
    Returns:
        List of pytrees split into groups of size `n_batches`.
    """
    # First reshape the pytree into (n_batches, n_per_batch, ...)
    batched_pytree = reshape_walkers_to_batches(pytree, n_batches, axis)
    # Then flatten the pytree into leaves
    leaves, treedef = tree_util.tree_flatten(batched_pytree)

    # Now we slice the leaves into groups of size `n_batches`
    # and reconstruct the pytrees in a new list
    new_pytrees = []
    for i in range(n_batches):
        sliced_leaves = [jnp.take(leaf, i, axis=axis) for leaf in leaves]
        new_pytree = tree_util.tree_unflatten(treedef, sliced_leaves)
        new_pytrees.append(new_pytree)

    return new_pytrees


@partial(jax.jit, static_argnums=1)
def concatenate_results(
    results: list[list[Union[OrderedDict, jnp.ndarray]]], axis: int = 0
) -> list[Union[OrderedDict, jnp.ndarray]]:
    batched_results = tree_util.tree_map(
        lambda *arrs: jnp.concatenate([*arrs], axis=axis), *results
    )
    return batched_results


@partial(jax.jit, static_argnums=(1, 2))
def get_batch_args(
    batch_args: list[Union[OrderedDict, jnp.ndarray]],
    n_batch: int,
    axis: int = 0,
) -> tuple[
    list[Union[OrderedDict, jnp.ndarray]],
    list[Union[OrderedDict, jnp.ndarray]],
    int,
]:
    """
    This function can split arbitrary pytrees into groups of size `n_batch`.
    The only condition is, that they have the same batch dimension. The batch
    dimension is not required to be divisible by `n_batch`.
    Args:
        batch_args: arbitrary number of arguments for func.
        n_batch: Dimension of the minibatches `batch_args` are split into.
        axis: Axis along which the pytrees are split.
    Returns:
        grouped_args: List of pytrees split into groups of size `n_batch`.
        remainder_args: List of pytrees containing the remaining entries that
            couldn't be grouped.
        remainder: Number of remaining entries that couldn't be grouped.
    """
    # Get the batch dimension from the first argument by looking at the first
    # leaf
    batch_dim = tree_util.tree_leaves(batch_args[0])[0].shape[axis]

    # If the batch dimension is smaller than n_batch, we can't split it
    if batch_dim < n_batch:
        remainder = 0
        n_groups = 1
    else:
        remainder = batch_dim % n_batch
        n_groups = batch_dim // n_batch

    # Now we split each argument into groups of size n_batch and the remaining
    # entries
    grouped_args = []
    remainder_args = []
    # Iterate through the elements of the list of pytrees
    for arg in batch_args:
        # Get the indices for the grouped and remaining entries
        cut_inds = jnp.arange(0, batch_dim - remainder)
        remainder_inds = jnp.arange(batch_dim - remainder, batch_dim)

        # Cut the argument to only include the grouped entries along the
        # specified axis. The result is a FlatWalkers pytree with shape
        # ((batch_dim - remainder)*n_walker_per_run, ...)
        cut_arg = tree_util.tree_map(
            lambda x: jnp.take(x, cut_inds, axis=axis), arg
        )
        # Decompose the cut argument into groups of size n_batch
        grouped_args.append(
            decompose_pytree_in_groups(cut_arg, n_groups, axis)
        )
        # Get the remaining entries along the specified axis
        remainder_args.append(
            tree_util.tree_map(
                lambda x: jnp.take(x, remainder_inds, axis=axis), arg
            )
        )

    return grouped_args, remainder_args, remainder


def evaluate_batch_func_pytree_args_remainder(
    n_batch: int,
    func: Callable,
    *batch_args: list[Union[OrderedDict, jnp.ndarray]],
    axis: int = 0,
    out_axis: int = 0,
) -> list[Union[OrderedDict, jnp.ndarray]]:
    """
    This function can batch evaluate a function `func` by passing
    arbitrary pytrees as `batch_args`. The only condition is, that they have
    the same batch dimension. The batch dimension is not required to be
    divisible by `n_batch`.

    Args:
        n_batch: Dimension of the minibatches `batch_args` are split into.
        func: Function that can process one minibatch of `batch_args`. Usually,
            this should be a jax.jit(jax.vmap(...)) transformed function.
        *batch_args: arbitrary number of arguments for func.
    Returns:
        List of combined input arrays, concatenated along the specified output
        axis.
    """

    # First, we split the arguments into groups of size `n_batch` and the
    # remaining entries
    grouped_args, remainder_args, remainder = get_batch_args(
        batch_args, n_batch, axis=axis
    )

    # Here, we iterate through the batches of size `n_batch`
    results = []
    for args in zip(*grouped_args):
        # Function is applied to one group of size `n_batch` simultaneously
        results.append(func(*args))

    # Here, the remaining arguments are passed
    if remainder != 0:
        results.append(func(*remainder_args))

    # Finally, we concatenate the results along the specified output axis
    batched_results = concatenate_results(results, axis=out_axis)

    return batched_results
