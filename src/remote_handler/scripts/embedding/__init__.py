"""Amorphous matrix embedding -- remote payloads.

Cuts a region of interest (ROI) out of a large trajectory, optionally orients
it using an OVITO surface-mesh normal, and repairs/trims the matrix left over
at the cut. See :mod:`remote_handler.scripts.embedding.extract` for the
remote entry point and :class:`remote_handler.workflows.samplers.
AmorphousEmbeddingSampler` for the host-side actor.
"""
