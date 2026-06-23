"""Cross-version shims for torch_sim API renames.

torch_sim 0.6 split the single Langevin-NPT integrator ``npt_langevin`` into
isotropic / anisotropic variants (and renamed the matching ``*_init`` helper
``npt_langevin_init`` -> ``npt_langevin_isotropic_init``). Our pipelines used
the pre-0.6 isotropic-by-default behaviour, so map onto the isotropic variant
while still supporting the older flat name (the floor is ``>=0.5.1``).

The newer ``_init`` / ``integrate`` accept ``**kwargs``, so extra arguments we
pass (e.g. ``seed``, ``external_pressure``) are absorbed where no longer used.
"""

import torch_sim as ts


def npt_langevin_integrator():
    """``Integrator`` enum member for isotropic Langevin NPT, any version."""
    return (
        getattr(ts.Integrator, "npt_langevin", None)
        or ts.Integrator.npt_langevin_isotropic
    )


def npt_langevin_init(**kwargs):
    """Initialise an isotropic Langevin-NPT state, any torch_sim version."""
    init = getattr(ts, "npt_langevin_init", None) or getattr(
        ts, "npt_langevin_isotropic_init"
    )
    return init(**kwargs)
