# remote_handler

[![pipeline status](https://gitlab.tuwien.ac.at/e165-03-1_theoretische_materialchemie/madsen-s-research-group/remote_handler/badges/main/pipeline.svg)](https://gitlab.tuwien.ac.at/e165-03-1_theoretische_materialchemie/madsen-s-research-group/remote_handler/-/commits/main)
[![coverage report](https://gitlab.tuwien.ac.at/e165-03-1_theoretische_materialchemie/madsen-s-research-group/remote_handler/badges/main/coverage.svg)](https://gitlab.tuwien.ac.at/e165-03-1_theoretische_materialchemie/madsen-s-research-group/remote_handler/-/commits/main)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://e165-03-1_theoretische_materialchemie.git-pages.tuwien.ac.at/madsen-s-research-group/remote_handler/)

A Python package for running **active-learning workflows for machine-learned
force fields** across remote HPC clusters. It abstracts SSH/SFTP file transfer
and job execution (SLURM, or scheduler-free), and orchestrates a
**Sampler → GroundTruthEvaluator → ModelTrainer** loop in which each actor can
run on a *different* machine.

📖 **[Full documentation](https://e165-03-1_theoretische_materialchemie.git-pages.tuwien.ac.at/madsen-s-research-group/remote_handler/)** —
the authoritative reference. This README is just the entry point.

---

## Installation

```bash
pip install -e .              # host-side orchestration (dependency-light)
pip install -e ".[dev,docs]"  # + test suite and docs tooling
```

The ML/scientific functionality (MACE, NeuralIL, …) lives behind
optional extras and is usually installed in the *remote* conda environment, not
on the host. See the
[Installation guide](https://e165-03-1_theoretische_materialchemie.git-pages.tuwien.ac.at/madsen-s-research-group/remote_handler/installation.html)
for the full list of extras and the host-vs-remote split.

## Getting started

Configure a cluster once with the bundled CLI, then refer to it by name; remote
hosts resolve through your `~/.ssh/config` (key-based auth, `ProxyJump`
supported):

```bash
remote_handler configure cluster lugus   # build & persist a cluster config
remote_handler check                     # validate it (add --connect to probe)
```

From there, see:

- **[Quickstart](https://e165-03-1_theoretische_materialchemie.git-pages.tuwien.ac.at/madsen-s-research-group/remote_handler/quickstart.html)** — a first end-to-end run.
- **[Examples](https://e165-03-1_theoretische_materialchemie.git-pages.tuwien.ac.at/madsen-s-research-group/remote_handler/examples.html)** — DFT labelling and MACE training.
- **[CLI](https://e165-03-1_theoretische_materialchemie.git-pages.tuwien.ac.at/madsen-s-research-group/remote_handler/cli.html)** — configure / check / show.

## How it fits together

Three layers: `core/` (transport + execution primitives), `workflows/` (the
active-learning actors), and `scripts/` (payloads shipped to and run on the
cluster). The
[Core Concepts](https://e165-03-1_theoretische_materialchemie.git-pages.tuwien.ac.at/madsen-s-research-group/remote_handler/concepts.html)
page explains the boundaries — host vs. remote, the task abstraction, the
pluggable executor, and the active-learning loop.

## Example projects using remote_handler

- [Replica-exchange nested sampling](https://gitlab.tuwien.ac.at/e165-03-1_theoretische_materialchemie/madsen-s-research-group/jaxrens_active_learning)
- [LAMMPS](https://gitlab.tuwien.ac.at/e165-03-1_theoretische_materialchemie/madsen-s-research-group/lammps_active_learning)
- Clinamen2 — _link TBD_
- SSPD — _link TBD_

## Contributing

Dev setup, test tiers, code style, and the release/versioning workflow are in
the
[Contributing guide](https://e165-03-1_theoretische_materialchemie.git-pages.tuwien.ac.at/madsen-s-research-group/remote_handler/contributing.html).
In short:

```bash
pip install -e ".[dev]" && pre-commit install
pytest                                   # runs with no configuration required
```

## License

See [LICENSE](LICENSE).

## Citation

If you use this package in your research, please cite:

```bibtex
@software{remote_handler,
  author = {Unglert, Nico},
  title  = {remote_handler: Streamlining Active Learning Workflows},
  year   = {2026},
  url    = {https://gitlab.tuwien.ac.at/e165-03-1_theoretische_materialchemie/madsen-s-research-group/remote_handler}
}
```
