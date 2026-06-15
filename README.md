# remote_handler

[![pipeline status](https://gitlab.tuwien.ac.at/e165-03-1_theoretische_materialchemie/madsen-s-research-group/remote_handler/badges/main/pipeline.svg)](https://gitlab.tuwien.ac.at/e165-03-1_theoretische_materialchemie/madsen-s-research-group/remote_handler/-/commits/main)
[![coverage report](https://gitlab.tuwien.ac.at/e165-03-1_theoretische_materialchemie/madsen-s-research-group/remote_handler/badges/main/coverage.svg)](https://gitlab.tuwien.ac.at/e165-03-1_theoretische_materialchemie/madsen-s-research-group/remote_handler/-/commits/main)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://remote-handler-ed37f2.git-pages.tuwien.ac.at)

A Python package for streamlining active learning workflows on remote clusters. Handles SSH/SFTP connections, SLURM job submissions, and orchestrates distributed computational workflows.

For more detailed documentation please refer to the [documentation](https://remote-handler-ed37f2.git-pages.tuwien.ac.at).


## Features

- **Remote file management**: Unified interface for local and remote filesystem operations via SSH/SFTP
- **SLURM integration**: Automated job submission and management using Jinja2 templates
- **Active learning workflows**: Orchestrate Sampler → Evaluator → Trainer loops across multiple machines
- **Task abstraction**: Flexible framework for defining and executing computational tasks

## Installation

```bash
# Create a conda environment (optional)
conda create --name remote_handler python=3.10
conda activate remote_handler

# Install the package
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with documentation dependencies
pip install -e ".[docs]"
```

## Quick Start

### SSH Configuration

Configure your remote host in `~/.ssh/config` with key-based authentication:

```bash
Host lugus
    User USERNAME
    HostName lugus.theochem.tuwien.ac.at
```

### SLURM Configuration

Create a cluster config file (e.g., `lugus.json`):

```json
{
    "remote_machine": "lugus",
    "partition": "normal",
    "savepath": "/home/username/remote_handler",
    "conda_env": "myenv",
    "template": "lugus.sh.j2"
}
```

Set environment variables:

```bash
export REMOTE_HANDLER_CONFIG_DIR=/path/to/config_files/
export REMOTE_HANDLER_SLURM_TEMPLATES=/path/to/custom_templates/
```

### Basic Usage

**Working with remote files:**

```python
from remote_handler.file_handler import RemoteFileSystem
from pathlib import Path

# Connect to remote filesystem
remote_fs = RemoteFileSystem(remote_alias="lugus")

# Create directory and transfer files
remote_fs.makedirs(Path("my_project"), exist_ok=True)
remote_fs.put("local_file.txt", "my_project/remote_file.txt")
```

**Active learning workflow:**

```python
from remote_handler.workflows import ActiveLearningHandler

# Define your Sampler, Evaluator, and Trainer classes
# (see documentation for full examples)

handler = ActiveLearningHandler(
    sampler=my_sampler,
    evaluator=my_evaluator,
    trainer=my_trainer,
    config=config
)

# Run the active learning loop
handler.run(n_iterations=10)
```

For detailed examples and API documentation, visit the [full documentation](https://remote-handler-ed37f2.git-pages.tuwien.ac.at).

## Development

### Testing

```bash
# Run all tests
pytest

# Run specific test with verbose output
pytest -k test_name -s --log-cli-level info
```

### Code Formatting

Install pre-commit hooks to maintain code quality:

```bash
pre-commit install
```

### Building Documentation

```bash
cd docs
make html
# Open docs/build/html/index.html in browser
```

## Project Structure

```
remote_handler/
├── src/remote_handler/
│   ├── file_handler.py       # File system abstractions
│   ├── ssh_handler.py         # SSH connection management
│   ├── slurm_handler.py       # SLURM job management
│   ├── tasks/                 # Task definitions
│   ├── workflows/             # Active learning workflows
│   └── templates/             # Jinja2 templates for SLURM scripts
├── test/                      # Test suite
├── examples/                  # Example workflows
└── docs/                      # Sphinx documentation
```

## Contributing

Contributions are welcome! Please:

1. Write tests for new functionality
2. Follow the existing code style
3. Ensure all tests pass before submitting PRs
4. Use pre-commit hooks for formatting

## License

See LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{remote_handler,
  author = {Unglert, Nico},
  title = {remote_handler: Streamlining Active Learning Workflows},
  year = {2026},
  url = {https://gitlab.tuwien.ac.at/e165-03-1_theoretische_materialchemie/madsen-s-research-group/remote_handler}
}
```
