"""Custom exception hierarchy for remote_handler.

A simple, pragmatic exception hierarchy providing better error context
than generic Python exceptions.
"""


class RemoteHandlerError(Exception):
    """Base exception for all remote_handler errors."""

    pass


class ConfigurationError(RemoteHandlerError):
    """Configuration file or environment variable issues.

    Examples:
        - Missing config files
        - Invalid config values
        - Missing environment variables
    """

    pass


class RemoteConnectionError(RemoteHandlerError):
    """SSH/SFTP connection failures.

    Examples:
        - Cannot connect to remote host
        - Connection lost during operation
        - SSH config issues
    """

    pass


class FileSystemError(RemoteHandlerError):
    """File system operation failures (local or remote).

    Examples:
        - Path not found
        - Permission denied
        - File transfer failures
    """

    pass


class TaskError(RemoteHandlerError):
    """Task/job execution failures.

    Examples:
        - SLURM job submission failed
        - Task validation errors
        - Job not finished or failed
    """

    pass


class WorkflowError(RemoteHandlerError):
    """Active learning workflow failures.

    Examples:
        - Sampling/evaluation/training step failed
        - Invalid workflow state
        - Max iterations reached
    """

    pass


class DependencyError(RemoteHandlerError):
    """Missing or incompatible dependencies.

    Examples:
        - Required package not installed
        - Version incompatibility
    """

    pass
