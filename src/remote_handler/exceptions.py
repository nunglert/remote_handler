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


class ScriptInputError(RemoteHandlerError):
    """Script input validation failures.

    Examples:
        - Missing required input parameters
        - Invalid input file format
        - Input parameter validation errors
        - Malformed JSON input files
    """

    pass


class ScheduleKeyWarning(UserWarning):
    """A schedule key names nothing the actor can take as an attribute.

    The value is then only passed to the actor's ``update`` as a keyword
    argument -- correct for an actor that reads it there, silently ignored
    otherwise (e.g. a misspelt key). Turn it into an error with
    ``warnings.simplefilter("error", ScheduleKeyWarning)``.
    """

    pass
