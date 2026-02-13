import os
import re
from datetime import datetime
from pathlib import Path

from natsort import natsorted


def grep(path: str, expression: str, with_path: bool = True):
    """Function to extract information from the input file.

    Args:
        path: Path which is crawled for an expression.
        expression: Regular expression pattern.
        with_path: Controls whether full paths to the files are returned or
            only filenames.
    Returns: List of files matching the expression.
    Example:
        >>> grep("/home/user/dir", ".*\.out", with_path=True)
        ['/home/user/dir/file1.out', '/home/user/dir/file2.out']
    """
    results = []
    for file in os.listdir(path):
        if re.search(expression, file):
            if with_path:
                results.append(os.path.join(path, file))
            else:
                results.append(file)
    return natsorted(results)


# tag_path and secure_path have been moved to file_utils.py to avoid duplication
# Import them from there if needed:
# from remote_handler.core.file_utils import tag_path, secure_path


class SimpleLogger:
    """A simple logger that can print messages to the console or write them to a file.
    If a log file is specified, messages will be appended to that file.
    Otherwise, messages will be printed to the console.

    Args:
        log_file: Optional path to a log file. If None, messages are printed to the
            console.
    Example:
        logger = SimpleLogger(log_file="my_log.txt")
        logger.print("This is a log message.")
    """

    def __init__(self, log_file=None):
        self.log_file = log_file

    def print(self, *args):
        # Join all arguments into a single string with spaces, like `print`
        message = " ".join(map(str, args))

        if self.log_file is not None:
            # Write the message to the specified file
            with open(self.log_file, "a") as f:
                f.write(message + "\n")
        else:
            print(message)
