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


def tag_path(part0, part1, is_directory=False):
    """Provided a path to a file, this modifies the path by inserting a
    timestamp in front of the file format.
    Args:
        part0: path to the directory
        part1: filename (with extension)
        is_directory: whether the path is a directory
    Returns:
        new path with timestamp inserted
    """
    date_str = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    if not is_directory:
        file_parts = part1.split(".")
        new_filename = (
            "".join(file_parts[:-1]) + "_" + date_str + "." + file_parts[-1]
        )
    else:
        new_filename = part1 + "_" + date_str

    save_file = Path(part0) / new_filename
    return save_file


def secure_path(save_file, is_directory=False) -> str: 
    """Provided a path to a file, this checks whether such a file already
    exists. If not, it leaves the path unchanged. If there is already such a
    file in the specified path, this routine modifies the path by inserting a
    timestamp in front of the file format.

    Note: atm this works only for files with a single dot (E.g.: prefix.outfix
        works, but for pre.fix.outfix the tagged path will only consider the
        last dot)

    Examples:
        "/home/gerrit/kek.lel" -> "/home/gerrit/kek_2024-06-11_15_11_26.lel"

    Args:
        save_file: path to the file to be checked
        is_directory: whether the path is a directory
    Returns:
        new path with timestamp inserted if the file existed already
    """
    if is_directory:  # delete trailing slash
        save_file = str(Path(save_file))

    working_dir, filename = os.path.split(save_file)
    if os.path.exists(save_file):
        save_file = tag_path(working_dir, filename, is_directory)

    return save_file


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
