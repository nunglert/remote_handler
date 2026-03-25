import os
import shutil
from datetime import datetime
from pathlib import Path


def tag_path(part0: str, part1: str, is_directory: bool = False) -> Path:
    """Append a timestamp to a filename or directory name to create a unique path.

    Args:
        part0: Path to the directory
        part1: Filename (with extension) or directory name
        is_directory: Whether the path is a directory

    Returns:
        New path with timestamp inserted

    Example:
        >>> tag_path("/home/user", "data.csv", False)
        Path('/home/user/data_2024-01-15_14_30_45.csv')
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


def secure_path(save_file: str, is_directory: bool = False) -> str:
    """Check if a path exists and add timestamp if it does.

    If a file/directory already exists at the given path, this modifies the path
    by inserting a timestamp to avoid overwriting.

    Args:
        save_file: Path to the file or directory to check
        is_directory: Whether the path is a directory

    Returns:
        Original path if it doesn't exist, or path with timestamp if it does

    Note:
        Currently works only for files with a single dot (e.g., prefix.ext works,
        but for pre.fix.ext only the last dot is considered)

    Example:
        >>> secure_path("/home/user/data.csv")
        '/home/user/data_2024-01-15_14_30_45.csv'  # if data.csv exists
    """
    if is_directory:  # delete trailing slash
        save_file = str(Path(save_file))

    working_dir, filename = os.path.split(save_file)
    if os.path.exists(save_file):
        save_file = str(tag_path(working_dir, filename, is_directory))

    return save_file


def secure_declare_path(
    filename: str, move: bool = True, is_directory: bool = False
) -> str:
    """
    Checks whether for a given `filename` a file exists already on the
    disk and in that case renames the existing file by adding a time tag such
    that `filename` can be safely used without overwriting or appending an
    existing file. If `move` is set to False, the existing file is overwritten.
    Args:
        filename: path to the file to be checked
        move: whether to rename the existing file (True) or overwrite it (False)
        is_directory: whether the path is a directory
    Returns:
        the original filename (which is now safe to use)
    """
    if os.path.exists(filename):
        if move:
            shutil.move(filename, secure_path(filename, is_directory))
        else:
            os.remove(filename)
    return filename
