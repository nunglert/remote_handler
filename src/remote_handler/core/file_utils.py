import os
import shutil
from datetime import datetime
from pathlib import Path


def tag_path(part0, part1, is_directory=False):
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


def secure_path(save_file, is_directory=False):
    """Provided a path to a file, this checks whether such a file already
    exists. If not, it leaves the path unchanged. If there is already such a
    file in the specified path, this routine modifies the path by inserting a
    timestamp in front of the file format.

    Note: atm this works only for files with a single dot (E.g.: prefix.outfix
        works, but for pre.fix.outfix the tagged path will only consider the
        last dot)

    Examples:
        "/home/gerrit/kek.lel" -> "/home/gerrit/kek_2024-06-11_15_11_26.lel"
    """
    if is_directory:  # delete trailing slash
        save_file = str(Path(save_file))

    working_dir, filename = os.path.split(save_file)
    if os.path.exists(save_file):
        save_file = tag_path(working_dir, filename, is_directory)

    return save_file


def secure_declare_path(
    filename: str, move: bool = True, is_directory: bool = False
) -> str:
    """Checks whether for a given `filename` a file exists already on the
    disk and in that case renames the existing file by adding a time tag such
    that `filename` can be safely used without overwriting or appending an
    existing file. If `move` is set to False, the existing file is overwritten.
    """
    if os.path.exists(filename):
        if move:
            shutil.move(filename, secure_path(filename, is_directory))
        else:
            os.remove(filename)
    return filename
