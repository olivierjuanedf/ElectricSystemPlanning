import logging
import os
import shutil
from pathlib import Path
from typing import List, Union


def check_file_existence(file: str, file_descr: str = None):
    """
    Check existence of a given file before doing some operations on it (e.g., reading it)

    :param file: file whose existence must be checked
    :param file_descr: description of this file, used in error msg if missing
    """
    if not os.path.isfile(file):
        msg_prefix = 'File' if file_descr is None else f'{file_descr} file'
        raise Exception(f'{msg_prefix} {file} does not exist -> STOP')


def uniformize_path_os(path_str: str) -> str:
    return os.path.normpath(path_str)


def make_dir(full_path: str, with_warning: bool = False):
    if os.path.exists(full_path):
        if with_warning:
            logging.warning(f'Directory {full_path} already exists -> not created again')
    else:
        Path(full_path).mkdir(parents=True)


def delete_files(directory: str, str_in_file: str = None, suffix: str = None):
    path = Path(directory)
    str_search = f'*{suffix}' if suffix is not None else str_in_file
    for file in path.rglob(str_search):
        if file.is_file():
            logging.debug(f"Deleting: {file}")
            file.unlink()


def remove_folder(folder_path: Union[str, Path]):
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)
    shutil.rmtree(folder_path)


def empty_folder(folder_path: Union[str, Path]):
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)

    for item in folder_path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()  # suppr. file or link
        elif item.is_dir():
            shutil.rmtree(item)  # recursive suppr.


def find_project_root(start_path: Path) -> Path:
    for path in [start_path, *start_path.parents]:
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
    raise RuntimeError("Project root not found")


def get_files_from_prefix(folder: str, file_prefix: str, return_full_path: bool = False) -> List[str]:
    selec_files = [file for file in os.listdir(folder)
                   if os.path.isfile(os.path.join(folder, file)) and file.startswith(file_prefix)]
    if return_full_path:
        selec_files = [os.path.join(folder, file) for file in selec_files]
    return selec_files
