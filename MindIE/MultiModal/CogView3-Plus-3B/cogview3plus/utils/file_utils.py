import os
MAX_PATH_LENGTH = 4096


def standardize_path(path: str, max_path_length=MAX_PATH_LENGTH, check_link=True):
    """
    Check and standardize path.
    Args:
        path (str): 未标准化路径
        max_path_length (int): 文件路径最大长度
        check_link (bool): 是否校验软链接
    Return: 
        path (str): 标准化后的绝对路径
    """
    check_path_is_none(path)
    check_path_length_lt(path, max_path_length)
    if check_link:
        check_path_is_link(path)
    path = os.path.realpath(path)
    return path


def is_path_exists(path: str):
    return os.path.exists(path)


def check_path_is_none(path: str):
    if path is None:
        raise ValueError("The path should not be None.")


def check_path_is_link(path: str):
    if os.path.islink(os.path.normpath(path)):
        print("warning", f"The path:{path} is a symbolic link file.")


def check_path_length_lt(path: str, max_path_length=MAX_PATH_LENGTH):
    if path.__len__() > max_path_length:
        raise ValueError(f"The length of path is {path.__len__()}, which exceeds the limit {max_path_length}.")
