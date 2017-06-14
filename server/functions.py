import os


def get_abs_path(*args):
    """ Concatenates path's parts sent through args and returns the
        absolute path to the file
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *args))
