import logging
import sys
from pathlib import Path
from types import SimpleNamespace



def setup_logger(name=__name__, log_file="log/run.log"):
    """Return a logger that writes to stdout and to log/run.log."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    # stdout
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # file
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_config(module):
    """
    Convert a config module with uppercase constants into a simple namespace.

    Parameters
    ----------
    module: Python module
        A module object (e.g., imported ``config``) containing configuration
        constants in uppercase. This helper mirrors them into an object for
        convenient attribute access and validation.
    """

    settings = {
        key: getattr(module, key)
        for key in dir(module)
        if key.isupper() and not key.startswith("__")
    }
    return SimpleNamespace(**settings)
