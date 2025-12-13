# Utility functions
# Common helper functions used across the project.
import logging
import sys
from types import SimpleNamespace


def setup_logger(name=__name__):
    """Return a stdout logger configured with timestamped messages."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
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
