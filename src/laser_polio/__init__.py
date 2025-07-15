from .distributions import *  # noqa F403
from .pars import *  # noqa F403
from .plotting import *  # noqa F403
from .model import *  # noqa F403
from .run_sim import *  # noqa F403
from pathlib import Path
import os

# from .seir_mpm import *
from .utils import *  # noqa F403

__version__ = "0.2.9"

def find_project_root(marker: str = "pyproject.toml") -> Path:
    """
    Find the root directory of the project by walking up the directory tree from the current file.

    This function assumes that the presence of a specific "marker" file (e.g., pyproject.toml)
    denotes the root of the project. This is a common convention in Python projects using modern
    packaging tools like Poetry or setuptools.

    It works by starting at the location of the current file (__file__) and walking up the
    parent directories until it finds the marker file. If the marker is found, it returns that
    directory. If not, it defaults to returning the parent directory of the current file.

    When this package is installed in `site-packages`, the marker file will likely not exist,
    so the fallback behavior will return the path to the installed package itself, e.g.,
    `/path/to/site-packages/laser_polio`.

    It is possible to override this attempt at auto-discovery of the root directory with the
    environment variable `POLIO_ROOT`.

    Args:
        marker (str): The filename to search for (default is "pyproject.toml").

    Returns:
        Path: The directory where the marker file is found, or the parent directory of this file
              if the marker is not found.
    """

    env_root = os.getenv("POLIO_ROOT")
    if env_root:
        return Path(env_root)

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    # fall back to package root if marker not found
    return current.parent

root = find_project_root()
# TBD: convert to logging
print( f"laser_polio root = {root}" )
