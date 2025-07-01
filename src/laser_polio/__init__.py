from .distributions import *  # noqa F403
from .pars import *  # noqa F403
from .plotting import *  # noqa F403
from .model import *  # noqa F403
from .run_sim import *  # noqa F403
from pathlib import Path

# from .seir_mpm import *
from .utils import *  # noqa F403

__version__ = "0.2.7"

root = Path(__file__).resolve().parents[2]
