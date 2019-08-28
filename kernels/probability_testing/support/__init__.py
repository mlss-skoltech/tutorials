# flake8: noqa
import sys
assert sys.version_info >= (3, 6)

from .kernels import LazyKernel
from .mmd import mmd2_u_stat_variance
from .utils import as_tensors, maybe_squeeze, pil_grid
