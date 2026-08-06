import os
import sys
import sysconfig
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# DeepSpeed JIT-builds its C++/CUDA ops (CPUAdam for CPU offload, async_io for
# NVMe offload) via torch.utils.cpp_extension, which shells out to `ninja` on
# PATH. `ninja` ships as a wheel, so it lives in the environment's own bin/ —
# but running `.venv/bin/python -m pytest` without activating the venv leaves
# that directory off PATH, and the affected tests then *skip* instead of fail.
# sysconfig, not Path(sys.executable).parent: .venv/bin/python is a symlink to
# the system interpreter, so resolving it lands in /usr/bin. get_path("scripts")
# is also correct on Windows (Scripts\).
_SCRIPTS_DIR = sysconfig.get_path("scripts")
if _SCRIPTS_DIR and _SCRIPTS_DIR not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _SCRIPTS_DIR + os.pathsep + os.environ.get("PATH", "")
