from __future__ import annotations

import platform
from typing import Dict


def system_info() -> Dict[str, str]:
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }
