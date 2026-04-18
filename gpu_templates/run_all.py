from __future__ import annotations

from typing import Callable, Dict, List

from gpu_templates.cuml_template import run_cuml_demo
from gpu_templates.cupy_template import run_cupy_demo
from gpu_templates.device_utils import system_info
from gpu_templates.pytorch_template import run_pytorch_demo
from gpu_templates.tensorflow_template import run_tensorflow_demo
from gpu_templates.transformers_template import run_transformers_demo

DemoFunc = Callable[[], Dict[str, object]]


def run_all() -> List[Dict[str, object]]:
    demos: List[DemoFunc] = [
        run_pytorch_demo,
        run_tensorflow_demo,
        run_transformers_demo,
        run_cupy_demo,
        run_cuml_demo,
    ]

    rows: List[Dict[str, object]] = []
    for demo in demos:
        try:
            rows.append(demo())
        except Exception as exc:
            rows.append(
                {
                    "framework": demo.__name__,
                    "available": False,
                    "device": "unknown",
                    "ok": False,
                    "message": f"Unhandled error: {exc}",
                }
            )
    return rows


def print_report() -> None:
    info = system_info()
    print("System Info")
    print("-----------")
    for key, value in info.items():
        print(f"{key}: {value}")
    print()

    print("GPU Framework Check")
    print("-------------------")
    for row in run_all():
        framework = str(row.get("framework", "unknown")).ljust(10)
        available = str(row.get("available", False)).ljust(5)
        device = str(row.get("device", "unknown")).ljust(6)
        ok = str(row.get("ok", False)).ljust(5)
        message = str(row.get("message", ""))
        print(
            f"{framework} | available={available} | device={device} | ok={ok} | {message}"
        )


if __name__ == "__main__":
    print_report()
