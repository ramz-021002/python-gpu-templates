from __future__ import annotations

from typing import Any, Dict


def run_pytorch_demo() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "framework": "pytorch",
        "available": False,
        "device": "cpu",
        "ok": False,
        "message": "",
    }

    try:
        import torch
    except Exception as exc:
        result["message"] = f"PyTorch not installed or failed to import: {exc}"
        return result

    result["available"] = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    x = torch.randn(1024, 1024, device=device)
    y = torch.randn(1024, 1024, device=device)
    z = x @ y

    result["device"] = str(device)
    result["ok"] = True
    result["message"] = f"Matmul successful, mean={z.mean().item():.6f}"
    return result


if __name__ == "__main__":
    print(run_pytorch_demo())
