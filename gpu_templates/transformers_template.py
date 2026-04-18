from __future__ import annotations

import importlib
from typing import Any, Dict


def run_transformers_demo() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "framework": "transformers",
        "available": False,
        "device": "cpu",
        "ok": False,
        "message": "",
    }

    try:
        import torch
        transformers = importlib.import_module("transformers")
    except Exception as exc:
        result["message"] = f"Transformers not installed or failed to import: {exc}"
        return result

    result["available"] = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    try:
        # Build a tiny local BERT to avoid downloading model weights.
        config = transformers.BertConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            vocab_size=1000,
        )
        model = transformers.BertModel(config).to(device)

        input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        mean_value = float(outputs.last_hidden_state.mean().item())
    except Exception as exc:
        result["message"] = f"Transformers runtime error: {exc}"
        return result

    result["device"] = str(device)
    result["ok"] = True
    result["message"] = f"Tiny BERT forward pass successful, mean={mean_value:.6f}"
    return result


if __name__ == "__main__":
    print(run_transformers_demo())