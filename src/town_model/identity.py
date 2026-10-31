"""Generic tensor-state hashing used by persisted visual feature identities."""

from __future__ import annotations

import hashlib
import json
from torch import Tensor


def hash_tensor_state(state: dict[str, Tensor]) -> str:
    """Hash a named tensor mapping without interpreting its scientific meaning."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
