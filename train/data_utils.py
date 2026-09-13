"""Shared helpers for reading raw OCR sample schemas.

Stdlib-only, so CPU-side tooling can import it without pulling in torch.
"""

from typing import Any

# Image path keys, in lookup priority order.
IMAGE_PATH_KEYS = ("img_path", "image_path")


def get_img_path(sample: dict[str, Any], keys: tuple[str, ...] = IMAGE_PATH_KEYS) -> str | None:
    """Return the sample's image path, or None. A key may hold one path or a list of paths."""
    for key in keys:
        value = sample.get(key)
        if isinstance(value, (list, tuple)):
            value = next((item for item in value if item), None)
        if value:
            return str(value)
    return None
