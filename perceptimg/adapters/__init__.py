"""Image adapters for Clean Architecture.

This module provides implementations of the ImageAdapter Protocol
for different image frameworks. The core domain depends on abstractions
(interfaces.py) while concrete implementations live in this adapters layer.

Dependency Rule:
    core → interfaces (depends on abstractions)
    adapters → implements interfaces (depends on frameworks like PIL)
"""

from .pil_adapter import (
    PILImageAdapter,
    PILImageIO,
    bytes_to_image,
    image_to_bytes,
    load_image,
)

__all__ = [
    "PILImageAdapter",
    "PILImageIO",
    "load_image",
    "bytes_to_image",
    "image_to_bytes",
]
