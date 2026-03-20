"""Minimal usage example for perceptimg."""

from pathlib import Path

from PIL import Image, ImageDraw

from perceptimg import Policy, optimize
from perceptimg.utils import logging_config


def _ensure_sample(path: Path) -> Path:
    if path.exists():
        return path
    image = Image.new("RGB", (128, 96), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 20, 118, 60), outline="black", width=3)
    draw.text((16, 28), "perceptimg", fill="black")
    image.save(path)
    return path


if __name__ == "__main__":
    logging_config.configure_logging(json_output=False)
    sample_image = _ensure_sample(Path(__file__).parent / "sample.png")
    policy = Policy(max_size_kb=150, min_ssim=0.97, preserve_text=True, target_use_case="web")
    result = optimize(sample_image, policy)
    print(result.report)
