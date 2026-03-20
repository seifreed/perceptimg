<p align="center">
  <img src="https://img.shields.io/badge/perceptimg-Image%20Optimization-blue?style=for-the-badge" alt="perceptimg">
</p>

<h1 align="center">perceptimg</h1>

<p align="center">
  <strong>Perceptual, policy-driven image optimization for modern workflows</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/perceptimg/"><img src="https://img.shields.io/pypi/v/perceptimg?style=flat-square&logo=pypi&logoColor=white" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/perceptimg/"><img src="https://img.shields.io/pypi/pyversions/perceptimg?style=flat-square&logo=python&logoColor=white" alt="Python Versions"></a>
  <a href="https://github.com/seifreed/perceptimg/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"></a>
  <a href="https://github.com/seifreed/perceptimg/actions"><img src="https://img.shields.io/github/actions/workflow/status/seifreed/perceptimg/ci.yml?style=flat-square&logo=github&label=CI" alt="CI Status"></a>
  <img src="https://img.shields.io/badge/coverage-100%25-brightgreen?style=flat-square" alt="Coverage">
</p>

<p align="center">
  <a href="https://github.com/seifreed/perceptimg/stargazers"><img src="https://img.shields.io/github/stars/seifreed/perceptimg?style=flat-square" alt="GitHub Stars"></a>
  <a href="https://github.com/seifreed/perceptimg/issues"><img src="https://img.shields.io/github/issues/seifreed/perceptimg?style=flat-square" alt="GitHub Issues"></a>
  <a href="https://buymeacoffee.com/seifreed"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-support-yellow?style=flat-square&logo=buy-me-a-coffee&logoColor=white" alt="Buy Me a Coffee"></a>
</p>

---

## Overview

**perceptimg** is a Python library for perceptual, policy-driven image optimization. Unlike traditional tools that optimize for file size or a single quality knob, perceptimg analyzes image content, interprets declarative policies, tests multiple strategies, and selects the best candidate based on perceptual quality metrics.

### Key Features

| Feature | Description |
|---------|-------------|
| **Policy-Driven** | Declarative constraints for size, quality, and use case |
| **Content-Aware** | Analyzes images for text, faces, and edge density |
| **Multi-Format** | JPEG, PNG, WebP, AVIF, JXL, HEIF, GIF, TIFF, APNG |
| **Perceptual Metrics** | SSIM, PSNR, and weighted perceptual scoring |
| **Explainable Results** | Full decision trail for every optimization |
| **Batch Processing** | Parallel processing with progress callbacks |
| **Enterprise Ready** | Checkpointing, retry, rate limiting, Prometheus metrics |
| **Clean Architecture** | Dependency inversion, modular engines, extensible |

### Supported Formats

```
Core Formats     JPEG, PNG, TIFF, GIF (always available via Pillow)
Modern Formats   WebP, AVIF, JPEG XL (JXL), HEIF/HEIC, APNG (codec-dependent)
```

---

## Installation

### From PyPI (Recommended)

```bash
pip install perceptimg
```

### From Source

```bash
git clone https://github.com/seifreed/perceptimg.git
cd perceptimg
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

---

## Quick Start

```python
from perceptimg import optimize, Policy

# Define policy constraints
policy = Policy(
    max_size_kb=150,
    min_ssim=0.97,
    preserve_text=True,
    target_use_case="web"
)

# Optimize image
result = optimize("input.png", policy)

# Access results
print(f"Format: {result.report.chosen_format}")
print(f"Size: {result.report.size_before_kb:.1f}KB → {result.report.size_after_kb:.1f}KB")
print(f"SSIM: {result.report.ssim:.4f}")
print(f"Reasons: {result.report.reasons}")
```

---

## Usage

### Command Line Interface

```bash
# Basic optimization
perceptimg optimize input.png --policy policy.json --out output.webp

# With inline constraints
perceptimg optimize input.png --max-size-kb 100 --min-ssim 0.95 --formats webp,avif

# JSON output
perceptimg optimize input.png --policy policy.json --log-json
```

### Available Options

| Option | Description |
|--------|-------------|
| `--policy` | Path to JSON policy file |
| `--out` | Output file path |
| `--max-size-kb` | Maximum output size in KB |
| `--min-ssim` | Minimum SSIM threshold (0.0-1.0) |
| `--formats` | Preferred formats (comma-separated) |
| `--preserve-text` | Prefer text clarity |
| `--preserve-faces` | Prefer face quality |
| `--log-json` | Structured JSON logging |
| `--log-level` | Log level (DEBUG, INFO, WARNING) |

---

## Policy Configuration

### Python API

```python
from perceptimg import Policy

policy = Policy(
    max_size_kb=120,
    min_ssim=0.98,
    preserve_text=True,
    preserve_faces=True,
    allow_lossy=True,
    preferred_formats=("webp", "avif", "jpeg"),
    target_use_case="mobile",
)
```

### JSON Configuration

```json
{
  "max_size_kb": 150,
  "min_ssim": 0.97,
  "preserve_text": true,
  "preserve_faces": false,
  "allow_lossy": true,
  "preferred_formats": ["webp", "avif", "jpeg"],
  "target_use_case": "web"
}
```

---

## Batch Processing

### Basic Batch

```python
from perceptimg import Policy, optimize_batch

policy = Policy(max_size_kb=100, min_ssim=0.9)
images = ["img1.png", "img2.png", "img3.png"]

result = optimize_batch(images, policy)
print(f"Success: {len(result.successful)}/{result.total}")
print(f"Success rate: {result.success_rate:.1%}")
```

### Async Processing

```python
import asyncio
from perceptimg import Policy, optimize_batch_async

async def main():
    policy = Policy(max_size_kb=100)
    result = await optimize_batch_async(images, policy)
    print(f"Processed {result.total} images")

asyncio.run(main())
```

### Memory-Efficient Streaming

```python
from perceptimg import Policy, optimize_lazy

policy = Policy(max_size_kb=100)

for path, result in optimize_lazy(large_image_list, policy):
    if isinstance(result, Exception):
        print(f"Error {path}: {result}")
    else:
        print(f"OK {path}: {result.report.size_after_kb:.1f}KB")
```

---

## Enterprise Features

### Checkpoint/Resume

```python
from perceptimg import Policy, optimize_batch_with_checkpoint

result = optimize_batch_with_checkpoint(
    images,
    policy,
    checkpoint_path="checkpoint.json",
    checkpoint_interval=10,  # Save every 10 images
)
# Resume after interruption by calling again with same checkpoint_path
```

### Retry with Exponential Backoff

```python
from perceptimg import Policy, optimize_batch_with_retry
from perceptimg.core.retry import RetryConfig

retry_config = RetryConfig(max_retries=3, base_delay_ms=100)

result = optimize_batch_with_retry(
    images,
    policy,
    retry_config=retry_config,
    continue_on_error=True,
)
```

### Rate Limiting

```python
from perceptimg import Policy, optimize_batch_with_rate_limit
from perceptimg.core.rate_limiter import RateLimitConfig

rate_limit = RateLimitConfig(requests_per_second=5)

result = optimize_batch_with_rate_limit(
    images,
    policy,
    rate_limit=rate_limit,
)
```

### Prometheus Metrics

```python
from perceptimg import Policy, optimize_batch_with_metrics
from perceptimg.core.metrics_exporter import MetricsCollector

metrics = MetricsCollector()
result, stats = optimize_batch_with_metrics(images, policy, metrics=metrics)

print(f"Average SSIM: {stats['average_ssim']:.2f}")
print(f"Compression ratio: {stats['average_compression_ratio']:.1%}")
```

---

## Architecture

```
perceptimg/
├── adapters/           # Framework adapters (PIL)
│   └── pil_adapter.py
├── core/               # Domain logic
│   ├── interfaces.py   # Abstractions (ImageAdapter Protocol)
│   ├── analyzer.py     # Content analysis
│   ├── metrics.py      # SSIM, PSNR, perceptual scoring
│   ├── optimizer.py    # Orchestration
│   ├── policy.py       # Declarative constraints
│   ├── strategy.py     # Candidate generation
│   └── batch/          # Batch processing
├── engines/            # Format-specific encoders
│   ├── webp_engine.py
│   ├── avif_engine.py
│   └── ...
└── utils/              # IO, heuristics, logging
```

### Clean Architecture Compliance

| Layer | Responsibility |
|-------|----------------|
| **Core** | Business logic, domain models, policies |
| **Adapters** | Framework-specific implementations |
| **Engines** | Format-specific encoding |
| **Utils** | Infrastructure services |

---

## Extensibility

### Custom Optimization Engine

```python
from perceptimg.engines.base import OptimizationEngine, EngineResult
from perceptimg.core.optimizer import Optimizer, register_engine

class MyCustomEngine(OptimizationEngine):
    format = "custom"
    priority = 100
    
    @property
    def is_available(self) -> bool:
        return True
    
    def optimize(self, image, strategy) -> EngineResult:
        # Custom optimization logic
        return EngineResult(data=optimized_bytes, format="custom", quality=90)

# Register custom engine
optimizer = Optimizer()
register_engine(optimizer, MyCustomEngine())
```

### Custom Analyzer

```python
from perceptimg.core.analyzer import Analyzer

class CustomAnalyzer(Analyzer):
    def analyze(self, image):
        result = super().analyze(image)
        # Add custom heuristics
        result.custom_score = self._compute_custom(image)
        return result
```

---

## API Reference

### OptimizationResult

| Field | Type | Description |
|-------|------|-------------|
| `image_bytes` | `bytes` | Optimized image data |
| `image` | `PIL.Image` | Pillow image object |
| `report` | `OptimizationReport` | Detailed results |

### OptimizationReport

| Field | Type | Description |
|-------|------|-------------|
| `chosen_format` | `str` | Selected format |
| `quality` | `int` | Quality setting |
| `size_before_kb` | `float` | Original size |
| `size_after_kb` | `float` | Optimized size |
| `ssim` | `float` | SSIM score |
| `psnr` | `float` | PSNR score |
| `perceptual_score` | `float` | Weighted quality score |
| `reasons` | `list[str]` | Decision trail |

---

## Requirements

- Python 3.10+
- Pillow (PIL fork)
- NumPy
- scikit-image
- See [pyproject.toml](pyproject.toml) for full dependencies

---

## Quality

This project enforces:

| Tool | Purpose |
|------|---------|
| `ruff` | Linting |
| `black` | Formatting |
| `mypy` | Type checking |
| `bandit` | Security analysis |
| `pytest` | Testing (145 tests) |

```bash
# Run all quality checks
ruff check perceptimg tests
black perceptimg tests --check
mypy perceptimg --ignore-missing-imports
bandit -r perceptimg -q --exclude perceptimg/tests
pytest tests/
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Support the Project

If you find perceptimg useful, consider supporting its development:

<a href="https://buymeacoffee.com/seifreed" target="_blank">
  <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50">
</a>

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Attribution Required:**
- Author: **Marc Rivero** | [@seifreed](https://github.com/seifreed)
- Repository: [github.com/seifreed/perceptimg](https://github.com/seifreed/perceptimg)

---

<p align="center">
  <sub>Made with dedication for the image optimization community</sub>
</p>