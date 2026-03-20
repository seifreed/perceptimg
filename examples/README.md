# Ejemplos y policies

## Policies listas para usar
- `policies/web_high_fidelity.json`: web, alta fidelidad con texto/rostros, tamaño moderado.
- `policies/mobile_aggressive.json`: mobile, agresiva en tamaño con SSIM mínimo 0.94.
- `policies/print_preserve_detail.json`: print, sin pérdida y sin límite de tamaño.

## Usar la CLI
```bash
perceptimg optimize ../data/image.png --policy policies/web_high_fidelity.json --out ../data/image_optimized.webp --log-json
```

CLI shortcuts (sin archivo policy):
```bash
perceptimg optimize input.png --policy policies/web_high_fidelity.json \  # base
  --max-size-kb 150 --min-ssim 0.96 --formats webp,avif,jpeg --ssim-weight 0.8 --size-weight 0.2

# O sin policy file:
perceptimg optimize input.png --max-size-kb 150 --min-ssim 0.96 --formats webp,avif --preserve-text --target-use-case web
```

In-memory (sin escribir policy):
```python
from perceptimg import Policy, optimize_image
from PIL import Image

image = Image.open("input.png")
policy = Policy(max_size_kb=150, min_ssim=0.96).with_updates(preferred_formats=("webp","avif"))
result = optimize_image(image, policy)
print(result.report.to_dict())
```

## Registrar un engine propio (plugin)
```python
from perceptimg import optimize, Policy
from perceptimg.core.optimizer import Optimizer, register_engine
from perceptimg.engines.base import OptimizationEngine, EngineResult
from PIL import Image


class DummyEngine(OptimizationEngine):
    format = "dummy"

    def can_handle(self, fmt: str) -> bool:
        return fmt == self.format

    def optimize(self, image: Image.Image, strategy) -> EngineResult:
        # No-op: devuelve los mismos bytes (solo para demostrar plugin)
        data = image.tobytes()
        return EngineResult(data=data, format=self.format, quality=strategy.quality, metadata={})


optimizer = Optimizer()
register_engine(optimizer, DummyEngine())
policy = Policy(preferred_formats=("dummy",))
# optimizer.optimize(...) ahora puede usar DummyEngine
```
