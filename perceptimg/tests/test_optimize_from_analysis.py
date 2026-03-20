from PIL import Image

from perceptimg import Policy
from perceptimg.core.optimizer import Optimizer


def test_optimize_from_analysis_uses_provided_analysis() -> None:
    image = Image.new("RGB", (16, 16), "purple")
    policy = Policy(max_size_kb=50)
    opt = Optimizer()
    analysis = opt.analyzer.analyze(image)

    result = opt.optimize_from_analysis(image, analysis, policy)
    assert result.report.chosen_format
    assert result.report.size_after_kb <= 50 or policy.max_size_kb is None
