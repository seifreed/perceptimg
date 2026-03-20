from PIL import Image

from perceptimg.core.analyzer import Analyzer


def test_analyzer_detects_text_like_edges() -> None:
    image = Image.new("RGB", (64, 32), "white")
    for x in range(10, 54):
        for y in range(8, 24):
            image.putpixel((x, y), (0, 0, 0))
    result = Analyzer().analyze(image)
    assert result.probable_text is True
    assert result.edge_density > 0


def test_analyzer_detects_skin_tone_faces() -> None:
    image = Image.new("RGB", (32, 32), (210, 170, 140))
    result = Analyzer().analyze(image)
    assert result.probable_faces is True
