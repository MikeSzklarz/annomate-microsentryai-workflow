import numpy as np

from core.logic.center_template import extract_template, locate_center


def _pattern(width=20, height=16):
    y, x = np.mgrid[:height, :width]
    return np.dstack(
        [
            (x * 7 + y * 3) % 255,
            (x * 5 + y * 11) % 255,
            (x * 13 + y * 17) % 255,
        ]
    ).astype(np.uint8)


def test_extract_template_returns_expected_anchor():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    pattern = _pattern()
    image[32:48, 40:60] = pattern

    template, anchor_x, anchor_y = extract_template(image, 50, 40, 20, 16)

    assert anchor_x == 10
    assert anchor_y == 8
    assert template.shape == (16, 20, 3)
    np.testing.assert_array_equal(template, pattern)


def test_extract_template_clips_at_image_border():
    image = np.zeros((12, 14, 3), dtype=np.uint8)

    template, anchor_x, anchor_y = extract_template(image, 7, 6, 20, 20)

    assert template.shape == (12, 14, 3)
    assert anchor_x == 7
    assert anchor_y == 6


def test_locate_center_matches_expected_center():
    template = _pattern()
    image = np.zeros((180, 220, 3), dtype=np.uint8)
    image[122:138, 140:160] = template

    center_x, center_y, score = locate_center(image, template, 10, 8)

    assert center_x == 150
    assert center_y == 130
    assert score > 0.99


def test_locate_center_applies_anchor_offset():
    template = _pattern()
    image = np.zeros((90, 120, 3), dtype=np.uint8)
    image[40:56, 30:50] = template

    center_x, center_y, score = locate_center(image, template, 3, 5)

    assert center_x == 33
    assert center_y == 45
    assert score > 0.99


def test_locate_center_returns_score_for_noisy_image():
    rng = np.random.default_rng(42)
    template = _pattern()
    image = rng.integers(0, 255, size=(90, 120, 3), dtype=np.uint8)

    _center_x, _center_y, score = locate_center(image, template, 10, 8)

    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0
