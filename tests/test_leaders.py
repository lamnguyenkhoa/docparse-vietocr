import cv2
import numpy as np

from app.leaders import find_dot_leaders


def blank(w=1200, h=200):
    return np.full((h, w), 255, np.uint8)


def test_row_of_even_dots_is_one_leader():
    img = blank()
    for x in range(200, 800, 15):
        cv2.circle(img, (x, 100), 3, 0, -1)
    boxes = find_dot_leaders(img, [])
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert 190 <= x0 <= 200 and 785 <= x1 <= 800 and 94 <= y0 <= 100 <= y1 <= 106


def test_two_runs_split_by_a_gap():
    img = blank()
    for x in list(range(100, 400, 15)) + list(range(600, 900, 15)):
        cv2.circle(img, (x, 100), 3, 0, -1)
    assert len(find_dot_leaders(img, [])) == 2


def test_scattered_noise_is_not_a_leader():
    rng = np.random.default_rng(0)
    img = blank()
    for _ in range(60):
        cv2.circle(img, (int(rng.integers(20, 1180)), int(rng.integers(20, 180))), 3, 0, -1)
    assert find_dot_leaders(img, []) == []


def test_masked_dots_are_ignored():
    img = blank()
    for x in range(200, 800, 15):
        cv2.circle(img, (x, 100), 3, 0, -1)
    assert find_dot_leaders(img, [[150, 80, 850, 120]]) == []


def test_text_strokes_are_not_dots():
    img = blank()
    cv2.putText(img, "Ho va ten: Nguyen Van A", (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
    assert find_dot_leaders(img, []) == []
