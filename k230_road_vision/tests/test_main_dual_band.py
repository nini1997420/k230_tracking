# -*- coding: utf-8 -*-
"""PC checks for the deployed single-file dual-band/RGB logic."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main


class RGBFrame:
    def __init__(self, width=320, height=240, fill=(220, 220, 220)):
        self.shape = (height, width, 3)
        self.pixels = [[list(fill) for _ in range(width)] for _ in range(height)]

    def __getitem__(self, key):
        y, x, c = key
        return self.pixels[y][x][c]

    def paint(self, y, x0, x1, color):
        for x in range(max(0, x0), min(self.shape[1], x1)):
            self.pixels[y][x] = list(color)


class ChannelView:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class SliceRGBFrame(RGBFrame):
    def __init__(self):
        super().__init__()
        self.scalar_reads = 0

    def __getitem__(self, key):
        y, x, c = key
        if isinstance(x, slice) and isinstance(c, int):
            start = 0 if x.start is None else x.start
            stop = self.shape[1] if x.stop is None else x.stop
            return ChannelView([self.pixels[y][i][c] for i in range(start, stop)])
        self.scalar_reads += 1
        return super().__getitem__(key)


def make_frame(with_crossbar=False, frame_class=RGBFrame):
    frame = frame_class()
    extractor = main.RoadBoundaryExtractor()
    for y in extractor.sample_y:
        ratio = float(y - 68) / 108.0
        left_x = int(105 - 55 * ratio)
        right_x = int(215 + 55 * ratio)
        frame.paint(y, left_x - 4, left_x + 5, (35, 35, 35))
        frame.paint(y, right_x - 4, right_x + 5, (35, 35, 35))
        frame.paint(y, 154, 167, (180, 25, 25))

    if with_crossbar:
        for y in (76, 84, 92):
            frame.paint(y, 30, 290, (30, 30, 30))
    return frame


def test_band_positions():
    extractor = main.RoadBoundaryExtractor()
    assert min(extractor.control_sample_y) >= main.CONTROL_BAND_TOP
    assert max(extractor.control_sample_y) < main.CONTROL_BAND_BOTTOM
    assert min(extractor.lookahead_sample_y) >= main.LOOKAHEAD_BAND_TOP
    assert max(extractor.lookahead_sample_y) < main.LOOKAHEAD_BAND_BOTTOM
    assert max(extractor.sample_y) < 180


def test_red_centre_tape_is_rejected():
    extractor = main.RoadBoundaryExtractor()
    boundary = extractor.extract(make_frame())
    assert boundary.left_valid and boundary.right_valid
    assert all(abs(x - 160) > 20
               for x, _ in boundary.left_points + boundary.right_points)


def test_wide_crossbar_confirms_intersection():
    extractor = main.RoadBoundaryExtractor()
    boundary = extractor.extract(make_frame(with_crossbar=True))
    geometry = main.RoadGeometry().compute(boundary, now_ms=0)
    detector = main.RoadStructureDetector()
    result = None
    for _ in range(main.JUNCTION_CONFIRM_FRAMES):
        result = detector.detect(boundary, geometry)
    assert result.intersection_candidate
    assert result.structure_confirmed


def test_rgb_rows_use_slice_fast_path():
    frame = make_frame(frame_class=SliceRGBFrame)
    boundary = main.RoadBoundaryExtractor().extract(frame)
    assert boundary.left_valid and boundary.right_valid
    assert frame.scalar_reads == 0


def test_full_width_edge_boundaries_are_detected():
    frame = SliceRGBFrame()
    extractor = main.RoadBoundaryExtractor()
    for y in extractor.sample_y:
        frame.paint(y, 0, 9, (30, 30, 30))
        frame.paint(y, 311, 320, (30, 30, 30))
        frame.paint(y, 154, 167, (180, 25, 25))

    boundary = extractor.extract(frame)
    assert boundary.left_valid and boundary.right_valid
    assert min(x for x, _ in boundary.left_points) <= 4
    assert max(x for x, _ in boundary.right_points) >= 315


if __name__ == "__main__":
    tests = [
        test_band_positions,
        test_red_centre_tape_is_rejected,
        test_wide_crossbar_confirms_intersection,
        test_rgb_rows_use_slice_fast_path,
        test_full_width_edge_boundaries_are_detected,
    ]
    for test in tests:
        test()
        print("PASS:", test.__name__)
    print("Dual-band deployed-main checks passed.")
