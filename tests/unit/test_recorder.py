"""Headless recording and frame-composition tests."""

import imageio.v3 as iio
import numpy as np
import pytest

from die.agents.static import BrownianAgent
from die.sim import Simulation
from die.view.record import AnimationRecorder
from die.view.render import compose_grid, to_uint8_rgb


class TestFrameComposition:
    def test_to_uint8_rgb_from_float_rgb(self):
        img = to_uint8_rgb(np.full((4, 4, 3), 0.5))
        assert img.shape == (4, 4, 3)
        assert img.dtype == np.uint8
        assert img.max() == 127

    def test_to_uint8_rgb_composites_alpha(self):
        rgba = np.ones((4, 4, 4))
        rgba[..., 3] = 0.  # fully transparent -> black
        assert to_uint8_rgb(rgba).max() == 0

    def test_to_uint8_rgb_clips_out_of_range(self):
        img = to_uint8_rgb(np.full((2, 2, 3), 7.5))
        assert img.max() == 255

    def test_compose_grid_tiles_and_resizes(self):
        frames = [np.zeros((8, 8, 3)), np.ones((1, 1, 3)), np.zeros((8, 8, 4))]
        grid = compose_grid(frames, ncols=2)
        # 3 tiles pad to 4 in a 2x2 grid of 8x8 tiles
        assert grid.shape == (16, 16, 3)
        assert grid.dtype == np.uint8


class TestAnimationRecorder:
    def test_records_gif_headless(self, small_env, tmp_path):
        out = tmp_path / 'run.gif'
        recorder = AnimationRecorder(out, fps=5)
        sim = Simulation(small_env, BrownianAgent(), observers=[recorder], seed=42)
        sim.run(5)

        assert out.exists()
        frames = iio.imread(out, index=...)
        assert frames.shape[0] == 5

    def test_every_skips_frames(self, small_env, tmp_path):
        out = tmp_path / 'sparse.gif'
        recorder = AnimationRecorder(out, fps=5, every=2)
        sim = Simulation(small_env, BrownianAgent(), observers=[recorder], seed=42)
        sim.run(5)
        frames = iio.imread(out, index=...)
        assert frames.shape[0] == 3  # steps 0, 2, 4

    def test_rejects_unknown_format(self, tmp_path):
        with pytest.raises(ValueError, match='Unsupported'):
            AnimationRecorder(tmp_path / 'out.avi')
