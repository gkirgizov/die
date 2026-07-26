"""Headless animation recording (GIF/MP4) via imageio -- no matplotlib involved."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from die.sim.observers import SimObserver
from die.view.render import compose_grid, to_uint8_rgb

if TYPE_CHECKING:
    from die.sim.simulation import RunResult, Simulation, StepEvent


class AnimationRecorder(SimObserver):
    """Records simulation frames to an animation file (.gif or .mp4).

    Works fully headless: frames come from ``sim.render_frames()`` as numpy
    arrays and are written with imageio. MP4 output requires the ``video``
    extra (imageio-ffmpeg).
    """

    def __init__(self,
                 out_path: str | Path,
                 fps: int = 10,
                 every: int = 1,
                 mode: Literal['grid', 'medium'] = 'grid'):
        self._out_path = Path(out_path)
        self._fps = fps
        self._every = every
        self._mode = mode
        self._frames: list[np.ndarray] = []
        self._written = False

        suffix = self._out_path.suffix.lower()
        if suffix not in ('.gif', '.mp4'):
            raise ValueError(f'Unsupported animation format: {suffix!r} '
                             f'(expected .gif or .mp4)')
        if suffix == '.mp4':
            try:
                import imageio_ffmpeg  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "MP4 output requires the 'video' extra: "
                    "uv add 'die[video]' / pip install 'die[video]'") from exc

    def on_run_start(self, sim: Simulation, num_steps: int) -> None:
        self._frames = []
        self._written = False

    def on_step(self, sim: Simulation, event: StepEvent) -> None:
        if event.step % self._every != 0:
            return
        frames = sim.render_frames()
        image = compose_grid(frames) if self._mode == 'grid' else to_uint8_rgb(frames[0])
        self._frames.append(image)

    def on_run_end(self, sim: Simulation, result: RunResult) -> None:
        self.close()

    def close(self) -> None:
        if self._written or not self._frames:
            return
        import imageio.v3 as iio

        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        stack = np.stack(self._frames)
        if self._out_path.suffix.lower() == '.gif':
            iio.imwrite(self._out_path, stack,
                        duration=int(1000 / self._fps), loop=0)
        else:
            iio.imwrite(self._out_path, stack, fps=self._fps)
        self._written = True
