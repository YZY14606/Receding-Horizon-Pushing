"""Minimal Gym wrapper for recording rendered episodes as videos."""

from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np

from mani_skill.utils import common
from mani_skill.utils.visualization.misc import images_to_video, tile_images


class VideoRecorder(gym.Wrapper):
    """Capture environment renders and write one video per accepted episode."""

    def __init__(
        self,
        env: gym.Env,
        output_dir: str,
        enabled: bool = False,
        fps: int = 30,
    ) -> None:
        super().__init__(env)
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.fps = fps
        self.frames = []
        self._episode_index = 0
        self._closed = False

    @property
    def traj_id(self) -> str:
        """Identifier shared by video, pose, push, and visualization outputs."""
        return f"traj_{self._episode_index}"

    def capture_frame(self):
        frame = common.to_numpy(self.env.render())
        if frame.ndim == 3:
            return frame
        if len(frame) == 1:
            return frame[0]
        return tile_images(frame, nrows=int(np.sqrt(len(frame))))

    def reset(self, *args, **kwargs):
        self.discard_video()
        return super().reset(*args, **kwargs)

    def step(self, action):
        if self.enabled and not self.frames:
            self.frames.append(self.capture_frame())

        result = super().step(action)

        if self.enabled:
            self.frames.append(self.capture_frame())
        return result

    def flush_video(
        self,
        dir_path: Optional[str] = None,
        name: Optional[str] = None,
        save: bool = True,
    ) -> None:
        """Save the buffered frames, or clear them when save is false."""
        if save and self.enabled and len(self.frames) > 1:
            output_dir = Path(dir_path) if dir_path is not None else self.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            images_to_video(
                self.frames,
                str(output_dir),
                video_name=name or self.traj_id,
                fps=self.fps,
            )
        self.frames.clear()

    def discard_video(self) -> None:
        self.frames.clear()

    def advance_episode(self) -> None:
        """Advance the shared artifact ID after an accepted episode."""
        self.discard_video()
        self._episode_index += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.discard_video()
        super().close()
