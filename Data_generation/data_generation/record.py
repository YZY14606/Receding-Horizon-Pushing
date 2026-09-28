import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from mani_skill.utils import common
from mani_skill.utils.visualization.misc import images_to_video, tile_images


class RecordVideo(gym.Wrapper):
    """Lightweight video recorder with episode naming for data collection."""

    def __init__(
        self,
        env,
        output_dir: str,
        trajectory_name: str | None = None,
        save_video: bool = False,
        video_fps: int = 30,
        avoid_overwriting_video: bool = False,
    ) -> None:
        super().__init__(env)
        self.output_dir = Path(output_dir)
        self.time_name = trajectory_name or time.strftime("%Y%m%d_%H%M%S")
        self.traj_id = "traj_0"
        self._episode_id = 0

        self.save_video = save_video
        self.video_fps = video_fps
        self.avoid_overwriting_video = avoid_overwriting_video
        self.render_images = []
        self._video_steps = 0
        self._video_id = -1
        self._closed = False

        if self.save_video:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_image(self):
        image = common.to_numpy(self.env.render())
        if image.ndim == 3:
            image = image[None]
        if image.ndim > 3:
            if len(image) == 1:
                image = image[0]
            else:
                image = tile_images(
                    image,
                    nrows=max(1, int(np.sqrt(self.unwrapped.num_envs))),
                )
        return image

    def step(self, action):
        if self.save_video and self._video_steps == 0:
            self.render_images.append(self.capture_image())

        result = super().step(action)

        if self.save_video:
            self.render_images.append(self.capture_image())
            self._video_steps += 1
        return result

    def advance_episode(self) -> None:
        """Advance the directory identity used by the data-generation code."""
        self._episode_id += 1
        self.traj_id = f"traj_{self._episode_id}"

    def flush_video(
        self,
        name: str | None = None,
        suffix: str = "",
        verbose: bool = False,
        ignore_empty_transition: bool = True,
        save: bool = True,
    ) -> None:
        if not self.render_images:
            return
        if ignore_empty_transition and len(self.render_images) == 1:
            self._clear_video_buffer()
            return

        if save:
            self._video_id += 1
            video_name = str(self._video_id) if name is None else name
            if suffix:
                video_name += f"_{suffix}"
            if self.avoid_overwriting_video:
                video_name = self._next_available_video_name(video_name)

            images_to_video(
                self.render_images,
                str(self.output_dir),
                video_name=video_name,
                fps=self.video_fps,
                verbose=verbose,
            )
        self._clear_video_buffer()

    def _next_available_video_name(self, video_name: str) -> str:
        candidate = video_name.replace(" ", "_").replace("\n", "_")
        while (self.output_dir / f"{candidate}.mp4").exists():
            self._video_id += 1
            candidate = str(self._video_id)
        return candidate

    def _clear_video_buffer(self) -> None:
        self._video_steps = 0
        self.render_images = []

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.save_video and self.render_images:
            self.flush_video()
        return super().close()
