"""Environment creation, preprocessing wrappers, and seeding helpers."""

from __future__ import annotations

from collections import deque
import os
import random
import sys

import cv2
import gymnasium as gym
import numpy as np
import torch

from off_road_wrapper import CarRacingOffRoadWrapper


def configure_headless_sdl(render_mode: str | None) -> None:
	"""Use SDL dummy video driver on headless Linux compute nodes when not rendering."""
	if (
		render_mode != "human"
		and sys.platform.startswith("linux")
		and not os.environ.get("DISPLAY")
	):
		os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def set_global_seeds(seed: int) -> None:
	"""Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


class ActionRepeatWrapper(gym.Wrapper):
	"""Repeats each chosen action for `skip` raw environment frames."""

	def __init__(self, env: gym.Env, skip: int = 4) -> None:
		super().__init__(env)
		if skip < 1:
			raise ValueError(f"frame_skip must be >= 1, got {skip}")
		self.skip = skip

	def step(self, action):
		total_reward = 0.0
		raw_frames = 0
		terminated = False
		truncated = False
		obs = None
		info = {}

		for _ in range(self.skip):
			obs, reward, terminated, truncated, info = self.env.step(action)
			total_reward += float(reward)
			raw_frames += 1
			if terminated or truncated:
				break

		info = dict(info)
		info["env_frames"] = raw_frames
		return obs, total_reward, terminated, truncated, info


class ImagePreprocessWrapper(gym.ObservationWrapper):
	"""Converts RGB frames (H, W, 3) into resized grayscale uint8 frames (S, S)."""

	def __init__(self, env: gym.Env, img_size: int = 84) -> None:
		super().__init__(env)
		self.img_size = img_size
		self.observation_space = gym.spaces.Box(
			low=0,
			high=255,
			shape=(img_size, img_size),
			dtype=np.uint8,
		)

	def observation(self, obs: np.ndarray) -> np.ndarray:
		if obs.ndim == 3 and obs.shape[-1] == 3:
			gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
		elif obs.ndim == 3 and obs.shape[-1] == 1:
			gray = obs[:, :, 0]
		else:
			gray = obs

		if gray.shape != (self.img_size, self.img_size):
			gray = cv2.resize(
				gray,
				(self.img_size, self.img_size),
				interpolation=cv2.INTER_AREA,
			)
		return gray.astype(np.uint8, copy=False)


class FrameStackWrapper(gym.ObservationWrapper):
	"""Stacks `num_stack` consecutive 2D frames into a (C, H, W) uint8 observation."""

	def __init__(self, env: gym.Env, num_stack: int = 4) -> None:
		super().__init__(env)
		if num_stack < 1:
			raise ValueError(f"frame_stack must be >= 1, got {num_stack}")
		self.num_stack = num_stack
		self.frames: deque[np.ndarray] = deque(maxlen=num_stack)

		assert len(env.observation_space.shape) == 2, (
			f"FrameStackWrapper expects 2D grayscale frames, got {env.observation_space.shape}"
		)
		h, w = env.observation_space.shape
		self.observation_space = gym.spaces.Box(
			low=0,
			high=255,
			shape=(num_stack, h, w),
			dtype=np.uint8,
		)

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		self.frames.clear()
		for _ in range(self.num_stack):
			self.frames.append(obs)
		return self._get_obs(), info

	def observation(self, obs: np.ndarray) -> np.ndarray:
		self.frames.append(obs)
		return self._get_obs()

	def _get_obs(self) -> np.ndarray:
		return np.stack(self.frames, axis=0)


def make_env(
	env_id: str = "CarRacing-v3",
	continuous: bool = True,
	render_mode: str | None = None,
	frame_skip: int = 4,
	frame_stack: int = 4,
	img_size: int = 84,
	seed: int | None = None,
	terminate_on_off_road: bool = False,
	off_road_penalty: float = 0.0,
) -> gym.Env:
	"""Create and wrap a Gymnasium environment for training or evaluation."""
	configure_headless_sdl(render_mode)

	if "CarRacing" in env_id:
		env = gym.make(env_id, render_mode=render_mode, continuous=continuous)
	else:
		env = gym.make(env_id, render_mode=render_mode)

	if terminate_on_off_road and "CarRacing" in env_id:
		env = CarRacingOffRoadWrapper(env, off_road_penalty=off_road_penalty)

	if frame_skip > 1:
		env = ActionRepeatWrapper(env, skip=frame_skip)

	if len(env.observation_space.shape) == 3:
		env = ImagePreprocessWrapper(env, img_size=img_size)
		env = FrameStackWrapper(env, num_stack=frame_stack)

	if seed is not None:
		env.action_space.seed(seed)
		env.observation_space.seed(seed)

	return env

