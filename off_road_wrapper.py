"""Custom Gymnasium wrapper for CarRacing off-road early termination and reward shaping."""

from __future__ import annotations

import gymnasium as gym


class CarRacingOffRoadWrapper(gym.Wrapper):
	"""Terminates an episode when the car drives off the road.

	In Gymnasium's CarRacing (v2/v3), the track is composed of discrete asphalt tiles.
	Box2D's contact listener (FrictionDetector) tracks road tile contacts in real time:
	each wheel in `car.wheels` maintains a `tiles` set containing the track tiles currently
	in contact with that wheel. When a wheel leaves the asphalt and touches grass,
	its `tiles` set becomes empty.

	Parameters
	----------
	env : gym.Env
		The Gymnasium environment to wrap (typically CarRacing-v3).
	all_wheels_off : bool, default=True
		If True, terminates when ALL wheels leave the road (completely off-track).
		If False, terminates when ANY wheel leaves the road (stricter boundary).
	max_consecutive_off_road : int, default=1
		Number of consecutive environment steps off-road before triggering termination.
		Set to 1 for immediate termination the instant the car is off the road.
	off_road_penalty : float, default=0.0
		Optional extra penalty subtracted from the reward when off-road termination occurs
		(e.g., 0.0 to just end the episode, or 100.0 to match CarRacing crash penalty).
	"""

	def __init__(
		self,
		env: gym.Env,
		all_wheels_off: bool = True,
		max_consecutive_off_road: int = 1,
		off_road_penalty: float = 0.0,
	) -> None:
		super().__init__(env)
		self.all_wheels_off = all_wheels_off
		self.max_consecutive_off_road = max(1, int(max_consecutive_off_road))
		self.off_road_penalty = float(off_road_penalty)
		self.off_road_steps = 0

	def reset(self, **kwargs):
		self.off_road_steps = 0
		return self.env.reset(**kwargs)

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)

		# Do not override if the base environment already ended the episode
		if terminated or truncated:
			return obs, reward, terminated, truncated, info

		# Check if the car is currently off the road
		if self._is_car_off_road():
			self.off_road_steps += 1
			if self.off_road_steps >= self.max_consecutive_off_road:
				terminated = True
				reward -= self.off_road_penalty
				info = dict(info)
				info["off_road_termination"] = True
		else:
			self.off_road_steps = 0

		return obs, reward, terminated, truncated, info

	def _is_car_off_road(self) -> bool:
		"""Query CarRacing's Box2D wheel objects to check track tile contact."""
		unwrapped = self.env.unwrapped
		car = getattr(unwrapped, "car", None)
		if car is None:
			return False

		wheels = getattr(car, "wheels", None)
		if not wheels:
			return False

		# A wheel is on the asphalt if its `tiles` set has at least one tile
		wheels_on_road = sum(1 for w in wheels if bool(getattr(w, "tiles", None)))

		if self.all_wheels_off:
			# Off-road when all 4 wheels have left the track
			return wheels_on_road == 0
		else:
			# Off-road when at least 1 wheel has left the track
			return wheels_on_road < len(wheels)
