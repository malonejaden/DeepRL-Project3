"""Proximal Policy Optimization (PPO-Clip) implementation with GPU and GAE support."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import platform
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from torch.optim import Adam

from network import CNNNetwork, FeedForwardNN


class PPO:
	"""PPO-Clip agent supporting discrete and continuous actions, CNN and MLP backbones, and GAE."""

	def __init__(self, policy_class=None, env: gym.Env | None = None, **hyperparameters) -> None:
		assert env is not None, "PPO requires an environment instance"
		self.env = env
		self._init_hyperparameters(hyperparameters)

		# Device selection: CUDA -> MPS -> CPU
		if self.device_str:
			self.device = torch.device(self.device_str)
		elif torch.cuda.is_available():
			self.device = torch.device("cuda")
		elif torch.backends.mps.is_available():
			self.device = torch.device("mps")
		else:
			self.device = torch.device("cpu")

		# Detect action space type
		if isinstance(env.action_space, gym.spaces.Discrete):
			self.is_discrete = True
			self.act_dim = int(env.action_space.n)
			self.act_low = None
			self.act_high = None
		elif isinstance(env.action_space, gym.spaces.Box):
			self.is_discrete = False
			self.act_dim = int(np.prod(env.action_space.shape))
			self.act_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.device)
			self.act_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.device)
			self.log_std = nn.Parameter(torch.full((self.act_dim,), -0.5, device=self.device))
		else:
			raise ValueError(f"Unsupported action space: {env.action_space}")

		# Initialize Actor and Critic networks
		obs_shape = env.observation_space.shape
		self.is_image = (len(obs_shape) == 3)

		if policy_class is not None:
			self.actor = policy_class(obs_shape[0], self.act_dim).to(self.device)
			self.critic = policy_class(obs_shape[0], 1).to(self.device)
		elif self.is_image:
			in_channels, img_size = obs_shape[0], obs_shape[1]
			self.actor = CNNNetwork(in_channels, self.act_dim, img_size=img_size, output_gain=0.01).to(self.device)
			self.critic = CNNNetwork(in_channels, 1, img_size=img_size, output_gain=1.0).to(self.device)
		else:
			in_dim = int(np.prod(obs_shape))
			self.actor = FeedForwardNN(in_dim, self.act_dim, output_gain=0.01).to(self.device)
			self.critic = FeedForwardNN(in_dim, 1, output_gain=1.0).to(self.device)

		# Optimizers
		actor_params = list(self.actor.parameters())
		if not self.is_discrete:
			actor_params.append(self.log_std)
		self.actor_optim = Adam(actor_params, lr=self.lr, eps=1e-5)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)

		# Output directory and CSV loggers
		self.output_dir = Path(self.output_dir_str) if self.output_dir_str else Path(f"results/{self.condition}/seed_{self.seed}")
		self.output_dir.mkdir(parents=True, exist_ok=True)
		(self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

		self._init_csv_loggers()
		self._save_config()

		self.best_eval_return = -float("inf")
		self.total_env_frames = 0

	def _init_hyperparameters(self, hyperparameters: dict) -> None:
		"""Initialize default hyperparameters and override with kwargs."""
		self.condition = "baseline"
		self.seed = 11
		self.output_dir_str = ""
		self.device_str = ""

		self.total_timesteps = 500_000
		self.timesteps_per_batch = 2048
		self.minibatch_size = 128
		self.n_updates_per_iteration = 10
		self.max_timesteps_per_episode = 1000
		self.lr = 3e-4
		self.gamma = 0.99
		self.clip = 0.2
		self.ent_coef = 0.01
		self.max_grad_norm = 0.5

		self.use_gae = True
		self.gae_lambda = 0.95

		self.eval_freq = 10
		self.eval_episodes = 5
		self.save_freq = 10
		self.render = False

		for param, val in hyperparameters.items():
			setattr(self, param, val)

	def _init_csv_loggers(self) -> None:
		self.train_csv_path = self.output_dir / "train_metrics.csv"
		if not self.train_csv_path.exists():
			with open(self.train_csv_path, "w", newline="", encoding="utf-8") as f:
				writer = csv.writer(f)
				writer.writerow([
					"iteration", "timesteps", "env_frames", "episodes_done",
					"mean_ep_len", "mean_ep_rew", "actor_loss", "critic_loss",
					"entropy", "iter_time_sec", "total_time_sec", "steps_per_sec",
				])

		self.eval_csv_path = self.output_dir / "eval_metrics.csv"
		if not self.eval_csv_path.exists():
			with open(self.eval_csv_path, "w", newline="", encoding="utf-8") as f:
				writer = csv.writer(f)
				writer.writerow([
					"iteration", "timesteps", "env_frames",
					"eval_mean_rew", "eval_std_rew", "eval_mean_len", "total_time_sec",
				])

	def _save_config(self) -> None:
		device_name = str(self.device)
		if self.device.type == "cuda":
			device_name = f"{torch.cuda.get_device_name(0)} ({self.device})"

		config = {
			"condition": self.condition,
			"seed": self.seed,
			"device": device_name,
			"system": platform.platform(),
			"python": platform.python_version(),
			"torch": torch.__version__,
			"gymnasium": gym.__version__,
			"is_discrete": self.is_discrete,
			"is_image": self.is_image,
			"total_timesteps": self.total_timesteps,
			"timesteps_per_batch": self.timesteps_per_batch,
			"minibatch_size": self.minibatch_size,
			"n_updates_per_iteration": self.n_updates_per_iteration,
			"max_timesteps_per_episode": self.max_timesteps_per_episode,
			"lr": self.lr,
			"gamma": self.gamma,
			"use_gae": self.use_gae,
			"gae_lambda": self.gae_lambda,
			"clip": self.clip,
			"ent_coef": self.ent_coef,
			"max_grad_norm": self.max_grad_norm,
		}
		with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
			json.dump(config, f, indent=2)

	def get_action(self, obs: np.ndarray, deterministic: bool = False):
		"""Query an action from the policy."""
		with torch.no_grad():
			if self.is_discrete:
				logits = self.actor(obs)
				dist = Categorical(logits=logits)
				action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
				log_prob = dist.log_prob(action)
				return int(action.item()), action.detach(), log_prob.detach()
			else:
				mean = self.actor(obs)
				std = self.log_std.exp().expand_as(mean)
				dist = Normal(mean, std)
				action = mean if deterministic else dist.sample()
				log_prob = dist.log_prob(action).sum(dim=-1)
				clamped_action = torch.clamp(action, self.act_low, self.act_high)
				return clamped_action.cpu().numpy(), action.detach(), log_prob.detach()

	def evaluate(self, batch_obs: torch.Tensor, batch_acts: torch.Tensor):
		"""Evaluate batch observations and actions."""
		V = self.critic(batch_obs).squeeze(-1)
		if self.is_discrete:
			logits = self.actor(batch_obs)
			dist = Categorical(logits=logits)
			log_probs = dist.log_prob(batch_acts)
			entropy = dist.entropy()
		else:
			mean = self.actor(batch_obs)
			std = self.log_std.exp().expand_as(mean)
			dist = Normal(mean, std)
			log_probs = dist.log_prob(batch_acts).sum(dim=-1)
			entropy = dist.entropy().sum(dim=-1)
		return V, log_probs, entropy

	def rollout(self):
		"""Collect a batch of simulated transitions and compute returns/advantages."""
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_returns = []
		batch_advantages = []
		batch_lens = []
		batch_rews = []
		batch_raw_frames = 0

		t = 0
		while t < self.timesteps_per_batch:
			ep_obs = []
			ep_acts = []
			ep_log_probs = []
			ep_rews = []

			obs, _ = self.env.reset()
			done = False
			terminated = False
			truncated = False

			for _ in range(self.max_timesteps_per_episode):
				t += 1
				ep_obs.append(obs)

				env_action, raw_action, log_prob = self.get_action(obs, deterministic=False)
				next_obs, rew, terminated, truncated, info = self.env.step(env_action)
				batch_raw_frames += int(info.get("env_frames", 1))

				done = terminated or truncated
				ep_rews.append(rew)
				ep_acts.append(raw_action)
				ep_log_probs.append(log_prob)

				obs = next_obs
				if done:
					break

			# Compute advantages and returns for this completed episode
			ep_T = len(ep_rews)
			batch_lens.append(ep_T)
			batch_rews.append(ep_rews)

			# Convert episode observations to tensor to get critic values
			ep_obs_arr = np.asarray(ep_obs)
			with torch.no_grad():
				ep_obs_tensor = torch.as_tensor(ep_obs_arr, device=self.device)
				ep_vals = self.critic(ep_obs_tensor).squeeze(-1)
				if ep_vals.ndim == 0:
					ep_vals = ep_vals.unsqueeze(0)

				# Bootstrap value if episode truncated (time limit reached, not terminal crash)
				if terminated:
					last_val = 0.0
				else:
					last_val = self.critic(next_obs).squeeze(-1).item()

				ep_advs = np.zeros(ep_T, dtype=np.float32)
				ep_rets = np.zeros(ep_T, dtype=np.float32)

				if self.use_gae:
					gae = 0.0
					for step in reversed(range(ep_T)):
						next_v = last_val if step == ep_T - 1 else ep_vals[step + 1].item()
						delta = ep_rews[step] + self.gamma * next_v - ep_vals[step].item()
						gae = delta + self.gamma * self.gae_lambda * gae
						ep_advs[step] = gae
						ep_rets[step] = gae + ep_vals[step].item()
				else:
					running_ret = last_val
					for step in reversed(range(ep_T)):
						running_ret = ep_rews[step] + self.gamma * running_ret
						ep_rets[step] = running_ret
						ep_advs[step] = running_ret - ep_vals[step].item()

			batch_obs.extend(ep_obs)
			batch_acts.extend(ep_acts)
			batch_log_probs.extend(ep_log_probs)
			batch_returns.extend(ep_rets)
			batch_advantages.extend(ep_advs)

		# Convert batch lists to tensors on target device
		batch_obs_tensor = torch.as_tensor(np.asarray(batch_obs), device=self.device)
		if self.is_discrete:
			batch_acts_tensor = torch.stack(batch_acts).long().to(self.device)
		else:
			batch_acts_tensor = torch.stack(batch_acts).float().to(self.device)
		batch_log_probs_tensor = torch.stack(batch_log_probs).float().to(self.device)
		batch_returns_tensor = torch.as_tensor(np.asarray(batch_returns), dtype=torch.float32, device=self.device)
		batch_advantages_tensor = torch.as_tensor(np.asarray(batch_advantages), dtype=torch.float32, device=self.device)

		# Normalize advantages across batch
		batch_advantages_tensor = (batch_advantages_tensor - batch_advantages_tensor.mean()) / (
			batch_advantages_tensor.std() + 1e-8
		)

		return (
			batch_obs_tensor,
			batch_acts_tensor,
			batch_log_probs_tensor,
			batch_returns_tensor,
			batch_advantages_tensor,
			batch_lens,
			batch_rews,
			batch_raw_frames,
		)

	def evaluate_policy(self, eval_env: gym.Env) -> tuple[float, float, float]:
		"""Run deterministic evaluation episodes and return mean reward, std reward, and mean length."""
		returns = []
		lengths = []
		for _ in range(self.eval_episodes):
			obs, _ = eval_env.reset()
			done = False
			ep_rew = 0.0
			ep_len = 0
			while not done and ep_len < self.max_timesteps_per_episode:
				env_action, _, _ = self.get_action(obs, deterministic=True)
				obs, rew, terminated, truncated, _ = eval_env.step(env_action)
				ep_rew += float(rew)
				ep_len += 1
				done = terminated or truncated
			returns.append(ep_rew)
			lengths.append(ep_len)

		return float(np.mean(returns)), float(np.std(returns)), float(np.mean(lengths))

	def learn(self, total_timesteps: int | None = None, eval_env: gym.Env | None = None) -> None:
		"""Main PPO training loop with minibatch SGD updates and evaluation."""
		total_timesteps = total_timesteps or self.total_timesteps
		print(f"[{self.condition} | Seed {self.seed}] Training on {self.device} for {total_timesteps:,} timesteps...")
		print(f"PPO Settings: GAE={self.use_gae} (lambda={self.gae_lambda}), Clip={self.clip}, EntCoef={self.ent_coef}, LR={self.lr}")

		t_so_far = 0
		i_so_far = 0
		episodes_so_far = 0
		start_time = time.time()

		# Initial evaluation before training begins
		if eval_env is not None:
			eval_mean, eval_std, eval_len = self.evaluate_policy(eval_env)
			print(f"[Eval @ Step 0] Mean Return: {eval_mean:.2f} +/- {eval_std:.2f} | Mean Len: {eval_len:.1f}")
			with open(self.eval_csv_path, "a", newline="", encoding="utf-8") as f:
				writer = csv.writer(f)
				writer.writerow([0, 0, 0, eval_mean, eval_std, eval_len, 0.0])

		while t_so_far < total_timesteps:
			iter_start = time.time()

			(
				batch_obs,
				batch_acts,
				batch_log_probs,
				batch_returns,
				batch_advs,
				batch_lens,
				batch_rews,
				batch_raw_frames,
			) = self.rollout()

			batch_size = len(batch_obs)
			t_so_far += batch_size
			self.total_env_frames += batch_raw_frames
			i_so_far += 1
			episodes_so_far += len(batch_lens)

			actor_losses = []
			critic_losses = []
			entropies = []

			# Minibatch SGD updates
			for _ in range(self.n_updates_per_iteration):
				indices = torch.randperm(batch_size, device=self.device)
				for start_idx in range(0, batch_size, self.minibatch_size):
					mb_idx = indices[start_idx : start_idx + self.minibatch_size]

					mb_obs = batch_obs[mb_idx]
					mb_acts = batch_acts[mb_idx]
					mb_old_log_probs = batch_log_probs[mb_idx]
					mb_returns = batch_returns[mb_idx]
					mb_advs = batch_advs[mb_idx]

					V, curr_log_probs, entropy = self.evaluate(mb_obs, mb_acts)

					# Policy surrogate loss
					ratios = torch.exp(curr_log_probs - mb_old_log_probs)
					surr1 = ratios * mb_advs
					surr2 = torch.clamp(ratios, 1.0 - self.clip, 1.0 + self.clip) * mb_advs
					actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy.mean()

					# Value loss
					critic_loss = 0.5 * nn.MSELoss()(V, mb_returns)

					# Actor update
					self.actor_optim.zero_grad()
					actor_loss.backward()
					nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
					self.actor_optim.step()

					# Critic update
					self.critic_optim.zero_grad()
					critic_loss.backward()
					nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
					self.critic_optim.step()

					actor_losses.append(actor_loss.item())
					critic_losses.append(critic_loss.item())
					entropies.append(entropy.mean().item())

			iter_time = time.time() - iter_start
			total_time = time.time() - start_time
			steps_per_sec = batch_size / max(iter_time, 1e-6)

			avg_rew = float(np.mean([sum(r) for r in batch_rews]))
			avg_len = float(np.mean(batch_lens))
			avg_aloss = float(np.mean(actor_losses))
			avg_closs = float(np.mean(critic_losses))
			avg_ent = float(np.mean(entropies))

			# Print summary
			print(
				f"Iter {i_so_far:03d} | Timesteps {t_so_far:7d}/{total_timesteps} "
				f"| EpRew {avg_rew:7.2f} | EpLen {avg_len:5.1f} "
				f"| ALoss {avg_aloss:7.4f} | CLoss {avg_closs:7.4f} | Ent {avg_ent:5.3f} "
				f"| {steps_per_sec:5.1f} steps/s"
			)

			# Log training metrics
			with open(self.train_csv_path, "a", newline="", encoding="utf-8") as f:
				writer = csv.writer(f)
				writer.writerow([
					i_so_far, t_so_far, self.total_env_frames, episodes_so_far,
					f"{avg_len:.2f}", f"{avg_rew:.2f}", f"{avg_aloss:.5f}", f"{avg_closs:.5f}",
					f"{avg_ent:.4f}", f"{iter_time:.2f}", f"{total_time:.2f}", f"{steps_per_sec:.1f}",
				])

			# Periodic evaluation
			if eval_env is not None and (i_so_far % self.eval_freq == 0 or t_so_far >= total_timesteps):
				eval_mean, eval_std, eval_len = self.evaluate_policy(eval_env)
				print(f" >>> [Eval @ Iter {i_so_far}] Mean Return: {eval_mean:.2f} +/- {eval_std:.2f} (Len: {eval_len:.1f})")
				with open(self.eval_csv_path, "a", newline="", encoding="utf-8") as f:
					writer = csv.writer(f)
					writer.writerow([
						i_so_far, t_so_far, self.total_env_frames,
						f"{eval_mean:.2f}", f"{eval_std:.2f}", f"{eval_len:.2f}", f"{total_time:.2f}",
					])

				if eval_mean > self.best_eval_return:
					self.best_eval_return = eval_mean
					self.save_checkpoint("best")

			# Periodic checkpoint saving
			if i_so_far % self.save_freq == 0 or t_so_far >= total_timesteps:
				self.save_checkpoint("latest")

		print(f"Training completed in {total_time / 60.0:.2f} minutes. Outputs saved to {self.output_dir}")

	def save_checkpoint(self, tag: str) -> None:
		"""Save actor and critic state dicts and extra parameters."""
		ckpt_dir = self.output_dir / "checkpoints"
		torch.save(self.actor.state_dict(), ckpt_dir / f"actor_{tag}.pth")
		torch.save(self.critic.state_dict(), ckpt_dir / f"critic_{tag}.pth")
		extra = {
			"is_discrete": self.is_discrete,
			"act_dim": self.act_dim,
		}
		if not self.is_discrete:
			extra["log_std"] = self.log_std.detach().cpu()
		torch.save(extra, ckpt_dir / f"extra_{tag}.pth")
