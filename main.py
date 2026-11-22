"""Main entry point for training and evaluating PPO on CarRacing-v3."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from arguments import get_args
from env_utils import make_env, set_global_seeds
from ppo import PPO


def train(args) -> None:
	"""Set up environments, build PPO agent, and execute training loop."""
	set_global_seeds(args.seed)

	render_mode = "human" if args.render else None
	train_env = make_env(
		env_id=args.env,
		continuous=args.continuous,
		render_mode=render_mode,
		frame_skip=args.frame_skip,
		frame_stack=args.frame_stack,
		img_size=args.img_size,
		seed=args.seed,
		terminate_on_off_road=args.terminate_on_off_road,
		off_road_penalty=args.off_road_penalty,
	)

	# Dedicated evaluation environment with separate seed
	eval_env = make_env(
		env_id=args.env,
		continuous=args.continuous,
		render_mode=None,
		frame_skip=args.frame_skip,
		frame_stack=args.frame_stack,
		img_size=args.img_size,
		seed=args.seed + 1000,
		terminate_on_off_road=args.terminate_on_off_road,
		off_road_penalty=args.off_road_penalty,
	)

	hyperparameters = {
		"condition": args.condition,
		"seed": args.seed,
		"output_dir_str": args.output_dir,
		"device_str": args.device,
		"total_timesteps": args.total_timesteps,
		"timesteps_per_batch": args.timesteps_per_batch,
		"minibatch_size": args.minibatch_size,
		"n_updates_per_iteration": args.n_updates_per_iteration,
		"max_timesteps_per_episode": args.max_timesteps_per_episode,
		"lr": args.lr,
		"gamma": args.gamma,
		"clip": args.clip,
		"ent_coef": args.ent_coef,
		"max_grad_norm": args.max_grad_norm,
		"use_gae": args.use_gae,
		"gae_lambda": args.gae_lambda,
		"eval_freq": args.eval_freq,
		"eval_episodes": args.eval_episodes,
		"save_freq": args.save_freq,
		"render": args.render,
	}

	model = PPO(policy_class=None, env=train_env, **hyperparameters)

	# Optionally load existing weights to continue training
	if args.actor_model != "" and args.critic_model != "":
		print(f"Loading checkpoint weights: {args.actor_model} and {args.critic_model}")
		model.actor.load_state_dict(torch.load(args.actor_model, map_location=model.device))
		model.critic.load_state_dict(torch.load(args.critic_model, map_location=model.device))
		extra_path = Path(args.actor_model).parent / "extra_best.pth"
		if not model.is_discrete and extra_path.exists():
			extra_data = torch.load(extra_path, map_location=model.device)
			if "log_std" in extra_data:
				model.log_std.data.copy_(extra_data["log_std"].to(model.device))
		print("Successfully loaded model checkpoints.")
	elif args.actor_model != "" or args.critic_model != "":
		print("Error: Specify both --actor_model and --critic_model to continue training.")
		sys.exit(1)

	try:
		model.learn(total_timesteps=args.total_timesteps, eval_env=eval_env)
	finally:
		train_env.close()
		eval_env.close()


def test(args) -> None:
	"""Evaluate a trained actor checkpoint."""
	if not args.actor_model:
		print("Error: Provide --actor_model path to test a trained agent.")
		sys.exit(1)

	set_global_seeds(args.seed)
	render_mode = "human" if args.render else None
	test_env = make_env(
		env_id=args.env,
		continuous=args.continuous,
		render_mode=render_mode,
		frame_skip=args.frame_skip,
		frame_stack=args.frame_stack,
		img_size=args.img_size,
		seed=args.seed + 2000,
		terminate_on_off_road=args.terminate_on_off_road,
		off_road_penalty=args.off_road_penalty,
	)

	model = PPO(policy_class=None, env=test_env, device_str=args.device)
	print(f"Loading actor checkpoint: {args.actor_model}")
	model.actor.load_state_dict(torch.load(args.actor_model, map_location=model.device))
	model.actor.eval()

	extra_path = Path(args.actor_model).parent / "extra_best.pth"
	if not model.is_discrete and extra_path.exists():
		extra_data = torch.load(extra_path, map_location=model.device)
		if "log_std" in extra_data:
			model.log_std.data.copy_(extra_data["log_std"].to(model.device))

	print(f"Running {args.eval_episodes} evaluation episodes (deterministic=True)...")
	returns = []
	lengths = []

	for ep in range(1, args.eval_episodes + 1):
		obs, _ = test_env.reset()
		done = False
		ep_rew = 0.0
		ep_len = 0

		while not done and ep_len < args.max_timesteps_per_episode:
			env_action, _, _ = model.get_action(obs, deterministic=True)
			obs, rew, terminated, truncated, _ = test_env.step(env_action)
			ep_rew += float(rew)
			ep_len += 1
			done = terminated or truncated

		returns.append(ep_rew)
		lengths.append(ep_len)
		print(f"Episode {ep:02d}: Return = {ep_rew:.2f}, Length = {ep_len}")

	mean_ret, std_ret = float(np.mean(returns)), float(np.std(returns))
	mean_len = float(np.mean(lengths))
	print(f"\nFinal Test Results over {args.eval_episodes} episodes:")
	print(f"Mean Return: {mean_ret:.2f} +/- {std_ret:.2f}")
	print(f"Mean Length: {mean_len:.1f}")
	test_env.close()


def main():
	args = get_args()
	if args.mode == "train":
		train(args)
	elif args.mode == "test":
		test(args)


if __name__ == "__main__":
	main()
