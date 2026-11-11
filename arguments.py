"""Command-line argument parser for PPO training and evaluation."""

import argparse


def get_args():
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(description="PPO for CarRacing-v3 and Gymnasium environments")

	# Mode & Checkpoints
	parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Run mode: train or test")
	parser.add_argument("--actor_model", type=str, default="", help="Path to actor checkpoint (.pth)")
	parser.add_argument("--critic_model", type=str, default="", help="Path to critic checkpoint (.pth)")

	# Environment settings
	parser.add_argument("--env", type=str, default="CarRacing-v3", help="Gymnasium environment ID")
	parser.add_argument("--continuous", dest="continuous", action="store_true", default=True, help="Use continuous action space (default)")
	parser.add_argument("--discrete", dest="continuous", action="store_false", help="Use discrete action space")
	parser.add_argument("--frame_skip", type=int, default=4, help="Action repeat frame skip")
	parser.add_argument("--frame_stack", type=int, default=4, help="Number of stacked grayscale frames")
	parser.add_argument("--img_size", type=int, default=84, help="Resized frame dimension (H=W=img_size)")

	# Experiment tracking & reproducibility
	parser.add_argument("--seed", type=int, default=11, help="Random seed (target seeds: 11, 22, 33, 44, 55)")
	parser.add_argument("--condition", type=str, default="baseline", help="Condition label for ablation studies")
	parser.add_argument("--output_dir", type=str, default="", help="Custom output directory; defaults to results/<condition>/seed_<seed>")
	parser.add_argument("--device", type=str, default="", help="Device: cuda, mps, cpu, or empty for auto-detect")

	# PPO Hyperparameters & Ablation variables
	parser.add_argument("--total_timesteps", type=int, default=500_000, help="Total agent timesteps to train")
	parser.add_argument("--timesteps_per_batch", type=int, default=2048, help="Rollout timesteps collected per iteration")
	parser.add_argument("--minibatch_size", type=int, default=128, help="PPO SGD minibatch size")
	parser.add_argument("--n_updates_per_iteration", type=int, default=10, help="Number of gradient update epochs per rollout")
	parser.add_argument("--max_timesteps_per_episode", type=int, default=1000, help="Max steps per episode (matches CarRacing 1000-step limit)")
	parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for Adam optimizer")
	parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
	parser.add_argument("--clip", type=float, default=0.2, help="PPO clipping parameter epsilon (0.1, 0.2, 0.3)")
	parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy bonus coefficient (e.g. 0.0 vs 0.01)")
	parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Maximum gradient norm clipping")

	# Advantage Estimation Comparison
	parser.add_argument("--use_gae", dest="use_gae", action="store_true", default=True, help="Use Generalized Advantage Estimation (default)")
	parser.add_argument("--no_gae", dest="use_gae", action="store_false", help="Disable GAE, use Monte Carlo Rewards-To-Go")
	parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")

	# Evaluation & Logging
	parser.add_argument("--eval_freq", type=int, default=10, help="Evaluation frequency in rollout iterations")
	parser.add_argument("--eval_episodes", type=int, default=5, help="Number of episodes per evaluation")
	parser.add_argument("--save_freq", type=int, default=10, help="Checkpoint save frequency in rollout iterations")
	parser.add_argument("--render", action="store_true", default=False, help="Render environment display window")

	return parser.parse_args()
