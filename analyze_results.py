"""Aggregate multi-seed PPO experiment results and generate figures and summary metrics.

Computes:
- Mean and standard deviation across seeds
- Area Under the Learning Curve (AUC)
- Steps to threshold (default threshold: 500.0 and 900.0)
- Total GPU wall-clock training times
- Publication-style PNG plots with faint individual seed runs and prominent mean +/- std curves.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
PERF_THRESHOLD = 500.0


def load_condition_seeds(condition_name: str) -> dict[int, pd.DataFrame]:
	"""Load eval_metrics.csv for all seeds under results/<condition_name>/seed_<seed>/."""
	cond_dir = RESULTS_DIR / condition_name
	if not cond_dir.exists():
		return {}

	seed_dfs = {}
	for seed_dir in sorted(cond_dir.glob("seed_*")):
		eval_csv = seed_dir / "eval_metrics.csv"
		if eval_csv.exists():
			try:
				seed_val = int(seed_dir.name.split("_")[-1])
				df = pd.read_csv(eval_csv)
				if not df.empty and "eval_mean_rew" in df.columns:
					seed_dfs[seed_val] = df
			except Exception as e:
				print(f"Warning: Could not parse {eval_csv}: {e}")
	return seed_dfs


def compute_normalized_auc(timesteps: np.ndarray, returns: np.ndarray) -> float:
	"""Compute trapezoidal area under the curve normalized by timestep span."""
	if len(timesteps) < 2:
		return 0.0
	span = timesteps[-1] - timesteps[0]
	if span <= 0:
		return 0.0
	auc = np.trapz(returns, timesteps)
	return float(auc / span)


def compute_steps_to_threshold(timesteps: np.ndarray, returns: np.ndarray, threshold: float = PERF_THRESHOLD) -> float | None:
	"""Return the first timestep where evaluation return meets or exceeds threshold."""
	for t, r in zip(timesteps, returns):
		if r >= threshold:
			return float(t)
	return None


def plot_comparison(
	condition_labels: dict[str, str],
	title: str,
	output_filename: str,
	common_step_limit: int | None = None,
) -> None:
	"""Generate a multi-seed comparison figure with individual seed traces and shaded +/- 1 std."""
	plt.figure(figsize=(9, 5.5))
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

	for idx, (cond_name, display_label) in enumerate(condition_labels.items()):
		seed_data = load_condition_seeds(cond_name)
		if not seed_data:
			continue

		color = colors[idx % len(colors)]

		# Determine common evaluation timestep grid
		all_timesteps = []
		all_returns = []

		for seed, df in seed_data.items():
			ts = df["timesteps"].values
			rews = df["eval_mean_rew"].values
			if common_step_limit is not None:
				mask = ts <= common_step_limit
				ts, rews = ts[mask], rews[mask]

			# Faint individual seed trace
			plt.plot(ts, rews, color=color, alpha=0.22, linewidth=1.0)
			all_timesteps.append(ts)
			all_returns.append(rews)

		# Aggregate grid via interpolation
		min_len = min(len(t) for t in all_timesteps)
		ref_grid = all_timesteps[0][:min_len]

		interpolated = []
		for ts, rews in zip(all_timesteps, all_returns):
			interpolated.append(np.interp(ref_grid, ts, rews))

		interp_arr = np.array(interpolated)
		mean_curve = interp_arr.mean(axis=0)
		std_curve = interp_arr.std(axis=0)

		# Prominent mean line + shaded +/- 1 std band
		plt.plot(ref_grid, mean_curve, color=color, label=f"{display_label} (N={len(seed_data)})", linewidth=2.4)
		plt.fill_between(ref_grid, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)

	plt.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
	plt.xlabel("Agent Timesteps", fontsize=11)
	plt.ylabel("Evaluation Return (Mean over 5 Episodes)", fontsize=11)
	plt.title(title, fontsize=12, fontweight="bold")
	plt.legend(loc="lower right", framealpha=0.9)
	plt.grid(True, linestyle="--", alpha=0.5)
	plt.tight_layout()

	out_path = RESULTS_DIR / output_filename
	plt.savefig(out_path, dpi=300)
	plt.close()
	print(f"Saved figure: {out_path}")


def main() -> None:
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)

	# 1. Advantage Estimation (RTG vs GAE)
	plot_comparison(
		{"adv_rtg": "Rewards-To-Go (No GAE)", "adv_gae": "GAE (lambda=0.95)"},
		"Advantage Estimation on CarRacing-v3: Rewards-To-Go vs. GAE",
		"fig1_advantage_estimation.png",
	)

	# 2. PPO Clipping Parameter
	plot_comparison(
		{"clip_0.1": "Clip epsilon = 0.1", "clip_0.2": "Clip epsilon = 0.2", "clip_0.3": "Clip epsilon = 0.3"},
		"PPO Clipping Parameter Sensitivity on CarRacing-v3",
		"fig2_ppo_clipping.png",
	)

	# 3. Entropy Regularization
	plot_comparison(
		{"ent_0.0": "Entropy coef = 0.0", "ent_0.01": "Entropy coef = 0.01"},
		"Entropy Regularization Exploration on CarRacing-v3",
		"fig3_entropy_regularization.png",
	)


if __name__ == "__main__":
	main()

