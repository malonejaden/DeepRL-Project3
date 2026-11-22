"""Aggregate multi-seed PPO experiment results and generate figures and summary metrics.

Computes:
- Mean and standard deviation across seeds
- Area Under the Learning Curve (AUC)
- Steps to threshold (500.0 and 900.0)
- Total GPU wall-clock training times and mean episode lengths
- Comparison plots (SVG vector figure + publication PNGs when matplotlib is installed)
- Summary tables exported to CSV, Markdown, and stdout.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
	import pandas as pd
except ImportError:
	pd = None

try:
	import matplotlib.pyplot as plt
except ImportError:
	plt = None

RESULTS_DIR = Path("results")
PERF_THRESHOLD_500 = 500.0
PERF_THRESHOLD_900 = 900.0


def load_condition_seeds(condition_name: str) -> dict[int, dict[str, np.ndarray]]:
	"""Load eval_metrics.csv for all seeds under results/<condition_name>/seed_<seed>/."""
	cond_dir = RESULTS_DIR / condition_name
	if not cond_dir.exists():
		return {}

	seed_data: dict[int, dict[str, np.ndarray]] = {}
	for seed_dir in sorted(cond_dir.glob("seed_*")):
		eval_csv = seed_dir / "eval_metrics.csv"
		if not eval_csv.exists():
			continue

		try:
			seed_val = int(seed_dir.name.split("_")[-1])
		except ValueError:
			continue

		try:
			if pd is not None:
				df = pd.read_csv(eval_csv)
				if df.empty or "eval_mean_rew" not in df.columns:
					continue
				seed_data[seed_val] = {
					"timesteps": df["timesteps"].to_numpy(dtype=float),
					"eval_mean_rew": df["eval_mean_rew"].to_numpy(dtype=float),
					"eval_std_rew": df["eval_std_rew"].to_numpy(dtype=float) if "eval_std_rew" in df.columns else np.zeros(len(df)),
					"eval_mean_len": df["eval_mean_len"].to_numpy(dtype=float) if "eval_mean_len" in df.columns else np.zeros(len(df)),
					"total_time_sec": df["total_time_sec"].to_numpy(dtype=float) if "total_time_sec" in df.columns else np.zeros(len(df)),
				}
			else:
				with open(eval_csv, mode="r", encoding="utf-8") as f:
					reader = csv.DictReader(f)
					rows = list(reader)
				if not rows or "eval_mean_rew" not in rows[0]:
					continue
				seed_data[seed_val] = {
					"timesteps": np.array([float(r["timesteps"]) for r in rows]),
					"eval_mean_rew": np.array([float(r["eval_mean_rew"]) for r in rows]),
					"eval_std_rew": np.array([float(r.get("eval_std_rew", 0.0)) for r in rows]),
					"eval_mean_len": np.array([float(r.get("eval_mean_len", 0.0)) for r in rows]),
					"total_time_sec": np.array([float(r.get("total_time_sec", 0.0)) for r in rows]),
				}
		except Exception as e:
			print(f"Warning: Could not parse {eval_csv}: {e}")

	return seed_data


def compute_normalized_auc(timesteps: np.ndarray, returns: np.ndarray) -> float:
	"""Compute trapezoidal area under the curve normalized by timestep span."""
	if len(timesteps) < 2:
		return 0.0
	span = timesteps[-1] - timesteps[0]
	if span <= 0:
		return 0.0
	trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
	if trapz_fn is not None:
		auc = trapz_fn(returns, timesteps)
	else:
		auc = np.sum((returns[:-1] + returns[1:]) * 0.5 * np.diff(timesteps))
	return float(auc / span)


def compute_steps_to_threshold(
	timesteps: np.ndarray, returns: np.ndarray, threshold: float = PERF_THRESHOLD_500
) -> float | None:
	"""Return the first timestep where evaluation return meets or exceeds threshold."""
	for t, r in zip(timesteps, returns):
		if r >= threshold:
			return float(t)
	return None


def summarize_condition(condition_name: str, display_label: str, seed_data: dict[int, dict[str, np.ndarray]]) -> dict[str, Any]:
	"""Compute aggregate multi-seed metrics for a given experimental condition."""
	if not seed_data:
		return {}

	finals = []
	bests = []
	aucs = []
	steps_500 = []
	steps_900 = []
	total_times_hrs = []
	final_lengths = []

	per_seed_info = {}

	for seed, data in seed_data.items():
		ts = data["timesteps"]
		rews = data["eval_mean_rew"]
		lens = data["eval_mean_len"]
		times = data["total_time_sec"]

		final_rew = float(rews[-1]) if len(rews) > 0 else 0.0
		best_rew = float(np.max(rews)) if len(rews) > 0 else 0.0
		auc = compute_normalized_auc(ts, rews)
		s500 = compute_steps_to_threshold(ts, rews, PERF_THRESHOLD_500)
		s900 = compute_steps_to_threshold(ts, rews, PERF_THRESHOLD_900)
		time_hrs = float(times[-1] / 3600.0) if len(times) > 0 else 0.0
		final_len = float(lens[-1]) if len(lens) > 0 else 0.0

		finals.append(final_rew)
		bests.append(best_rew)
		aucs.append(auc)
		if s500 is not None:
			steps_500.append(s500)
		if s900 is not None:
			steps_900.append(s900)
		total_times_hrs.append(time_hrs)
		final_lengths.append(final_len)

		per_seed_info[seed] = {
			"final_rew": final_rew,
			"best_rew": best_rew,
			"auc": auc,
			"steps_500": s500,
			"steps_900": s900,
			"time_hrs": time_hrs,
			"final_len": final_len,
		}

	n_seeds = len(seed_data)
	return {
		"condition": condition_name,
		"label": display_label,
		"n_seeds": n_seeds,
		"final_mean": float(np.mean(finals)),
		"final_std": float(np.std(finals)),
		"best_mean": float(np.mean(bests)),
		"best_std": float(np.std(bests)),
		"auc_mean": float(np.mean(aucs)),
		"auc_std": float(np.std(aucs)),
		"reached_500_count": len(steps_500),
		"reached_500_ratio": f"{len(steps_500)}/{n_seeds}",
		"steps_500_avg": float(np.mean(steps_500)) if steps_500 else None,
		"reached_900_count": len(steps_900),
		"reached_900_ratio": f"{len(steps_900)}/{n_seeds}",
		"steps_900_avg": float(np.mean(steps_900)) if steps_900 else None,
		"time_hrs_mean": float(np.mean(total_times_hrs)),
		"time_hrs_std": float(np.std(total_times_hrs)),
		"final_len_mean": float(np.mean(final_lengths)),
		"final_len_std": float(np.std(final_lengths)),
		"per_seed": per_seed_info,
	}


def plot_comparison_svg(
	condition_labels: dict[str, str],
	title: str,
	output_filename: str = "fig_baseline_vs_off_road.svg",
) -> bool:
	"""Generate an SVG comparison figure without requiring third-party graphics libraries."""
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
	condition_data = {}

	for cond_name, display_label in condition_labels.items():
		sdata = load_condition_seeds(cond_name)
		if sdata:
			condition_data[cond_name] = (display_label, sdata)

	if not condition_data:
		return False

	width, height = 960, 560
	margin_left, margin_right = 90, 40
	margin_top, margin_bottom = 60, 70
	plot_w = width - margin_left - margin_right
	plot_h = height - margin_top - margin_bottom

	min_x, max_x = 0.0, 500000.0
	min_y, max_y = -100.0, 1000.0

	def to_svg_x(x_val: float) -> float:
		return margin_left + ((x_val - min_x) / (max_x - min_x)) * plot_w

	def to_svg_y(y_val: float) -> float:
		return margin_top + plot_h - ((y_val - min_y) / (max_y - min_y)) * plot_h

	svg_lines = [
		f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background-color:#ffffff; font-family:system-ui, -apple-system, sans-serif;">',
		f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
		f'<text x="{width/2}" y="{margin_top/2 + 5}" font-size="18" font-weight="bold" text-anchor="middle" fill="#111827">{title}</text>',
	]

	# Gridlines & Y-ticks
	y_ticks = [-100, 0, 200, 400, 500, 600, 800, 1000]
	for yt in y_ticks:
		sy = to_svg_y(yt)
		dash = 'stroke-dasharray="4,4"' if yt != 0 else ''
		stroke_col = "#374151" if yt == 0 else ("#10b981" if yt == 500 else "#e5e7eb")
		stroke_w = "1.5" if yt in [0, 500] else "1"
		svg_lines.append(f'<line x1="{margin_left}" y1="{sy}" x2="{margin_left + plot_w}" y2="{sy}" stroke="{stroke_col}" stroke-width="{stroke_w}" {dash}/>')
		svg_lines.append(f'<text x="{margin_left - 12}" y="{sy + 4}" font-size="12" text-anchor="end" fill="#6b7280">{yt}</text>')

	# X-ticks
	x_ticks = [0, 100000, 200000, 300000, 400000, 500000]
	for xt in x_ticks:
		sx = to_svg_x(xt)
		svg_lines.append(f'<line x1="{sx}" y1="{margin_top}" x2="{sx}" y2="{margin_top + plot_h}" stroke="#f3f4f6" stroke-width="1"/>')
		lbl = f"{int(xt/1000)}k" if xt > 0 else "0"
		svg_lines.append(f'<text x="{sx}" y="{margin_top + plot_h + 20}" font-size="12" text-anchor="middle" fill="#6b7280">{lbl}</text>')

	# Axis lines
	svg_lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#9ca3af" stroke-width="1.5"/>')
	svg_lines.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#9ca3af" stroke-width="1.5"/>')

	# Axis labels
	svg_lines.append(f'<text x="{margin_left + plot_w / 2}" y="{height - 18}" font-size="13" font-weight="600" text-anchor="middle" fill="#374151">Agent Timesteps</text>')
	svg_lines.append(f'<text transform="rotate(-90 28 {margin_top + plot_h/2})" x="28" y="{margin_top + plot_h/2}" font-size="13" font-weight="600" text-anchor="middle" fill="#374151">Evaluation Return (Mean over 5 Episodes)</text>')

	legend_items = []

	# Draw data curves for each condition
	for idx, (cond_name, (display_label, sdata)) in enumerate(condition_data.items()):
		color = colors[idx % len(colors)]
		legend_items.append((display_label, len(sdata), color))

		all_timesteps = []
		all_returns = []

		for seed, data in sdata.items():
			ts = data["timesteps"]
			rews = data["eval_mean_rew"]
			all_timesteps.append(ts)
			all_returns.append(rews)

			# Individual faint seed run
			pts = " ".join(f"{to_svg_x(t):.1f},{to_svg_y(r):.1f}" for t, r in zip(ts, rews))
			svg_lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.2" stroke-opacity="0.28" points="{pts}"/>')

		# Interpolate onto common grid for mean +/- std
		min_len = min(len(t) for t in all_timesteps)
		ref_grid = all_timesteps[0][:min_len]
		interpolated = []
		for ts, rews in zip(all_timesteps, all_returns):
			interpolated.append(np.interp(ref_grid, ts, rews))
		interp_arr = np.array(interpolated)
		mean_curve = interp_arr.mean(axis=0)
		std_curve = interp_arr.std(axis=0)

		# Std shading polygon
		upper_pts = [f"{to_svg_x(t):.1f},{to_svg_y(m + s):.1f}" for t, m, s in zip(ref_grid, mean_curve, std_curve)]
		lower_pts = [f"{to_svg_x(t):.1f},{to_svg_y(m - s):.1f}" for t, m, s in zip(reversed(ref_grid), reversed(mean_curve), reversed(std_curve))]
		band_pts = " ".join(upper_pts + lower_pts)
		svg_lines.append(f'<polygon points="{band_pts}" fill="{color}" fill-opacity="0.16"/>')

		# Bold mean line
		mean_pts = " ".join(f"{to_svg_x(t):.1f},{to_svg_y(m):.1f}" for t, m in zip(ref_grid, mean_curve))
		svg_lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.8" points="{mean_pts}"/>')

	# Legend Box in bottom right
	leg_x = margin_left + plot_w - 340
	leg_y = margin_top + plot_h - 130
	svg_lines.append(f'<rect x="{leg_x}" y="{leg_y}" width="330" height="115" rx="6" fill="#ffffff" fill-opacity="0.92" stroke="#d1d5db" stroke-width="1"/>')

	for i, (lbl, n_s, col) in enumerate(legend_items):
		iy = leg_y + 24 + i * 26
		svg_lines.append(f'<line x1="{leg_x + 16}" y1="{iy}" x2="{leg_x + 46}" y2="{iy}" stroke="{col}" stroke-width="3"/>')
		svg_lines.append(f'<text x="{leg_x + 56}" y="{iy + 4}" font-size="12" font-weight="500" fill="#1f2937">{lbl} (N={n_s})</text>')

	# Legend entry for solved threshold
	thresh_y = leg_y + 24 + len(legend_items) * 26
	svg_lines.append(f'<line x1="{leg_x + 16}" y1="{thresh_y}" x2="{leg_x + 46}" y2="{thresh_y}" stroke="#10b981" stroke-width="1.8" stroke-dasharray="4,3"/>')
	svg_lines.append(f'<text x="{leg_x + 56}" y="{thresh_y + 4}" font-size="12" fill="#047857">Solved Threshold (500.0)</text>')

	svg_lines.append("</svg>")

	out_path = RESULTS_DIR / output_filename
	with open(out_path, mode="w", encoding="utf-8") as f:
		f.write("\n".join(svg_lines))
	print(f"Saved SVG comparison figure: {out_path}")
	return True


def plot_comparison(
	condition_labels: dict[str, str],
	title: str,
	output_filename: str,
	common_step_limit: int | None = None,
) -> bool:
	"""Generate a multi-seed comparison figure with individual seed traces and shaded +/- 1 std."""
	if plt is None:
		return False

	plt.figure(figsize=(9.5, 6))
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
	plotted_any = False

	for idx, (cond_name, display_label) in enumerate(condition_labels.items()):
		seed_data = load_condition_seeds(cond_name)
		if not seed_data:
			continue

		plotted_any = True
		color = colors[idx % len(colors)]

		all_timesteps = []
		all_returns = []

		for seed, data in seed_data.items():
			ts = data["timesteps"]
			rews = data["eval_mean_rew"]
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
		plt.plot(
			ref_grid,
			mean_curve,
			color=color,
			label=f"{display_label} (N={len(seed_data)})",
			linewidth=2.4,
		)
		plt.fill_between(
			ref_grid,
			mean_curve - std_curve,
			mean_curve + std_curve,
			color=color,
			alpha=0.18,
		)

	if not plotted_any:
		plt.close()
		return False

	plt.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
	plt.axhline(PERF_THRESHOLD_500, color="green", linestyle="--", linewidth=0.9, alpha=0.6, label="Solved Threshold (500.0)")
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
	return True


def export_summary_tables(summaries: list[dict[str, Any]]) -> None:
	"""Save aggregate results to CSV, Markdown, and print formatted table to console."""
	if not summaries:
		print("No valid conditions found to summarize.")
		return

	# 1. Print formatted console output
	print("\n" + "=" * 90)
	print("MULTI-SEED PPO AGGREGATE RESULTS SUMMARY (CarRacing-v3)")
	print("=" * 90)
	header_fmt = "{:<16} | {:<5} | {:<16} | {:<16} | {:<16} | {:<10} | {:<10}"
	print(header_fmt.format("Condition", "Seeds", "Final Return", "Peak Return", "Norm AUC", "500 Reached", "Time (hrs)"))
	print("-" * 90)
	for s in summaries:
		final_str = f"{s['final_mean']:.1f} +/- {s['final_std']:.1f}"
		best_str = f"{s['best_mean']:.1f} +/- {s['best_std']:.1f}"
		auc_str = f"{s['auc_mean']:.1f} +/- {s['auc_std']:.1f}"
		time_str = f"{s['time_hrs_mean']:.2f}h +/- {s['time_hrs_std']:.2f}"
		print(header_fmt.format(
			s["condition"],
			f"{s['n_seeds']}",
			final_str,
			best_str,
			auc_str,
			s["reached_500_ratio"],
			time_str,
		))
	print("=" * 90 + "\n")

	# Detailed per-seed breakdown
	print("PER-SEED BREAKDOWN:")
	for s in summaries:
		print(f"\nCondition: {s['condition']} ({s['label']})")
		for seed, p in sorted(s["per_seed"].items()):
			s500_txt = f"{int(p['steps_500'])}" if p["steps_500"] is not None else "N/A"
			print(
				f"  Seed {seed:2d}: Final = {p['final_rew']:6.1f} | "
				f"Peak = {p['best_rew']:6.1f} | AUC = {p['auc']:6.1f} | "
				f"Steps->500 = {s500_txt:>7} | Time = {p['time_hrs']:.2f}h | Final Len = {p['final_len']:.1f}"
			)

	# 2. Save CSV summary
	csv_path = RESULTS_DIR / "summary_metrics.csv"
	csv_fields = [
		"condition",
		"label",
		"n_seeds",
		"final_return_mean",
		"final_return_std",
		"peak_return_mean",
		"peak_return_std",
		"normalized_auc_mean",
		"normalized_auc_std",
		"reached_500_ratio",
		"steps_to_500_mean",
		"reached_900_ratio",
		"training_time_hrs_mean",
		"training_time_hrs_std",
		"final_eval_len_mean",
	]
	with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=csv_fields)
		writer.writeheader()
		for s in summaries:
			writer.writerow({
				"condition": s["condition"],
				"label": s["label"],
				"n_seeds": s["n_seeds"],
				"final_return_mean": round(s["final_mean"], 2),
				"final_return_std": round(s["final_std"], 2),
				"peak_return_mean": round(s["best_mean"], 2),
				"peak_return_std": round(s["best_std"], 2),
				"normalized_auc_mean": round(s["auc_mean"], 2),
				"normalized_auc_std": round(s["auc_std"], 2),
				"reached_500_ratio": s["reached_500_ratio"],
				"steps_to_500_mean": round(s["steps_500_avg"], 1) if s["steps_500_avg"] else "N/A",
				"reached_900_ratio": s["reached_900_ratio"],
				"training_time_hrs_mean": round(s["time_hrs_mean"], 2),
				"training_time_hrs_std": round(s["time_hrs_std"], 2),
				"final_eval_len_mean": round(s["final_len_mean"], 2),
			})
	print(f"\nSaved summary CSV: {csv_path}")

	# 3. Save Markdown summary table
	md_path = RESULTS_DIR / "summary_metrics.md"
	md_lines = [
		"# Experiment Summary Metrics: CarRacing-v3",
		"",
		"| Condition | Seeds | Final Return (Mean ± Std) | Peak Return (Mean ± Std) | Normalized AUC (Mean ± Std) | Steps to 500 (Avg) | 500 Reached | Wall-Clock Time (hrs) | Final Ep Length |",
		"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
	]
	for s in summaries:
		s500_str = f"{s['steps_500_avg']:,.0f}" if s["steps_500_avg"] is not None else "N/A"
		md_lines.append(
			f"| **{s['condition']}** (`{s['label']}`) | {s['n_seeds']} | "
			f"{s['final_mean']:.1f} ± {s['final_std']:.1f} | "
			f"{s['best_mean']:.1f} ± {s['best_std']:.1f} | "
			f"{s['auc_mean']:.1f} ± {s['auc_std']:.1f} | "
			f"{s500_str} | {s['reached_500_ratio']} | "
			f"{s['time_hrs_mean']:.2f}h ± {s['time_hrs_std']:.2f}h | "
			f"{s['final_len_mean']:.1f} |"
		)
	md_lines.append("")
	with open(md_path, mode="w", encoding="utf-8") as f:
		f.write("\n".join(md_lines))
	print(f"Saved summary Markdown: {md_path}")


def main() -> None:
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)

	# 1. Primary Comparison: Baseline vs. Off-Road Termination Wrapper
	print("Analyzing Baseline vs. Off-Road Termination...")
	comparison_conditions = {
		"baseline": "Baseline (Standard Termination)",
		"off_road_term": "Off-Road Termination Wrapper",
	}

	# Vector figure (no matplotlib dependency)
	plot_comparison_svg(
		comparison_conditions,
		"CarRacing-v3 PPO: Baseline vs. Off-Road Early Termination",
		"fig_baseline_vs_off_road.svg",
	)

	# Publication PNG figures (when matplotlib is available)
	plot_comparison(
		comparison_conditions,
		"CarRacing-v3 PPO: Baseline vs. Off-Road Early Termination",
		"fig_baseline_vs_off_road.png",
	)
	plot_comparison(
		comparison_conditions,
		"CarRacing-v3 PPO: Baseline vs. Off-Road Early Termination",
		"baseline_vs_off_road.png",
	)

	# Optional additional comparisons if present
	plot_comparison(
		{"adv_rtg": "Rewards-To-Go (No GAE)", "adv_gae": "GAE (lambda=0.95)"},
		"Advantage Estimation on CarRacing-v3: Rewards-To-Go vs. GAE",
		"fig1_advantage_estimation.png",
	)
	plot_comparison(
		{"clip_0.1": "Clip epsilon = 0.1", "clip_0.2": "Clip epsilon = 0.2", "clip_0.3": "Clip epsilon = 0.3"},
		"PPO Clipping Parameter Sensitivity on CarRacing-v3",
		"fig2_ppo_clipping.png",
	)
	plot_comparison(
		{"ent_0.0": "Entropy coef = 0.0", "ent_0.01": "Entropy coef = 0.01"},
		"Entropy Regularization Exploration on CarRacing-v3",
		"fig3_entropy_regularization.png",
	)

	# Aggregate statistics for all detected conditions
	summaries = []
	all_conditions_to_check = [
		("baseline", "Baseline (Standard Termination)"),
		("off_road_term", "Off-Road Termination Wrapper"),
		("adv_gae", "GAE (lambda=0.95)"),
		("adv_rtg", "Rewards-To-Go (No GAE)"),
		("clip_0.1", "Clip epsilon = 0.1"),
		("clip_0.2", "Clip epsilon = 0.2"),
		("clip_0.3", "Clip epsilon = 0.3"),
		("ent_0.0", "Entropy coef = 0.0"),
		("ent_0.01", "Entropy coef = 0.01"),
	]

	seen_conditions = set()
	for cond_name, label in all_conditions_to_check:
		if cond_name in seen_conditions:
			continue
		seed_data = load_condition_seeds(cond_name)
		if seed_data:
			summary = summarize_condition(cond_name, label, seed_data)
			if summary:
				summaries.append(summary)
				seen_conditions.add(cond_name)

	# Check for any other condition folders present in results/
	for cond_path in sorted(RESULTS_DIR.iterdir()):
		if cond_path.is_dir() and cond_path.name not in seen_conditions and cond_path.name != "smoke_test":
			seed_data = load_condition_seeds(cond_path.name)
			if seed_data:
				summary = summarize_condition(cond_path.name, cond_path.name, seed_data)
				if summary:
					summaries.append(summary)
					seen_conditions.add(cond_path.name)

	export_summary_tables(summaries)


if __name__ == "__main__":
	main()
