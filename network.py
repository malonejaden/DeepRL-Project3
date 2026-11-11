"""Neural network architectures for PPO (MLP and CNN)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
	"""Apply orthogonal initialization with a constant bias."""
	if isinstance(layer, (nn.Linear, nn.Conv2d)):
		nn.init.orthogonal_(layer.weight, std)
		if layer.bias is not None:
			nn.init.constant_(layer.bias, bias_const)
	return layer


class FeedForwardNN(nn.Module):
	"""Standard 3-layer MLP for vector observations."""

	def __init__(self, in_dim: int, out_dim: int, output_gain: float = 1.0) -> None:
		super().__init__()
		self.layer1 = layer_init(nn.Linear(in_dim, 64))
		self.layer2 = layer_init(nn.Linear(64, 64))
		self.layer3 = layer_init(nn.Linear(64, out_dim), std=output_gain)

	def forward(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
		device = next(self.parameters()).device
		if isinstance(obs, np.ndarray):
			obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
		elif obs.device != device:
			obs = obs.to(device)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		return self.layer3(activation2)


class CNNNetwork(nn.Module):
	"""Nature-style CNN backbone with fully connected head for image observations."""

	def __init__(
		self,
		in_channels: int,
		out_dim: int,
		img_size: int = 84,
		output_gain: float = 1.0,
	) -> None:
		super().__init__()
		self.in_channels = in_channels
		self.out_dim = out_dim

		self.conv = nn.Sequential(
			layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
			nn.ReLU(),
			layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
			nn.ReLU(),
			layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
			nn.ReLU(),
			nn.Flatten(),
		)

		# Compute flattened conv dimension dynamically
		with torch.no_grad():
			dummy = torch.zeros(1, in_channels, img_size, img_size)
			conv_out_dim = self.conv(dummy).shape[1]

		self.fc = nn.Sequential(
			layer_init(nn.Linear(conv_out_dim, 512)),
			nn.ReLU(),
			layer_init(nn.Linear(512, out_dim), std=output_gain),
		)

	def forward(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
		device = next(self.parameters()).device
		if isinstance(obs, np.ndarray):
			obs = torch.as_tensor(obs, device=device)
		elif obs.device != device:
			obs = obs.to(device)

		unbatched = (obs.ndim == 3)
		if unbatched:
			obs = obs.unsqueeze(0)

		# Scale uint8 pixel inputs [0, 255] to float32 [0.0, 1.0]
		if obs.dtype == torch.uint8:
			obs = obs.float() / 255.0
		elif obs.dtype != torch.float32:
			obs = obs.float()

		features = self.conv(obs)
		out = self.fc(features)

		if unbatched:
			out = out.squeeze(0)
		return out
