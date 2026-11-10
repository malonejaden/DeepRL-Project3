import gymnasium as gym
import numpy as np
import torch
import sys

from arguments import get_args
from ppo import PPO
from network import FeedForwardNN


def train(env, hyperparameters, actor_model, critic_model):
	"""
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""	
	print(f"Training", flush=True)

	# Create a model for PPO.
	model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

	# Tries to load in an existing actor/critic model to continue training on
	if actor_model != '' and critic_model != '':
		print(f"Loading in {actor_model} and {critic_model}...", flush=True)
		model.actor.load_state_dict(torch.load(actor_model))
		model.critic.load_state_dict(torch.load(critic_model))
		print(f"Successfully loaded.", flush=True)
	elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
		print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
		sys.exit(0)
	else:
		print(f"Training from scratch.", flush=True)


	model.learn(total_timesteps=200_000_000)



def main(args):

    hyperparameters = {
		'timesteps_per_batch' : 2048,
		'max_timesteps_per_episode' : 200,
		'gamma' : .99,
		'n_updates_per_iteration': 10,
        'lr': 3e-4, 
        'clip': 0.2,
        'render': True,
        'render_every_i': 10
    }

    env = gym.make("CarRacing-v3", render_mode="human", continuous=False)


    if args.mode == 'train':
	    train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    # else:
	#     test(env=env, actor_model=args.actor_model)

if __name__ == '__main__':
	args = get_args() # Parse arguments from command line
	main(args)
    