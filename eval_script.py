import pickle
import torch
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
import gym
import gym_donkeycar
import numpy as np
import os
import argparse
from simple_ae_wrapper import SimpleAEWrapper

def main():
    parser = argparse.ArgumentParser(description='Evaluate DonkeyCar Policy')
    parser.add_argument('--checkpoint', type=str, default='./eval_models/CurriculumParams.pkl',
                      help='Path to the checkpoint file')
    parser.add_argument('--ae-path', type=str, 
                      default='./logs/ae-32_1745884521_best.pkl',
                      help='Path to the autoencoder checkpoint')
    parser.add_argument('--max-steps', type=int, default=4000,
                      help='Maximum number of steps to run')
    args = parser.parse_args()

    # Load policy state
    print(f"Loading checkpoint from {args.checkpoint}")
    with open(args.checkpoint, 'rb') as f:
        state_dict = torch.load(f)

    # Setup DonkeyCar environment
    sim_params = {
        "DONKEY_SIM_PATH": os.environ.get("DONKEY_SIM_PATH", "remote"),
        "SIM_HOST": "localhost",
        "SIM_PORT": 9091,
        "HEADLESS": False,
        "MAX_CTE_ERROR": 10.0
    }
    
    env = gym.make("donkey-generated-roads-v0", conf=sim_params)
    from rlkit.envs.wrappers import CustomInfoEnv, NormalizedBoxEnv
    env = CustomInfoEnv(env)
    env = NormalizedBoxEnv(env)
    
    # Add autoencoder wrapper
    print(f"Using autoencoder from {args.ae_path}")
    env = SimpleAEWrapper(env, args.ae_path)

    # Get environment dimensions after all wrappers
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    print(f"Final observation dim: {obs_dim}, action dim: {action_dim}")

    # Create policy with correct dimensions
    target_policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[256, 256, 256, 256, 256],
        dropout_probability=0.1,
    )

    # Load trained weights
    target_policy.load_state_dict(state_dict["trainer/policy"])
    # Make policy deterministic for evaluation
    eval_policy = MakeDeterministic(target_policy)

    # Run evaluation
    obs = env.reset()
    return_ep = 0
    step_count = 0
    done = False

    print("Starting evaluation...")
    while not done and step_count < args.max_steps:
        # Get deterministic action
        action, _ = eval_policy.get_action(obs)  # Unpack the tuple
        
        # Take step in environment
        obs, reward, done, info = env.step(action)
        return_ep += reward
        step_count += 1
        
        # Print progress
        if step_count % 10 == 0:
            print(f"Step {step_count}: Reward = {reward:.2f}, CTE = {info.get('cte', 'N/A')}")
        
        env.render()

    print("\nEvaluation Results:")
    print(f"Total Steps: {step_count}")
    print(f"Total Return: {return_ep:.2f}")
    
    env.close()

if __name__ == "__main__":
    main()


