import gym
from gym.wrappers import TimeLimit
import os

from rlkit.envs.wrappers import CustomInfoEnv, NormalizedBoxEnv
from ae.wrapper import AutoencoderWrapper


def make_env(name, use_ae=False, ae_path=None):
    env = gym.make(name)
    # Remove TimeLimit Wrapper
    if isinstance(env, TimeLimit):
        env = env.unwrapped
    env = CustomInfoEnv(env)
    env = NormalizedBoxEnv(env)
    
    # Add autoencoder wrapper if requested
    if use_ae:
        if ae_path is None:
            # You can set a default path or use environment variable
            ae_path = os.environ.get("AAE_PATH")
        env = AutoencoderWrapper(env, ae_path=ae_path)
    
    return env
