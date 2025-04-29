import os
import gym
import numpy as np
from ae.autoencoder import load_ae
from gym.spaces import Box

class SimpleAEWrapper(gym.Wrapper):
    """
    Simplified autoencoder wrapper that handles all observation dimensions consistently
    """
    def __init__(self, env, ae_path):
        super().__init__(env)
        self.ae = load_ae(ae_path) if ae_path else None
        
        # Update observation space to match encoded dimension
        encoded_dim = self.ae.z_size if self.ae else env.observation_space.shape[0]
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(encoded_dim,), dtype=np.float32
        )
        
        # Cache first observation to validate encoding works
        dummy_obs = env.reset()
        self._encode_obs(dummy_obs)
        print(f"SimpleAEWrapper: observation space shape = {self.observation_space.shape}")
    
    def _encode_obs(self, obs):
        """Encode observation consistently regardless of input shape"""
        if self.ae is None:
            return obs
            
        # Handle different input shapes (1D or 3D)
        if len(obs.shape) == 3:  # Image observation
            return self.ae.encode_from_raw_image(obs[:,:,::-1]).flatten()
        elif len(obs.shape) == 1:  # Already flattened
            return obs
        else:
            raise ValueError(f"Unsupported observation shape: {obs.shape}")
    
    def reset(self):
        obs = self.env.reset()
        return self._encode_obs(obs)
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._encode_obs(obs), reward, done, info
