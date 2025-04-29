import argparse
import os

import torch

import rlkit.torch.pytorch_util as ptu
import yaml
from rlkit.data_management.torch_replay_buffer import TorchReplayBuffer
from rlkit.envs import make_env
from rlkit.envs.vecenv import SubprocVectorEnv, VectorEnv, SharedCarVectorEnv
from rlkit.launchers.launcher_util import set_seed, setup_logger
from rlkit.samplers.data_collector import (VecMdpPathCollector, VecMdpStepCollector)
from rlkit.torch.qrsac.qrsac import QRSACTrainer
from rlkit.torch.qrsac.networks import QuantileMlp, softmax
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.torch_rl_algorithm import TorchVecOnlineRLAlgorithm

from gym.envs.registration import register
import gym
import gym_donkeycar

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

# Store the single donkey car environment globally
_donkey_env = None

def get_singleton_donkey_env(variant):
    """Create or return the single DonkeyCar environment instance"""
    global _donkey_env
    
    if _donkey_env is None:
        env_name = variant['env']
        
        # Configure DonkeyCar to use one simulator instance
        sim_params = {
            "DONKEY_SIM_PATH": os.environ.get("DONKEY_SIM_PATH", "remote"),
            "SIM_HOST": "localhost",
            "SIM_PORT": 9091,  # Use a single port
            "HEADLESS": False,
            "MAX_CTE_ERROR": 10.0
        }
        
        # Create only one environment
        env = gym.make(env_name, conf=sim_params)
        
        # Apply standard wrappers
        from rlkit.envs.wrappers import CustomInfoEnv, NormalizedBoxEnv
        env = CustomInfoEnv(env)
        env = NormalizedBoxEnv(env)
        
        # Apply autoencoder wrapper if needed
        use_ae = variant.get('use_ae', False)
        ae_path = variant.get('ae_path', None)
        if use_ae and ae_path:
            from simple_ae_wrapper import SimpleAEWrapper
            env = SimpleAEWrapper(env, ae_path)
        
        _donkey_env = env
    
    return _donkey_env

# Create a proxy environment that delegates to the singleton
class SingletonProxyEnv(gym.Wrapper):
    def __init__(self, env_getter):
        self.env_getter = env_getter
        env = self.env_getter()  # Get the environment
        super().__init__(env)
        
    def reset(self):
        return self.env_getter().reset()
        
    def step(self, action):
        return self.env_getter().step(action)
        
    def render(self):
        return self.env_getter().render()

def get_dummy_env(env, use_ae=False, ae_path=None):
    if use_ae and ae_path is not None:
        from ae.wrapper import AutoencoderWrapper
        return AutoencoderWrapper(env, ae_path=ae_path)
    return env

def experiment(variant):
    # Create the singleton environment
    singleton_getter = lambda: get_singleton_donkey_env(variant)
    
    # Use proxy environments that all delegate to the singleton
    dummy_env = singleton_getter()
    
    # Create vector envs that use proxies to the same underlying environment
    expl_env = VectorEnv([lambda: SingletonProxyEnv(singleton_getter) 
                          for _ in range(variant['expl_env_num'])])
    
    eval_env = VectorEnv([lambda: SingletonProxyEnv(singleton_getter) 
                          for _ in range(variant['eval_env_num'])])
    
    # The rest of your QRSAC setup remains unchanged
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size
    print(f"obs dim, action dim={obs_dim}, {action_dim}")
    
    # Continue with network architecture and training setup...

    M = variant['layer_size']
    num_quantiles = variant['num_quantiles']

    zf1 = QuantileMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=[M, M, M, M, M],
    )
    zf2 = QuantileMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=[M, M, M, M, M],
    )
    target_zf1 = QuantileMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=[M, M, M, M, M],
    )
    target_zf2 = QuantileMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=[M, M, M, M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M, M, M],
        dropout_probability = 0.1,
    )
    eval_policy = MakeDeterministic(policy)
    target_policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M, M, M],
        dropout_probability=0.1,
    )
    # fraction proposal network
    fp = target_fp = None
    if variant['trainer_kwargs'].get('tau_type') == 'fqf':
        fp = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=num_quantiles,
            hidden_sizes=[M // 2, M // 2],
            output_activation=softmax,
        )
        target_fp = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=num_quantiles,
            hidden_sizes=[M // 2, M // 2],
            output_activation=softmax,
        )
    eval_path_collector = VecMdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = VecMdpStepCollector(
        expl_env,
        policy,
    )
    replay_buffer = TorchReplayBuffer(
        variant['replay_buffer_size'],
        dummy_env,
    )
    trainer = QRSACTrainer(
        env=dummy_env,
        policy=policy,
        target_policy=target_policy,
        zf1=zf1,
        zf2=zf2,
        target_zf1=target_zf1,
        target_zf2=target_zf2,
        fp=fp,
        target_fp=target_fp,
        num_quantiles=num_quantiles,
        **variant['trainer_kwargs'],
    )
    algorithm = TorchVecOnlineRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs'],
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantile-Regression Soft Actor Critic')
    parser.add_argument('--config', type=str, default="configs/lunarlander.yaml")
    parser.add_argument('--gpu', type=int, default=0, help="using cpu with -1")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    with open(args.config, 'r', encoding="utf-8") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    variant["seed"] = 100 #args.seed
    log_prefix = "_".join(["qrsac", variant["env"][:-3].lower(), str(variant["version"])])
    setup_logger(log_prefix, variant=variant, seed=args.seed)
    if args.gpu >= 0:
        ptu.set_gpu_mode(True, args.gpu)
    set_seed(args.seed)

    #JETRACER
    # ctrl = XboxController()
    # print("TELEOPERATION READY")
    #
    # while True:
    #     _, _, A, _ = ctrl.read()
    #     if A == 1: break
    # while True:
    #
    #     steering, throttle, _, B = ctrl.read()
    #     if B == 1: break
    #     if throttle > 0.7:
    #         throttle = 0.7
    #     # t1 = threading.Thread(target=write_image_to_disk, args=(obs, i, steering, throttle, pose, act_time))
        # t1.start()

    experiment(variant)
