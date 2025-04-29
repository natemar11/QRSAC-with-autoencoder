import gym
import numpy as np
from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe

from rlkit.envs.env_utils import CloudpickleWrapper
#donkeycar
from ae.autoencoder import Autoencoder
from ae.autoencoder import load_ae


class BaseVectorEnv(ABC, gym.Wrapper):
    """Base class for vectorized environments wrapper. Usage:
    ::

        env_num = 8
        envs = VectorEnv([lambda: gym.make(task) for _ in range(env_num)])
        assert len(envs) == env_num

    It accepts a list of environment generators. In other words, an environment
    generator ``efn`` of a specific task means that ``efn()`` returns the
    environment of the given task, for example, ``gym.make(task)``.

    All of the VectorEnv must inherit :class:`~tianshou.env.BaseVectorEnv`.
    Here are some other usages:
    ::

        envs.seed(2)  # which is equal to the next line
        envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # set specific seed for each env
        obs = envs.reset()  # reset all environments
        obs = envs.reset([0, 5, 7])  # reset 3 specific environments
        obs, rew, done, info = envs.step([1] * 8)  # step synchronously
        envs.render()  # render all environments
        envs.close()  # close all environments
    """

    def __init__(self, env_fns):
        self._env_fns = env_fns
        self.env_num = len(env_fns)

    def __len__(self):
        """Return len(self), which is the number of environments."""
        return self.env_num

    @abstractmethod
    def reset(self, id=None):
        """Reset the state of all the environments and return initial
        observations if id is ``None``, otherwise reset the specific
        environments with given id, either an int or a list.
        """
        pass

    @abstractmethod
    def step(self, action):
        """Run one timestep of all the environments' dynamics. When the end of
        episode is reached, you are responsible for calling reset(id) to reset
        this environment's state.

        Accept a batch of action and return a tuple (obs, rew, done, info).

        :param numpy.ndarray action: a batch of action provided by the agent.

        :return: A tuple including four items:

            * ``obs`` a numpy.ndarray, the agent's observation of current \
                environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``done`` a numpy.ndarray, whether these episodes have ended, in \
                which case further step() calls will return undefined results
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)
        """
        pass

    @abstractmethod
    def seed(self, seed=None):
        """Set the seed for all environments. Accept ``None``, an int (which
        will extend ``i`` to ``[i, i + 1, i + 2, ...]``) or a list.
        """
        pass

    @abstractmethod
    def render(self, **kwargs):
        """Render all of the environments."""
        pass

    @abstractmethod
    def close(self):
        """Close all of the environments."""
        pass


class VectorEnv(BaseVectorEnv):
    """Dummy vectorized environment wrapper, implemented in for-loop.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.envs = [_() for _ in env_fns]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        #donkeycar
        # ae_path= "/mnt/c/Users/natha/OneDrive/Documents/Cursor/QRSAC/logs/ae-32_1745874108_best.pkl"
        # self.ae = load_ae(ae_path)
    def reset(self, id=None):
        if id is None:
        
            for e in self.envs:
                reset_env = e.reset()
                self._obs = np.stack([reset_env])

        else:
            if np.isscalar(id):
                id = [id]
            for i in id:
               
                self._obs[i] = self.envs[i].reset()
                # donkeycar 
                # self._obs[i] = self.ae.encode_from_raw_image(self.envs[i].reset()[:,:,::-1])
                # #donkeycar if additional obs data appended
                # self._obs[i] = np.concatenate((self.ae.encode_from_raw_image(self.envs[i].reset()[:, :, ::-1]), np.zeros(7).reshape(1, -1)), axis=1) #if 7 mo

        return self._obs

    def step(self, action):
        assert len(action) == self.env_num
        result = [e.step(a) for e, a in zip(self.envs, action)]
        self._obs, self._rew, self._done, self._info = zip(*result)
        
        #donkeycar
        # self._obs = self.ae.encode_from_raw_image(np.squeeze(np.array(self._obs)[:,:,::-1]))
        ##donkeycar if additional obs data appended
        # angle = self._info[0]['angle']
        # cte = self._info[0]['cte']
        # vel = self._info[0]['vel']
        # additional_values =  [cte] + list(angle) + list(vel)
        # additional_values_array = np.array(additional_values)
        # self._obs = np.concatenate((self._obs, additional_values_array.reshape(1, -1)), axis=1)

        self._obs = np.stack(self._obs)
        self._rew = np.stack(self._rew)
        self._done = np.stack(self._done)
        self._info = np.stack(self._info)
        return self._obs, self._rew, self._done, self._info

    def seed(self, seed=None):
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        result = []
        for e, s in zip(self.envs, seed):
            if hasattr(e, 'seed'):
                result.append(e.seed(s))
        return result

    def render(self, **kwargs):
        result = []
        for e in self.envs:
            if hasattr(e, 'render'):
                result.append(e.render(**kwargs))
        return result

    def close(self):
        return [e.close() for e in self.envs]


def worker(parent, p, env_fn_wrapper):
    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            cmd, data = p.recv()
            if cmd == 'step':
                p.send(env.step(data))
            elif cmd == 'reset':
                p.send(env.reset())
            elif cmd == 'close':
                p.send(env.close())
                p.close()
                break
            elif cmd == 'render':
                p.send(env.render(**data) if hasattr(env, 'render') else None)
            elif cmd == 'seed':
                p.send(env.seed(data) if hasattr(env, 'seed') else None)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on subprocess.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self, env_fns):
        dummy_env = env_fns[0]()
        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space
        super().__init__(env_fns)
        self.closed = False
        self.parent_remote, self.child_remote = \
            zip(*[Pipe() for _ in range(self.env_num)])
        self.processes = [
            Process(target=worker, args=(parent, child, CloudpickleWrapper(env_fn)), daemon=True)
            for (parent, child, env_fn) in zip(self.parent_remote, self.child_remote, env_fns)
        ]
        for p in self.processes:
            p.start()
        for c in self.child_remote:
            c.close()

    def step(self, action):
        assert len(action) == self.env_num
        for p, a in zip(self.parent_remote, action):
            p.send(['step', a])
        result = [p.recv() for p in self.parent_remote]
        self._obs, self._rew, self._done, self._info = zip(*result)
        self._obs = np.stack(self._obs)
        self._rew = np.stack(self._rew)
        self._done = np.stack(self._done)
        self._info = np.stack(self._info)
        return self._obs, self._rew, self._done, self._info

    def reset(self, id=None):
        if id is None:
            for p in self.parent_remote:
                p.send(['reset', None])
            self._obs = np.stack([p.recv() for p in self.parent_remote])
            return self._obs
        else:
            if np.isscalar(id):
                id = [id]
            for i in id:
                self.parent_remote[i].send(['reset', None])
            for i in id:
                self._obs[i] = self.parent_remote[i].recv()
            return self._obs

    def seed(self, seed=None):
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        for p, s in zip(self.parent_remote, seed):
            p.send(['seed', s])
        return [p.recv() for p in self.parent_remote]

    def render(self, **kwargs):
        for p in self.parent_remote:
            p.send(['render', kwargs])
        return [p.recv() for p in self.parent_remote]

    def close(self):
        if self.closed:
            return
        for p in self.parent_remote:
            p.send(['close', None])
        result = [p.recv() for p in self.parent_remote]
        self.closed = True
        for p in self.processes:
            p.join()
        return result

# Create a modified VectorEnv class that shares a single environment
class SharedCarVectorEnv(BaseVectorEnv):
    """Vector environment that uses the same underlying car for all instances"""
    
    def __init__(self, env_fn, num_envs):
        super().__init__([env_fn] * num_envs)  # Still create the expected structure
        # But only instantiate one actual environment
        self.shared_env = env_fn()
        self.observation_space = self.shared_env.observation_space
        self.action_space = self.shared_env.action_space
        self.num_envs = num_envs
        
    def reset(self, id=None):
        # Reset the shared environment only once
        obs = self.shared_env.reset()
        # Return the same observation for all environments
        self._obs = np.stack([obs] * self.num_envs)
        return self._obs
        
    def step(self, actions):
        # Take only the first action (or average them if you prefer)
        action = actions[0]  # Use first environment's action
        # Or use average: action = np.mean(actions, axis=0)
        
        # Step the shared environment once
        obs, reward, done, info = self.shared_env.step(action)
        
        # Return the same result for all environments
        self._obs = np.stack([obs] * self.num_envs)
        self._rewards = np.stack([reward] * self.num_envs)
        self._dones = np.stack([done] * self.num_envs)
        self._infos = np.stack([info] * self.num_envs)
        
        return self._obs, self._rewards, self._dones, self._infos
        
    def render(self, **kwargs):
        return self.shared_env.render(**kwargs)
        
    def seed(self, seed=None):
        if seed is not None:
            return self.shared_env.seed(seed)
        return None
        
    def close(self):
        return self.shared_env.close()
