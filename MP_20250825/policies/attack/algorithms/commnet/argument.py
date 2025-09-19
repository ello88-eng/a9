from dataclasses import dataclass

from environmentComm import Environment


@dataclass
class MainArguments:
    seed: int = 123
    mode: str = "CTDE"  # CTDE, Random, IAC, DQN
    batch_size: int = 32  # 32, 256
    actor_dim: int = 64
    critic_dim: int = 256
    train_epoch: int = 6500
    cuda: bool = True
    gamma: float = 0.98
    actor_lr: float = 1e-2  # 1e-2
    critic_lr: float = 1e-3  # 1e-3,
    epsilon: float = 0.275  # 0.8, # 1
    inference_epoch: int = 100
    anneal_epsilon: float = 0.00005  # 0.00005, # 0.001
    min_epsilon: float = 0.01
    # 'td_lambda' : 0.02, # 0.8
    replay_capacity: int = 50000  # 10000
    target_update_cycle: int = 20
    grad_norm_clip: int = 10
    k: int = 2
    train_size: int = 32  # 2000,


def get_args():
    return MainArguments()


def get_env_info(args: MainArguments, env: Environment):
    env_info = env.get_info()
    args.episode_limit = env_info["episode_limit"]
    args.n_Comm_agents = env_info["n_Comm_agents"]
    args.n_DQN_agents = env_info["n_DQN_agents"]
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    # args.state_dim     = env_info["state_dim"]
    args.obs_dim = env_info["obs_dim"]
    # # o = np.hstack([o_idx, o_partial])
    return args
