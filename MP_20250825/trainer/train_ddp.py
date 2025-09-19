import argparse
import os
import threading
import sys
import pathlib
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from actor_critic import ActorCritic
from env import UnrealGymEnv
from replay_buffer import ReplayBuffer
from torch.nn.parallel import DistributedDataParallel as DDP
import time

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from commu.sim.sender.sim_sender import send_exit
from utils.logger import logger


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    input_dim = 40000 + 7 + 80  # Scenario (40000) + Agent (7) + Target (80)
    output_dim = 2
    agent = ActorCritic(input_dim, output_dim)
    agent.actor = DDP(agent.actor, device_ids=[rank])
    agent.critic = DDP(agent.critic, device_ids=[rank])
    replay_buffer = ReplayBuffer(capacity=5000)
    envs = [UnrealGymEnv(udp_port=2929 + (i + 1) * 10000) for i in range(args.num_envs)]

    def run_environment(
        env: UnrealGymEnv,
        agent: ActorCritic,
        replay_buffer: ReplayBuffer,
        env_id: int,
        max_episode_len: int = 1000,
        total_episodes: int = 100,
    ):
        for episode in range(total_episodes):
            observation, info = env.reset()
            done = False
            step_count = 0
            while not done and step_count < max_episode_len:
                if "state" not in observation:
                    logger.info(f"Error: 'state' key not in observation: {observation}")
                    break
                state = observation["state"]
                scenario = state["scenario"]
                for agent_id, agent_data in state["agent"].items():
                    agent_state = np.concatenate(
                        [agent_data.pos, agent_data.attitude, [agent_data.duration]]
                    )
                    target_state_list = []
                    for target_id, target_data in state["target"].items():
                        target_state = np.concatenate(
                            [target_data.pos, [target_data.value]]
                        )
                        target_state_list.append(target_state)
                    target_state_array = np.concatenate(target_state_list)
                    obs_flat = np.concatenate(
                        [scenario, agent_state, target_state_array]
                    )
                    action = agent.select_action(obs_flat)
                    next_observation, reward, terminated, truncated, info = env.step(
                        action
                    )
                    if "state" not in next_observation:
                        logger.info(
                            f"Error: 'state' key not in next_observation: {next_observation}"
                        )
                        break
                    next_state = next_observation["state"]
                    next_agent_state = np.concatenate(
                        [
                            next_state["agent"][agent_id].pos,
                            next_state["agent"][agent_id].attitude,
                            [next_state["agent"][agent_id].duration],
                        ]
                    )
                    next_target_state_list = []
                    for target_id, next_target_data in next_state["target"].items():
                        next_target_state = np.concatenate(
                            [next_target_data.pos, [next_target_data.value]]
                        )
                        next_target_state_list.append(next_target_state)
                    next_target_state_array = np.concatenate(next_target_state_list)
                    next_obs_flat = np.concatenate(
                        [
                            next_state["scenario"],
                            next_agent_state,
                            next_target_state_array,
                        ]
                    )
                    replay_buffer.add(obs_flat, action, next_obs_flat, reward, done)
                step_count += 1
                logger.info(f"Step: {step_count}")
            logger.info(f"{env_id} : Episode {episode + 1}/{total_episodes} finished")
            # 에피소드 종료에 따른 모의 종료 메시지 전송
            send_exit()

    def distributed_train(agent, replay_buffer, total_timesteps=1000000, batch_size=32):
        timestep = 0
        while timestep < total_timesteps:
            if len(replay_buffer) >= batch_size:
                agent.update(replay_buffer, batch_size)
                timestep += batch_size
                time.sleep(0.5)

    # Wait for external environment to be turned on
    time.sleep(5)
    logger.info("Gym env is created.")

    env_threads = []
    for env_id, env in enumerate(envs):
        thread = threading.Thread(
            target=run_environment,
            args=(
                env,
                agent,
                replay_buffer,
                env_id,
                args.max_episode_len,
                args.total_episodes,
            ),
        )
        env_threads.append(thread)
        thread.start()

    for thread in env_threads:
        thread.join()

    distributed_train(agent, replay_buffer, args.total_timesteps)

    if rank == 0:
        torch.save(agent.actor.module.state_dict(), "actor_network.pth")
        torch.save(agent.critic.module.state_dict(), "critic_network.pth")

    cleanup()


if __name__ == "__main__":
    os.system("cls" if os.name in ["nt", "dos"] else "clear")

    parser = argparse.ArgumentParser(
        description="Run training with multiple environments using DDP."
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=2,
        help="Number of environments to run",
    )
    parser.add_argument(
        "--max_episode_len",
        type=int,
        default=1000,
        help="Maximum length of an episode",
    )
    parser.add_argument(
        "--total_episodes",
        type=int,
        default=100,
        help="Total episodes per environment",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=1000000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--world_size", type=int, default=1, help="Number of GPUs to use"
    )
    args = parser.parse_args()

    world_size = args.world_size
    logger.info(args)
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
