import argparse
import os
import pathlib
import sys
import threading
import time
from threading import Thread
from typing import List

import numpy as np
import torch
from actor_critic import ActorCritic
from env import UnrealGymEnv
from replay_buffer import ReplayBuffer

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from commu.sim.sender.sim_sender import send_exit
from mission_planning.configs.config import MainArgs
from utils.logger import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_environment(
    env: UnrealGymEnv,
    agent: ActorCritic,
    replay_buffer: ReplayBuffer,
    env_id: int,
    max_episode_len: int = 1000,
    total_episodes: int = 100,
):
    for episode in range(total_episodes):
        observation, _ = env.reset()
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
                for trg_id, trg_data in state["target"].items():
                    target_state = np.concatenate([trg_data.pos, [trg_data.value]])
                    target_state_list.append(target_state)
                target_state_array = np.concatenate(target_state_list)
                obs_flat = np.concatenate([scenario, agent_state, target_state_array])
                action = agent.select_action(obs_flat)
                next_observation, reward, terminated, truncated, info = env.step(action)
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
                for trg_id, next_target_data in next_state["target"].items():
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
        logger.info(f"{env_id} : Episode {episode + 1}/{total_episodes} finished")
        # 에피소드 종료에 따른 모의 종료 메시지 전송
        send_exit((MainArgs.addr_ts_ip, env.udp_port - 2929 + 7))


def train(agent, replay_buffer, total_timesteps=1000000, batch_size=32):
    timestep = 0
    while timestep < total_timesteps:
        if len(replay_buffer) >= batch_size:
            agent.update(replay_buffer, batch_size)
            logger.info(f"Step: {timestep}")
            timestep += batch_size
            time.sleep(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training with multiple environments."
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
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
        "--total_timesteps",
        type=int,
        default=1000000,
        help="Total timesteps for training",
    )
    args = parser.parse_args()
    logger.info(args)

    # Scenario (40000) + Agent (7) + Target (80) -> 40087
    input_dim = 40000 + 7 + 80
    output_dim = 2  # Example action dimension (way point)
    agent = ActorCritic(input_dim, output_dim)
    replay_buffer = ReplayBuffer(capacity=5000)
    envs = [UnrealGymEnv(udp_port=2929 + (i + 1) * 10000) for i in range(args.num_envs)]
    # wait External Env to be turned on
    time.sleep(5)
    logger.info("Gymnasium env is created.")
    # Environment threads for experience collection
    env_threads: List[Thread] = []
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
    # Training thread
    train_thread = threading.Thread(
        target=train, args=(agent, replay_buffer, args.total_timesteps)
    )
    train_thread.start()
    # Wait for all threads to complete
    for thread in env_threads:
        thread.join()
    train_thread.join()
    #
    torch.save(agent.actor.state_dict(), "actor_network.pth")
    torch.save(agent.critic.state_dict(), "critic_network.pth")
    #
    logger.info("Training finished!")
