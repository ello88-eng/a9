import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from replay_buffer import ReplayBuffer
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from utils.logger import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.lin_0 = nn.Linear(input_dim, 128)
        self.lin_1 = nn.Linear(128, 128)
        self.lin_2 = nn.Linear(128, output_dim)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.lin_0(x))
        x = F.relu(self.lin_1(x))
        x = self.lin_2(x)
        return x


class Critic(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.lin_0 = nn.Linear(input_dim + output_dim, 128)
        self.lin_1 = nn.Linear(128, 128)
        self.lin_2 = nn.Linear(128, 1)

    def forward(self, x, action) -> torch.Tensor:
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.lin_0(x))
        x = F.relu(self.lin_1(x))
        x = self.lin_2(x)
        return x


class ActorCritic:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
    ):
        self.actor = Actor(input_dim, output_dim).to(device)
        self.critic = Critic(input_dim, output_dim).to(device)
        self.actor_optim = optim.Adam(params=self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(params=self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = self.actor.forward(state)
        return action.cpu().detach().numpy()[0]

    def update(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 32,
        gamma: float = 0.99,
    ):
        states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
        # Critic update
        next_actions = self.actor.forward(next_states)
        target_q = rewards + gamma * self.critic.forward(next_states, next_actions) * (
            1 - dones
        )
        current_q = self.critic.forward(states, actions)
        critic_loss = F.mse_loss(current_q, target_q.detach())
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        # Actor update
        actor_loss = -self.critic.forward(states, self.actor(states)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        logger.info(
            f"Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}"
        )
