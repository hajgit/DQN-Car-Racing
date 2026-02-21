
import os
import torch
import csv
import gymnasium as gym
import numpy as np
from tensordict import TensorDict
from torch import nn
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from typing import Union

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info

class DQN(nn.Module):
    def __init__(self, in_dim: tuple, out_dim: int):
        super().__init__()
        channel_n, height, width = in_dim
        if height != 84 or width != 84:
            raise ValueError(f"Input must be (84,84). Got ({height},{width})")
        self.net = nn.Sequential(
            nn.Conv2d(channel_n, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(
        self,
        state_space_shape,
        action_n,
        load_state="",
        load_model=None,
        double_q=True,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.9999925,
        epsilon_min=0.05
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.state_shape = state_space_shape
        self.action_n = action_n
        self.load_state = load_state
        self.double_q = double_q
        self.save_dir = './training/saved_models/'
        self.log_dir = './training/logs/'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.updating_net = DQN(self.state_shape, self.action_n).to(self.device)
        self.frozen_net = DQN(self.state_shape, self.action_n).to(self.device)
        self.optimizer = torch.optim.Adam(self.updating_net.parameters(), lr=0.0002)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(25000, device=torch.device("cpu"))
        )
        self.act_taken = 0
        self.n_updates = 0
        if load_state:
            self._load_model(load_model)

    def _load_model(self, model_name):
        if model_name is None:
            raise ValueError("Model name required")
        path = os.path.join(self.save_dir, model_name)
        data = torch.load(path, map_location=self.device)
        self.updating_net.load_state_dict(data['upd_model_state_dict'])
        self.frozen_net.load_state_dict(data['frz_model_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        if self.load_state == 'eval':
            self.updating_net.eval()
            self.frozen_net.eval()
            self.epsilon = 0
        else:
            self.act_taken = data['action_number']
            self.epsilon = data['epsilon']

    def store(self, state, action, reward, new_state, terminated):
        self.buffer.add(TensorDict({
            "state": torch.tensor(state, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.long),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "new_state": torch.tensor(new_state, dtype=torch.float32),
            "terminated": torch.tensor(terminated, dtype=torch.bool)
        }, batch_size=[]))

    def get_samples(self, batch_size):
        batch = self.buffer.sample(batch_size)
        return (
            batch["state"].to(self.device),
            batch["action"].squeeze().to(self.device),
            batch["reward"].squeeze().to(self.device),
            batch["new_state"].to(self.device),
            batch["terminated"].squeeze().to(self.device)
        )

    def take_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_n)
        else:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = self.updating_net(state_t).argmax().item()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
        self.act_taken += 1
        return action

    def update_net(self, batch_size):
        self.n_updates += 1
        states, actions, rewards, new_states, dones = self.get_samples(batch_size)
        q_values = self.updating_net(states)
        q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double_q:
                next_actions = self.updating_net(new_states).argmax(1)
                q_next = self.frozen_net(new_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                q_next = self.frozen_net(new_states).max(1)[0]
            q_target = rewards + (1 - dones.float()) * self.gamma * q_next
        loss = self.loss_fn(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return q_current, loss.item()

    def save(self, save_dir, name):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{name}_{self.act_taken}.pt")
        torch.save({
            'upd_model_state_dict': self.updating_net.state_dict(),
            'frz_model_state_dict': self.frozen_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_number': self.act_taken,
            'epsilon': self.epsilon
        }, path)
        print(f" Saved: {path}")

    def write_log(self, dates, times, rewards, lengths, losses, epsilons, filename='log.csv'):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, filename)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['date'] + dates)
            writer.writerow(['time'] + times)
            writer.writerow(['reward'] + rewards)
            writer.writerow(['length'] + lengths)
            writer.writerow(['loss'] + losses)
            writer.writerow(['epsilon'] + epsilons)
