import numpy as np
import torch
from typing import Any
from magent2.environments import battle_v4


class ReplayBuffer:
    def __init__(
            self,
            buffer_size: int,
            observation_space: Any,  # expects an object with .shape 
            action_dim: int = 1,
            device: str = "cpu",
            ) -> None:
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_dim = action_dim
        self.device = device
        self.pos = 0
        self.full = False
        self.observations = np.zeros((buffer_size, *observation_space.shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *observation_space.shape), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        self.reset()

    def size(self) -> int:
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, next_obs: np.ndarray, done: np.ndarray) -> None:
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.pos += 1

        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    
    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)
    
    def _get_samples(self, batch_inds: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = self.observations[batch_inds]
        actions = self.actions[batch_inds]
        rewards = self.rewards[batch_inds]
        next_observations = self.next_observations[batch_inds]
        dones = self.dones[batch_inds]
        return self.to_torch(observations).permute([0, 3, 1, 2]), self.to_torch(actions), self.to_torch(rewards), self.to_torch(next_observations).permute([0, 3, 1, 2]), self.to_torch(dones)

    def reset(self) -> None:
        self.pos = 0
        self.full = False
    

    def to_torch(self, np_array: np.ndarray, copy: bool = True) -> torch.Tensor:
        if copy:
            return torch.tensor(np_array).to(self.device)
        return torch.as_tensor(np_array).to(self.device)

if __name__ == "__main__":
    # Dummy spaces
    env = battle_v4.env(map_size=45, render_mode="rgb_array")
    env.reset()
    obs_space = env.observation_space("red_0")
    act_space = env.action_space("red_0")
    print(obs_space.shape)
    print(act_space.n)  
    print(obs_space.dtype)
    print(act_space.dtype)

    buffer = ReplayBuffer(buffer_size=10, observation_space=obs_space, action_dim=1, device="cpu")

    # Add some dummy transitions
    for i in range(15):
        obs = np.ones(obs_space.shape, dtype=np.float32) * i
        action = np.array(np.random.randint(0, act_space.n), dtype=np.int64)
        reward = np.array(i, dtype=np.float32)
        next_obs = obs + 1
        done = np.array(i % 3 == 0, dtype=np.float32)
        buffer.add(obs, action, reward, next_obs, done)

    print(f"Buffer size: {buffer.size()}")
    batch = buffer.sample(5)
    print("Sampled batch:")
    for arr in batch:
        print(arr.shape)