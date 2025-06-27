from utils.replay_buffer import ReplayBuffer
from magent2.environments import battle_v4
from torch_model import QNetwork
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
import random

device = "cpu"

env = battle_v4.env(map_size=45, render_mode="rgb_array")

q_network = QNetwork(env.observation_space("red_0").shape, env.action_space("red_0").n).to(device)
target_network = QNetwork(env.observation_space("red_0").shape, env.action_space("red_0").n).to(device)
target_network.load_state_dict(q_network.state_dict())

replay_buffer = ReplayBuffer(1000, env.observation_space("red_0"), device=device, action_dim=1)
for i in range(15):
    obs = np.ones(env.observation_space("red_0").shape, dtype=np.float32) * i
    action = np.array(np.random.randint(0, 21), dtype=np.int64)
    reward = np.array(i, dtype=np.float32)
    next_obs = obs + 1
    done = np.array(i % 3 == 0, dtype=np.float32)
    replay_buffer.add(obs, action, reward, next_obs, done)

data = replay_buffer.sample(32)
observations, actions, rewards, next_observations, dones = data

with torch.no_grad():
    # Double DQN: use the main network to select actions for the target network
    q_values = q_network(next_observations)
    actions_next = torch.argmax(q_values, dim=1)
    td_target = rewards + 0.99 * (1 - dones) * target_network(next_observations).gather(1, actions_next.unsqueeze(1)).squeeze(1)
q_vals_current = q_network(observations)  # shape: [batch_size, num_actions]
actions = actions.view(-1)
old_val = q_vals_current.gather(1, actions.unsqueeze(1)).squeeze(1)

env.reset(seed=42)

red_network = QNetwork(env.observation_space("red_0").shape, env.action_space("red_0").n)
red_network.load_state_dict(torch.load("red.pt", weights_only=True, map_location="cpu"))

track_data = {f"blue_{i}": {"observation": None, "action": None, "done": None} for i in range(81)}

for global_step in range(200):
    count_action = 0
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            if agent.startswith("blue_"):
                if track_data[agent]["done"] == 0 :
                    prev_obs = track_data[agent]["observation"]
                    prev_action = track_data[agent]["action"]
                    
                    track_data[agent]["observation"] = observation
                    track_data[agent]["action"] = None
                    track_data[agent]["done"] = 1
                    replay_buffer.add(np.array(prev_obs, dtype=np.float32),
                                np.array(prev_action, dtype=np.int64),
                                np.array(reward, dtype=np.float32),
                                np.array(observation, dtype=np.float32),
                                np.array(1, dtype=np.float32))
            action = None

        else:
            agent_handle = agent.split("_")[0]
            if agent_handle == "red":
                observation = (
                torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                )
                with torch.no_grad():
                    q_values = red_network(observation)
                action = torch.argmax(q_values, dim=1).numpy()[0]
            else:
                if random.random() < 0.5:
                    action = env.action_space(agent).sample()
                else:
                    observation_tensor = torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_values = q_network(observation_tensor)
                    action = torch.argmax(q_values, dim=1).numpy()[0]
                if (track_data[agent]["observation"] is not None and
                    track_data[agent]["action"] is not None and
                    track_data[agent]["done"] is not None):
                    prev_obs = track_data[agent]["observation"]
                    prev_action = track_data[agent]["action"]
                    replay_buffer.add(np.array(prev_obs, dtype=np.float32),
                            np.array(prev_action, dtype=np.int64),
                            np.array(reward, dtype=np.float32),
                            np.array(observation, dtype=np.float32),
                            np.array(0, dtype=np.float32))

                track_data[agent]["observation"] = observation
                track_data[agent]["action"] = action
                track_data[agent]["done"] = 0
        env.step(action)
        count_action += 1
    if (len(env.agents) == 0):
        env.reset()
        print("All agents have died, resetting environment at step:", global_step)
        print(f"Num action of current step: {count_action}")

print(f"Buffer size: {replay_buffer.size()}")
batch = replay_buffer.sample(1000)
# print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape, batch[4].shape)
# print("Sampled batch:")
# for arr in batch[4]:
#     print(arr.item(), end=" ")