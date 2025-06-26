from utils.replay_buffer import ReplayBuffer
from magent2.environments import battle_v4
from torch_model import QNetwork
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
import random
from tqdm import tqdm


@dataclass
class TrainingConfig:
    seed: int = 42
    cuda: bool = True
    save_model: bool = True
    save_nums: int = 5
    
    total_timesteps: int = 10000000
    learning_rate: float = 0.0001
    buffer_size: int = 1000000
    gamma: float = 0.8
    tau: float = 1
    target_network_frequency: int = 1000
    batch_size: int = 64
    start_e: float = 1
    end_e: float = 0.01
    exploration_fraction: float = 0.1
    learning_starts: int = 80000
    train_frequency: int = 4

def make_env():
    env = battle_v4.env(map_size=45, render_mode="rgb_array")
    return env

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def main():
    config = TrainingConfig(total_timesteps=1000000, target_network_frequency=500, learning_starts=5000, buffer_size=50000)
    # config = TrainingConfig()
    env = make_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    red_network = QNetwork(env.observation_space("red_0").shape, env.action_space("red_0").n)
    red_network.load_state_dict(torch.load("red.pt", weights_only=True, map_location="cpu"))

    q_network = QNetwork(env.observation_space("red_0").shape, env.action_space("red_0").n).to(device)
    target_network = QNetwork(env.observation_space("red_0").shape, env.action_space("red_0").n).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=config.learning_rate)
    replay_buffer = ReplayBuffer(config.buffer_size, env.observation_space("red_0"), device=device, action_dim=1)

    env.reset(seed=config.seed)

    track_data = {f"blue_{i}": {"observation": None, "action": None, "done": None} for i in range(81)}


    for global_step in tqdm(range(config.total_timesteps)):
        epsilon = linear_schedule(config.start_e, config.end_e, config.exploration_fraction * config.total_timesteps, global_step)
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
                    if random.random() < epsilon:
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
        if global_step >= config.learning_starts:
            if global_step % config.train_frequency == 0:
                data = replay_buffer.sample(config.batch_size)
                observations, actions, rewards, next_observations, dones = data
                with torch.no_grad():
                    # Double DQN: use the main network to select actions for the target network
                    q_values = q_network(next_observations)
                    actions_next = torch.argmax(q_values, dim=1)
                    td_target = rewards.squeeze() + config.gamma * (1 - dones.squeeze()) * target_network(next_observations).gather(1, actions_next.unsqueeze(1)).squeeze(1)
                q_vals_current = q_network(observations)  # shape: [batch_size, num_actions]
                actions = actions.view(-1)
                old_val = q_vals_current.gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = nn.functional.mse_loss(old_val, td_target)
                if global_step % 100000 == 0:
                    print(f"Step {global_step}, Loss: {loss.item()}, Epsilon: {epsilon:.4f}, Q-Value: {old_val.mean().item()}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if global_step % config.target_network_frequency == 0:
                for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(config.tau * param.data + (1 - config.tau) * target_param.data)

        if config.save_model and global_step > config.learning_starts and (global_step - config.learning_starts) % ((config.total_timesteps - config.learning_starts) // config.save_nums) == 0:
            save_dir = f"model_{global_step}.pt"
            # os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            torch.save(q_network.state_dict(), save_dir)
    if config.save_model:
        save_dir = f"final_model.pt"
        # os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        torch.save(q_network.state_dict(), save_dir)
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
# This script trains a reinforcement learning agent using a DQN algorithm on the Magent battle environment.
# It includes a training loop, epsilon-greedy exploration strategy, and model saving functionality.
# The agent learns to navigate and interact with the environment, optimizing its actions based on rewards received.
# The training configuration is defined using a dataclass for better organization and readability.
            


