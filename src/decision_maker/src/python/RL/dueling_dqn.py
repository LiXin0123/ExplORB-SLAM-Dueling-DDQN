import torch
import torch.nn as nn
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from replay_buffer import ReplayBuffer

class DuelingDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DuelingDQN, self).__init__()
        # Feature extraction layers
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)

        # Value stream
        self.value_stream_fc1 = nn.Linear(64, 32)
        self.value_stream_fc2 = nn.Linear(32, 1)

        # Advantage stream
        self.advantage_stream_fc1 = nn.Linear(64, 32)
        self.advantage_stream_fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        # Forward pass through feature layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Value stream
        value = torch.relu(self.value_stream_fc1(x))
        value = self.value_stream_fc2(value)

        # Advantage stream
        advantage = torch.relu(self.advantage_stream_fc1(x))
        advantage = self.advantage_stream_fc2(advantage)

        # Combine value and advantage streams to get Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class DuelingDQNAgent:
    def __init__(self, model_path, gazebo_env, gamma, learning_rate, epsilon, epsilon_min, epsilon_decay,
                 save_interval, epochs, batch_size, penalty, robot_post_arr, robot_orie_arr, centr_arr, info_arr, best_centr_arr, repeat_count):
        # Parameters initialization
        self.robot_post_arr = robot_post_arr[0]
        self.robot_orie_arr = robot_orie_arr[0]
        self.centr_arr = centr_arr[0]
        self.info_arr = info_arr[0]
        self.best_centr_arr = best_centr_arr[0]

        self.robot_post_arr2 = robot_post_arr
        self.robot_orie_arr2 = robot_orie_arr
        self.centr_arr2 = centr_arr
        self.info_arr2 = info_arr
        self.best_centr_arr2 = best_centr_arr

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.save_interval = save_interval
        self.epochs = epochs
        self.batch_size = batch_size
        self.penalty = penalty
        self.gazebo_env = gazebo_env
        self.repeat_count = repeat_count  # Store repeat_count

        self.folder_path = f'/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/RL/models/{gazebo_env}/dueling_dqn'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.folder_path_plot = f'/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/RL/plots/{gazebo_env}/dueling_dqn'
        if not os.path.exists(self.folder_path_plot):
            os.makedirs(self.folder_path_plot)

        self.filepath = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dones = None
        self.replay_buffer = ReplayBuffer(1000)
        self.losses = []
        self.initialize_dueling_dqn()

    def prepare_input(self, robot_pos, robot_orie, centr, info):
        """Prepares the input for the Dueling DQN model."""
        robot_position = torch.tensor(robot_pos, dtype=torch.float32)
        robot_orientation = torch.tensor(robot_orie, dtype=torch.float32)
        centroid_record = torch.tensor(centr, dtype=torch.float32)
        info_gain_record = torch.tensor(info, dtype=torch.float32)

        robot_state = torch.cat((robot_position, robot_orientation))
        combined_data = torch.cat((centroid_record, info_gain_record), dim=1)
        sorted_data = combined_data[combined_data[:, -1].argsort(descending=True)]

        sorted_centroid_record = sorted_data[:, :-1]
        sorted_info_gain_record = sorted_data[:, -1]

        network_input = torch.cat(
            (robot_state, sorted_centroid_record.flatten(), sorted_info_gain_record.flatten()))

        input_size = network_input.numel()
        network_input = network_input.reshape(1, input_size)
        output_size = sorted_centroid_record.shape[0]

        return network_input, output_size, sorted_centroid_record

    def initialize_dueling_dqn(self):
        """Initializes the Dueling DQN and target Dueling DQN models."""
        network_input, output_size, _ = self.prepare_input(
            self.robot_post_arr, self.robot_orie_arr, self.centr_arr, self.info_arr)
        self.dueling_dqn = DuelingDQN(network_input.shape[1], output_size).to(self.device)
        self.target_dueling_dqn = DuelingDQN(network_input.shape[1], output_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.dueling_dqn.parameters())

        if os.path.isfile(self.filepath):
            self.load_model()
        else:
            self.save_model()

    def update_target_network(self):
        """Updates the target Dueling DQN parameters using the Dueling DQN parameters."""
        self.target_dueling_dqn.load_state_dict(self.dueling_dqn.state_dict())

    def save_model(self):
        """Saves the target Dueling DQN model."""
        torch.save(self.target_dueling_dqn.state_dict(), self.filepath)

    def load_model(self):
        """Loads the saved model into the target Dueling DQN."""
        self.dueling_dqn.load_state_dict(torch.load(self.filepath, map_location=self.device))
        self.target_dueling_dqn.load_state_dict(torch.load(self.filepath, map_location=self.device))

    def select_action(self, state, output_size, sorted_centroid_record):
        """Selects an action using the epsilon-greedy approach."""
        state = state.to(self.device)
        if random.random() < self.epsilon:
            action = random.randint(0, output_size - 1)
        else:
            with torch.no_grad():
                q_values = self.dueling_dqn(state).clone()
                indices = (sorted_centroid_record == torch.tensor([0.0, 0.0], device=self.device)).all(1).nonzero(as_tuple=True)[0]
                q_values[0, indices] = -self.penalty
                action = q_values.argmax(dim=1).item()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action

    def calculate_reward(self, predicted_centroid, best_centroid, info_gain, distance_cost, exploration_gain):
        """Custom reward function based on information gain, distance, and exploration."""
        reward = info_gain - distance_cost + exploration_gain
        match = torch.all(torch.eq(predicted_centroid, best_centroid))

        if match:
            reward += 1  # Bonus for choosing the best centroid
        else:
            reward -= 1  # Penalty for not choosing the best centroid

        # Apply penalty for selecting a zero-value centroid
        zero_centroid = torch.tensor([0.0, 0.0], device=self.device)
        if torch.all(torch.eq(predicted_centroid, zero_centroid)):
            reward -= self.penalty

        return reward

    def train(self):
        self.dones = torch.zeros((1,), device=self.device)
        for epoch in range(self.epochs):
            for i in range(len(self.robot_post_arr2) - 1):
                self.load_model()
                network_input, output_size, sorted_centroid_record = self.prepare_input(
                    self.robot_post_arr2[i], self.robot_orie_arr2[i], self.centr_arr2[i], self.info_arr2[i])

                actions = self.select_action(network_input, output_size, sorted_centroid_record)
                rewards, predicted_centroid = self.calculate_reward()

                next_state, _, _ = self.prepare_input(
                    self.robot_post_arr2[i + 1], self.robot_orie_arr2[i + 1], self.centr_arr2[i + 1], self.info_arr2[i + 1])
                done = torch.all(torch.eq(predicted_centroid, torch.tensor([0.0, 0.0], device=self.device)))

                self.replay_buffer.push(network_input, actions, rewards, next_state, done)

                if len(self.replay_buffer) >= self.batch_size:
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
                    states = torch.stack(states).to(self.device)
                    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(-1).to(self.device)
                    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
                    next_states = torch.stack(next_states).to(self.device)
                    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)

                    q_values = self.dueling_dqn(states)
                    next_q_values = self.dueling_dqn(next_states)
                    next_actions = next_q_values.argmax(dim=1, keepdim=True)
                    target_next_q_values = self.target_dueling_dqn(next_states)

                    max_next_q_values = target_next_q_values.gather(1, next_actions).detach()
                    targets = rewards + self.gamma * (max_next_q_values * (1 - dones))
                    targets = targets.expand_as(q_values)

                    loss = self.criterion(q_values, targets)
                    self.losses.append(loss.item())

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if (epoch + 1) % self.save_interval == 0:
                self.update_target_network()
                self.save_model()

            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    def save_plot(self):
        """Generates a plot for the training loss."""
        epochs = range(1, self.epochs + 1)
        losses = self.losses[:self.epochs]
        plt.plot(epochs, losses, label='Loss')
        z = np.polyfit(epochs, losses, 1)
        p = np.poly1d(z)
        plt.plot(epochs, p(epochs), "r--", label='Trend')
        plt.title(f'Training Loss: Dueling_DQN_{str(self.repeat_count)}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.folder_path_plot + '/' + 'dueling_dqn_' + str(self.repeat_count) + '.png')

    def update_epsilon(self):
        """Decays epsilon over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_max_info_gain_centroid(self):
        """Finds the centroid with the highest information gain."""
        self.target_dueling_dqn.eval()
        network_input, _, sorted_centroid_record = self.prepare_input(self.robot_post_arr, self.robot_orie_arr, self.centr_arr, self.info_arr)
        with torch.no_grad():
            output = self.target_dueling_dqn(network_input.to(self.device))

        indices = (sorted_centroid_record == torch.tensor([0.0, 0.0], device=self.device)).all(1).nonzero(as_tuple=True)[0]
        output[0, indices] = -self.penalty

        max_info_gain_centroid_idx = output.argmax(dim=1).item()
        max_info_gain_centroid_idx = max_info_gain_centroid_idx % sorted_centroid_record.shape[0]
        max_info_gain_centroid = sorted_centroid_record[max_info_gain_centroid_idx]

        return max_info_gain_centroid, max_info_gain_centroid_idx

    def predict_centroid(self, robot_position, robot_orientation, centroid_records, info_gain_records):
        """Predicts the best centroid based on the given robot position and orientation using the target network."""
        self.target_dueling_dqn.eval()
        network_input, _, sorted_centroid_record = self.prepare_input(robot_position, robot_orientation, centroid_records, info_gain_records)
        with torch.no_grad():
            output = self.target_dueling_dqn(network_input.to(self.device))

        indices = (sorted_centroid_record == torch.tensor([0.0, 0.0], device=self.device)).all(1).nonzero(as_tuple=True)[0]
        output[0, indices] = -self.penalty

        max_info_gain_centroid_idx = output.argmax(dim=1).item()
        max_info_gain_centroid_idx = max_info_gain_centroid_idx % sorted_centroid_record.shape[0]
        max_info_gain_centroid = sorted_centroid_record[max_info_gain_centroid_idx]

        return max_info_gain_centroid, max_info_gain_centroid_idx
