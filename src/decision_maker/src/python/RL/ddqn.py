import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer

class DDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DDQNAgent:
    def __init__(self, model_path, gamma, learning_rate, epsilon, epsilon_min, epsilon_decay, 
                 save_interval, epochs, batch_size, penalty, robot_post_arr, robot_orie_arr, centr_arr, info_arr, best_centr_arr):
        # Initialize parameters
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.save_interval = save_interval
        self.epochs = epochs
        self.batch_size = batch_size
        self.penalty = penalty
        self.model_path = model_path

        # Input data
        self.robot_post_arr = robot_post_arr
        self.robot_orie_arr = robot_orie_arr
        self.centr_arr = centr_arr
        self.info_arr = info_arr
        self.best_centr_arr = best_centr_arr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create directories if they don't exist
        self.folder_path = os.path.join(model_path, "ddqn")
        os.makedirs(self.folder_path, exist_ok=True)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(1000)

        # Initialize the DDQN model and target model
        self.ddqn = DDQN(input_size=128, output_size=64).to(self.device)
        self.target_ddqn = DDQN(input_size=128, output_size=64).to(self.device)

        self.optimizer = optim.Adam(self.ddqn.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # Load pre-trained model if available
        if os.path.exists(self.model_path):
            self.load_model()

    def load_model(self):
        self.ddqn.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.target_ddqn.load_state_dict(torch.load(self.model_path, map_location=self.device))

    def save_model(self):
        torch.save(self.target_ddqn.state_dict(), self.model_path)

    def update_target_network(self):
        """Update target network with DDQN's weights"""
        self.target_ddqn.load_state_dict(self.ddqn.state_dict())

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.ddqn.fc3.out_features - 1)
        else:
            with torch.no_grad():
                q_values = self.ddqn(state)
            return q_values.argmax().item()

    def prepare_input(self, robot_pos, robot_orie, centr, info):
        """Prepare the input for the DDQN model"""
        robot_position = torch.tensor(robot_pos, dtype=torch.float32).to(self.device)
        robot_orientation = torch.tensor(robot_orie, dtype=torch.float32).to(self.device)
        centroid_record = torch.tensor(centr, dtype=torch.float32).to(self.device)
        info_gain_record = torch.tensor(info, dtype=torch.float32).to(self.device)

        robot_state = torch.cat((robot_position, robot_orientation))
        combined_data = torch.cat((centroid_record, info_gain_record), dim=1)
        sorted_data = combined_data[combined_data[:, -1].argsort(descending=True)]
        sorted_centroid_record = sorted_data[:, :-1]
        sorted_info_gain_record = sorted_data[:, -1]

        input_data = torch.cat((robot_state, sorted_centroid_record.flatten(), sorted_info_gain_record.flatten()))
        input_size = input_data.numel()
        input_data = input_data.reshape(1, input_size)

        return input_data

    def calculate_reward(self):
        """Calculate the reward based on the predicted centroid and the best centroid"""
        predicted_centroid, _ = self.get_max_info_gain_centroid()

        target_centroid = torch.tensor(self.best_centr_arr, dtype=torch.float32, device=self.device)
        predicted_centroid = predicted_centroid.to(self.device)

        match = torch.all(torch.eq(predicted_centroid, target_centroid))
        if match:
            reward = 1
        else:
            reward = 0

        zero_centroid = torch.tensor([0.0, 0.0], device=self.device)
        if torch.all(torch.eq(predicted_centroid, zero_centroid)):
            reward -= self.penalty

        return reward, predicted_centroid

    def train(self):
        for epoch in range(self.epochs):
            for i in range(len(self.robot_post_arr) - 1):
                state = self.prepare_input(self.robot_post_arr[i], self.robot_orie_arr[i], self.centr_arr[i], self.info_arr[i])

                action = self.select_action(state)
                reward, predicted_centroid = self.calculate_reward()

                next_state = self.prepare_input(self.robot_post_arr[i+1], self.robot_orie_arr[i+1], self.centr_arr[i+1], self.info_arr[i+1])

                self.replay_buffer.push(state, action, reward, next_state)

                if len(self.replay_buffer) > self.batch_size:
                    self.update_model()

            if (epoch + 1) % self.save_interval == 0:
                self.update_target_network()
                self.save_model()

    def update_model(self):
        states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)

        q_values = self.ddqn(states).gather(1, actions.unsqueeze(-1))
        next_q_values = self.target_ddqn(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values.unsqueeze(1)

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_max_info_gain_centroid(self):
        self.target_ddqn.eval()
        state = self.prepare_input(self.robot_post_arr, self.robot_orie_arr, self.centr_arr, self.info_arr)
        with torch.no_grad():
            output = self.target_ddqn(state)
        indices = (self.centr_arr == torch.tensor([0.0, 0.0])).all(1).nonzero(as_tuple=True)[0]
        output[0, indices] = -self.penalty
        max_info_gain_centroid_idx = output.argmax(dim=1).item()
        max_info_gain_centroid = self.centr_arr[max_info_gain_centroid_idx]
        return max_info_gain_centroid, max_info_gain_centroid_idx
