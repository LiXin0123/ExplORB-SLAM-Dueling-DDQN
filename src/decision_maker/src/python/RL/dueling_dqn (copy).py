import torch
import torch.nn as nn
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from replay_buffer import ReplayBuffer
from train_model import repeat_count


class DuelingDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DuelingDQN, self).__init__()

        # Feature layer extracts the common features from the input state
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, 128),  # Fully connected layer
            nn.ReLU()  # Activation function
        )

        # Value stream estimates the state value
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),  # Fully connected layer
            nn.ReLU(),  # Activation function
            nn.Linear(64, 1)  # Fully connected layer with single output unit
        )

        # Advantage stream estimates the advantages of each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),  # Fully connected layer
            nn.ReLU(),  # Activation function
            # Fully connected layer with output units matching the number of actions
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # Pass the input state through the feature layer
        features = self.feature_layer(x)

        # Compute the state value by passing the features through the value stream
        values = self.value_stream(features)

        # Compute the advantages of each action by passing the features through the advantage stream
        advantages = self.advantage_stream(features)

        # Combine the value and advantages to obtain the Q-values
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


class DuelingDQNAgent:
    def __init__(self, model_path, gazebo_env, gamma, learning_rate, epsilon, epsilon_min, epsilon_decay,
                 save_interval, epochs, batch_size, penalty, robot_post_arr, robot_orie_arr, centr_arr, info_arr, best_centr_arr):
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

        self.folder_path = f'/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/RL/models/{gazebo_env}/dueling_dqn'
        # Create directory if it does not exist
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.folder_path_plot = f'/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/RL/plots/{gazebo_env}/dueling_dqn'
        # Create directory if it does not exist
        if not os.path.exists(self.folder_path_plot):
            os.makedirs(self.folder_path_plot)

        self.filepath = model_path

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dones = None

        # Initialize the replay buffer
        self.replay_buffer = ReplayBuffer(1000)

        # plot loss
        self.losses = []

        # Initialize the Dueling DQN network
        self.initialize_dueling_dqn()

    def prepare_input(self, robot_pos, robot_orie, centr, info):
        robot_position = torch.tensor(robot_pos, dtype=torch.float32)
        robot_orientation = torch.tensor(robot_orie, dtype=torch.float32)
        centroid_record = torch.tensor(centr, dtype=torch.float32)
        info_gain_record = torch.tensor(info, dtype=torch.float32)

        robot_state = torch.cat((robot_position, robot_orientation))
        combined_data = torch.cat((centroid_record, info_gain_record), dim=1)
        sorted_data = combined_data[combined_data[:, -
                                                  1].argsort(descending=True)]

        sorted_centroid_record = sorted_data[:, :-1]
        sorted_info_gain_record = sorted_data[:, -1]

        network_input = torch.cat(
            (robot_state, sorted_centroid_record.flatten(), sorted_info_gain_record.flatten()))
        input_size = network_input.numel()
        network_input = network_input.reshape(1, input_size)
        output_size = sorted_centroid_record.shape[0]

        return network_input, output_size, sorted_centroid_record

    def initialize_dueling_dqn(self):
        network_input, output_size, _ = self.prepare_input(
            self.robot_post_arr, self.robot_orie_arr, self.centr_arr, self.info_arr)
        self.dueling_dqn = DuelingDQN(
            network_input.shape[1], output_size).to(self.device)
        self.target_dueling_dqn = DuelingDQN(
            network_input.shape[1], output_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.dueling_dqn.parameters())

        # Check if a model already exists
        if os.path.isfile(self.filepath):
            # If a model does exist, load it
            self.load_model()
        else:
            # If no model exists, save the newly created one
            self.save_model()

    def update_target_network(self):
        self.target_dueling_dqn.load_state_dict(self.dueling_dqn.state_dict())

    def save_model(self):
        torch.save(self.target_dueling_dqn.state_dict(), self.filepath)

    def load_model(self):
        self.dueling_dqn.load_state_dict(torch.load(
            self.filepath, map_location=self.device))
        self.target_dueling_dqn.load_state_dict(torch.load(
            self.filepath, map_location=self.device))

    def select_action(self, state, output_size, sorted_centroid_record):
        """Selects an action using the epsilon-greedy approach."""

        state = state.to(self.device)
        
        if random.random() < self.epsilon:
            action = random.randint(0, output_size - 1)
        else:
            with torch.no_grad():
                q_values = self.dueling_dqn(state).clone()

                # Get the indices of the centroids which are [0.0, 0.0]
                indices = (sorted_centroid_record == torch.tensor(
                    [0.0, 0.0])).all(1).nonzero(as_tuple=True)[0]

                # Apply penalty to q_values at those indices
                q_values[0, indices] = -self.penalty

                action = q_values.argmax(dim=1).item()

        # Apply epsilon decay if epsilon is more than its minimum value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action
    
    

    def calculate_reward(self):
        """Calculates the reward for a given centroid."""
        predicted_centroid, _ = self.get_max_info_gain_centroid()

        target_centroid = torch.tensor(
            self.best_centr_arr, dtype=torch.float32, device=self.device)

        predicted_centroid = predicted_centroid.to(self.device)


        # Check if the predicted centroid matches the best centroid
        match = torch.all(torch.eq(predicted_centroid, target_centroid))

        # Increase the reward for the chosen predicted centroid if it matches the target centroid
        if match:
            reward = 1
        else:
            reward = 0

        # Apply penalty to the reward if the predicted centroid is [0.0, 0.0]
        zero_centroid = torch.tensor([0.0, 0.0], device=self.device)
        if torch.all(torch.eq(predicted_centroid, zero_centroid)):
            reward -= self.penalty

        return reward, predicted_centroid

    def train(self):
        self.dones = torch.zeros((1,), device=self.device)

        self.replay_buffer = ReplayBuffer(10000)
        for epoch in range(self.epochs):
            for i in range(len(self.robot_post_arr2)-1):
                self.load_model()
                network_input, output_size, sorted_centroid_record = self.prepare_input(
                    self.robot_post_arr2[i], self.robot_orie_arr2[i], self.centr_arr2[i], self.info_arr2[i])

                actions = self.select_action(
                    network_input, output_size, sorted_centroid_record)

                rewards, predicted_centroid = self.calculate_reward()

                next_state, _, _ = self.prepare_input(
                    self.robot_post_arr2[i+1], self.robot_orie_arr2[i+1], self.centr_arr2[i+1], self.info_arr2[i+1])
                done = torch.all(torch.eq(predicted_centroid,
                                 torch.tensor([0.0, 0.0], device=self.device)))

                self.replay_buffer.push(
                    network_input, actions, rewards, next_state, done)

                if len(self.replay_buffer) >= self.batch_size:
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                        self.batch_size)

                    states = torch.stack(states).to(self.device)
                    actions = torch.tensor(
                        actions, dtype=torch.long).unsqueeze(-1).to(self.device)
                    rewards = torch.tensor(
                        rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
                    next_states = torch.stack(next_states).to(self.device)
                    dones = torch.tensor(
                        dones, dtype=torch.float32).unsqueeze(-1).to(self.device)

                    q_values = self.dueling_dqn(states)
                    next_q_values = self.target_dueling_dqn(next_states)
                    max_next_q_values = next_q_values.max(
                        dim=1, keepdim=True)[0]

                    targets = q_values + self.learning_rate * \
                        (rewards + self.gamma *
                         max_next_q_values - q_values)

                    targets = targets.expand_as(q_values)
                    loss = self.criterion(targets, q_values)

                    # Append the loss to the losses list
                    self.losses.append(loss.item())

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.update_epsilon()

            if (epoch+1) % self.save_interval == 0:
                print("Updating target network")
                self.update_target_network()
                self.save_model()

            print(f"Epoch: {epoch+1}, MSE Loss: {loss.item()}")

    def save_plot(self):
        epochs = range(1, self.epochs + 1)

        losses = self.losses[:self.epochs]

        plt.plot(epochs, losses, label='Loss')

        # Calculate the trend line
        z = np.polyfit(epochs, losses, 1)
        p = np.poly1d(z)
        plt.plot(epochs, p(epochs), "r--", label='Trend')

        plt.title(f'Training Loss: Dueling_DQN_{str(repeat_count)}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.legend()

        # Save the plot to a file
        plt.savefig(self.folder_path_plot + '/' + 'dueling_dqn' +
                    '_' + str(repeat_count) + '.png')

    def update_epsilon(self):
        """Decays epsilon over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_max_info_gain_centroid(self):
        self.target_dueling_dqn.eval()
        """Finds the centroid with the highest information gain."""
        network_input, _, sorted_centroid_record = self.prepare_input(
            self.robot_post_arr, self.robot_orie_arr, self.centr_arr, self.info_arr
        )
        with torch.no_grad():
            output = self.target_dueling_dqn(network_input.to(self.device))

        # Get the indices of the centroids which are [0.0, 0.0]
        indices = (sorted_centroid_record == torch.tensor(
            [0.0, 0.0])).all(1).nonzero(as_tuple=True)[0]

        # Apply penalty to output at those indices
        output[0, indices] = -self.penalty

        max_info_gain_centroid_idx = output.argmax(dim=1).item()
        max_info_gain_centroid_idx = max_info_gain_centroid_idx % sorted_centroid_record.shape[
            0]
        max_info_gain_centroid = sorted_centroid_record[max_info_gain_centroid_idx]

        return max_info_gain_centroid, max_info_gain_centroid_idx

    # for testing
    def predict_centroid(self, robot_position, robot_orientation, centroid_records, info_gain_records):
        """Predicts the best centroid based on the given robot position and orientation using the target network."""
        self.target_dueling_dqn.eval()

        """Finds the centroid with the highest information gain."""
        network_input, _, sorted_centroid_record = self.prepare_input(
            robot_position, robot_orientation, centroid_records, info_gain_records
        )

        with torch.no_grad():
            output = self.target_dueling_dqn(network_input.to(self.device))

        # Get the indices of the centroids which are [0.0, 0.0]
        indices = (sorted_centroid_record == torch.tensor(
            [0.0, 0.0])).all(1).nonzero(as_tuple=True)[0]

        # Apply penalty to output at those indices
        output[0, indices] = -self.penalty

        max_info_gain_centroid_idx = output.argmax(dim=1).item()
        max_info_gain_centroid_idx = max_info_gain_centroid_idx % sorted_centroid_record.shape[
            0]
        max_info_gain_centroid = sorted_centroid_record[max_info_gain_centroid_idx]

        return max_info_gain_centroid, max_info_gain_centroid_idx