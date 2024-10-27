import numpy as np
import os
import random
from csv_handler import combine_csv, read_csv

# Define the environment and file paths
gazebo_env = 'aws_bookstore'
repeat_count = 20

folder_path = f'/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/train_data/{gazebo_env}/{repeat_count}'
combined_output_path = f'/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/combined_results/{gazebo_env}/{gazebo_env}_{repeat_count}.csv'
model_path = f"/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/RL/models/{gazebo_env}/q_learning/q_table.npy"

# RL Parameters
learning_rate = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.99
num_epochs = 200

# Initialize Q-table (state-action table)
q_table = {}

def preprocess_state(robot_position, robot_orientation, centroid_record):
    # Convert centroid_record to a NumPy array if it's a list
    if isinstance(centroid_record, list):
        centroid_record = np.array(centroid_record)

    # Flatten the centroid_record and round all values
    state = (
        tuple(np.round(robot_position)),
        tuple(np.round(robot_orientation)),
        tuple(np.round(centroid_record.flatten()))
    )
    
    return state

def select_action(state, possible_actions):
    """Epsilon-greedy policy for action selection."""
    if state not in q_table:
        q_table[state] = np.zeros(len(possible_actions))
    
    if random.uniform(0, 1) < epsilon:
        # Exploration: choose a random action
        return random.choice(range(len(possible_actions)))
    else:
        # Exploitation: choose the action with the highest Q-value
        return np.argmax(q_table[state])

def calculate_reward(predicted_centroid, best_centroid, distance_cost, exploration_gain, visit_penalty):
    """Calculate the reward using the provided factors."""
    reward = 0

    if np.array_equal(predicted_centroid, best_centroid):
        reward += 1
    
    if np.array_equal(predicted_centroid, [0.0, 0.0]):
        reward -= 1  # Penalty for zero centroid
    
    reward -= 0.5 * distance_cost
    reward += 0.25 * exploration_gain
    reward -= 0.2 * visit_penalty

    return reward

def update_q_table(state, action, reward, next_state, possible_actions):
    """Update the Q-table using the Q-learning formula."""
    if next_state not in q_table:
        q_table[next_state] = np.zeros(len(possible_actions))
    
    best_next_action = np.argmax(q_table[next_state])
    q_table[state][action] += learning_rate * (reward + gamma * q_table[next_state][best_next_action] - q_table[state][action])

def train():
    global epsilon
    # Combine and read CSV data
    combine_csv(folder_path, combined_output_path)
    robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroids = read_csv(combined_output_path)

    num_samples = len(robot_positions)

    for epoch in range(num_epochs):
        for i in range(num_samples - 1):
            state = preprocess_state(robot_positions[i], robot_orientations[i], centroid_records[i])
            next_state = preprocess_state(robot_positions[i + 1], robot_orientations[i + 1], centroid_records[i + 1])

            possible_actions = list(range(len(centroid_records[i])))
            action = select_action(state, possible_actions)

            # Calculate reward
            predicted_centroid = centroid_records[i][action]
            best_centroid = best_centroids[i]
            distance_cost = np.linalg.norm(np.array(robot_positions[i]) - np.array(predicted_centroid))
            exploration_gain = 1 / (1 + np.sum(predicted_centroid == centroid_records[i]))
            visit_penalty = np.sum(predicted_centroid == centroid_records[i])

            reward = calculate_reward(predicted_centroid, best_centroid, distance_cost, exploration_gain, visit_penalty)

            # Update Q-table
            update_q_table(state, action, reward, next_state, possible_actions)

        # Decay epsilon after each epoch
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Epoch {epoch+1}/{num_epochs} completed.")

    # Save the Q-table
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    np.save(model_path, q_table)
    print("Q-table saved successfully.")

def test():
    # Load Q-table
    if os.path.exists(model_path):
        global q_table
        q_table = np.load(model_path, allow_pickle=True).item()
        print("Q-table loaded successfully.")
    else:
        print("Q-table not found. Please train the model first.")
        return

    # Combine and read CSV data for testing
    robot_positions, robot_orientations, centroid_records, _, _ = read_csv(combined_output_path)

    for i in range(len(robot_positions)):
        state = preprocess_state(robot_positions[i], robot_orientations[i], centroid_records[i])
        possible_actions = list(range(len(centroid_records[i])))

        if state not in q_table:
            q_table[state] = np.zeros(len(possible_actions))
        
        action = np.argmax(q_table[state])
        predicted_centroid = centroid_records[i][action]
        print(f"Step {i+1}: Predicted centroid {predicted_centroid}")

if __name__ == "__main__":
    train()
    test()
