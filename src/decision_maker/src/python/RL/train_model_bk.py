from agent import *
from csv_handler import *
import sys
import torch
import torch.optim as optim

sys.path.append('/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/')

# RL Parameters
gazebo_env = 'aws_bookstore'
algo = 'dueling_ddqn'  # 目前根据代码，使用的是 DDQN 算法
repeat_count = 20
gamma = 0.90
learning_rate = 0.0005
epsilon = 1
epsilon_min = 0.1
epsilon_decay = 0.02
epochs = 200
save_interval = 10
batch_size = 1
penalty = 2

# CSV 文件路径
folder_path = f'/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/train_data/{gazebo_env}/{repeat_count}'

combined_output_path = f'/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/combined_results/{gazebo_env}/{gazebo_env}_{repeat_count}.csv'

model_path = f"/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/RL/models/{gazebo_env}/{algo}/{algo}_{repeat_count}.pth"

def train_model():
    # 合并 CSV 文件
    combine_csv(folder_path, combined_output_path)
    
    # 读取数据
    robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroids = read_csv(combined_output_path)

    # 初始化 Agent
    agent = Agent(model_path, algo, gazebo_env, gamma, learning_rate, epsilon, epsilon_min, epsilon_decay,
                  save_interval, epochs, batch_size, penalty,
                  robot_positions, robot_orientations,
                  centroid_records, info_gain_records, best_centroids)

    # 训练模型
    agent.train()

    # 保存训练损失的可视化
    agent.save_plot()

def test_model():
    # 读取数据
    robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroids = read_csv(combined_output_path)

    # 加载 Agent
    agent = Agent(model_path, algo, gazebo_env, gamma, learning_rate, epsilon, epsilon_min, epsilon_decay,
                  save_interval, epochs, batch_size, penalty,
                  robot_positions, robot_orientations,
                  centroid_records, info_gain_records, best_centroids)

    agent.load_model()

    # 输出每一行的预测结果
    for i in range(len(robot_positions)):
        predicted_centroid, max_info_gain_centroid_idx = agent.predict_centroid(
            robot_positions[i], robot_orientations[i], centroid_records[i], info_gain_records[i])
        print(f"The centroid with the highest information gain for row {i+1} is {predicted_centroid} Index: {max_info_gain_centroid_idx}")

if __name__ == "__main__":
    train_model()
    test_model()
