import sys
sys.path.append(
    '/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/RL')

from csv_handler import *
from agent import *


if __name__ == "__main__":
    gazebo_env = 'aws_bookstore'
    algo = 'dqn'
    repeat_count = 20
    model_name = f'{algo}_{str(repeat_count)}'
    completed_time_path_csv = f'/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/completed_time/{gazebo_env}/{algo}/{model_name}.csv'
    print(calculate_average_from_csv(completed_time_path_csv))
