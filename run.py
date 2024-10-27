import os
import time
import subprocess

# 定义需要启动的launch文件
launch_files = ['train_script.launch']

def kill_all_ros_nodes():
    node_list = subprocess.check_output(['rosnode', 'list']).decode('utf-8').splitlines()
    for node in node_list:
        print(f"Killing node: {node}")
        subprocess.run(['rosnode', 'kill', node])

def restart_ros():
    # 终止 ROS Master
    print("Shutting down ROS Master...")
    subprocess.run(['pkill', '-f', 'roscore'])
    time.sleep(5)
    
    # 启动 ROS Master
    print("Starting ROS Master...")
    subprocess.Popen(['roscore'])
    time.sleep(5)

# 定义每个launch文件需要运行的时间（秒）
duration = 800
# 定义需要重复运行的次数
num_runs = 4

for i in range(num_runs):
    print(f'Run {i+1}/{num_runs}')
    
    # 在运行每个launch文件前，重启ROS Master
    restart_ros()

    time.sleep(20)
    
    # 在第一个终端中运行第一个launch文件
    os.system('gnome-terminal -- roslaunch decision_maker train_script.launch')

    time.sleep(10)

    # 等待一定时间
    time.sleep(duration)
    
    # 杀死所有ROS节点
    kill_all_ros_nodes()
    print("ALL ROS nodes have been killed")

    time.sleep(10)

    os.system('pkill -f "decision_maker"')


    print(f'Finished run {i+1}/{num_runs}\n')
    time.sleep(30)
