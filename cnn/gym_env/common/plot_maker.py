import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
import os
import sys
import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_tensorboard_data(log_path, tag):
    """
    Load data from TensorBoard log file for a specific tag.
    
    Args:
        log_path (str): Path to the TensorBoard log file
        tag (str): The tag to extract (e.g., 'rollout/ep_rew_mean')
    
    Returns:
        tuple: (steps, values) arrays
    """
    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()  # Load all data
    
    if tag not in ea.Tags()['scalars']:
        raise ValueError(f"Tag {tag} not found in log file. Available tag: {ea.Tags()['scalars']}")
    
    # Get all scalar events for the tag
    events = ea.Scalars(tag)
    
    # Extract steps and values
    steps = np.array([event.step for event in events])
    values = np.array([event.value for event in events])
    
    return steps, values

def plot_tensorboard_data(log_path, tag, title=None, smoothing=95,file_name=None):
    COLOR = '#da0dd3ff'
    """
    Create a plot from TensorBoard log data for specified tag.
    
    Args:
        log_path (str): Path to the TensorBoard log file
        tag (list): List of tag to plot
        title (str, optional): Plot title
        smoothing (int, optional): Window size for moving average smoothing
    """
    plt.figure(figsize=(12, 6))
    
    steps, values = load_tensorboard_data(log_path, tag)
    
    # Convert steps to millions
    steps_millions = steps / 1e6
    
    if smoothing > 0:
        # Apply moving average smoothing
        kernel = np.ones(smoothing) / smoothing
        values_smooth = np.convolve(values, kernel, mode='valid')
        steps_smooth = steps_millions[smoothing-1:]
        plt.plot(steps_smooth, values_smooth, label=tag, color=COLOR, linewidth=1.5)
    else:
        plt.plot(steps_millions, values, label=tag, color=COLOR, linewidth=1.5)

    plt.xlim(left=0,right=20.5)  # Set x-axis range
    plt.xlabel('Timesteps (Million)')
    plt.ylim(bottom=0)  # Set y-axis range, adjust these values as needed
    plt.ylabel('Value')
    if title:
        plt.title(title)
    plt.legend(facecolor='white', edgecolor='gray', framealpha=0.8)
    plt.legend()
    plt.grid(True)
    
    # Format x-axis to show clean numbers with 2M increments
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))  # Set ticks every 2 million steps
    
    plt.show()
    if file_name is not None:
        plt.savefig(file_name)

def plot_multiple_tensorboard_data(log_path, tags, title=None, smoothing=95, file_name=None):
    # Define colors for multiple lines
   
    COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c") 
 
    plt.figure(figsize=(12, 6))
    for idx, tag in enumerate(tags):
        steps, values = load_tensorboard_data(log_path, tag)
        
        # Convert steps to millions
        steps_millions = steps / 1e6
        
        if smoothing > 0:
            # Apply moving average smoothing
            kernel = np.ones(smoothing) / smoothing
            values_smooth = np.convolve(values, kernel, mode='valid')
            steps_smooth = steps_millions[smoothing-1:]
            plt.plot(steps_smooth, values_smooth, label=tag, color=COLORS[idx % len(COLORS)], linewidth=1.5)
        else:
            plt.plot(steps_millions, values, label=tag, color=COLORS[idx % len(COLORS)], linewidth=1.5)

    plt.xlim(left=0, right=20.5)  # Set x-axis range
    plt.xlabel('Timesteps (Million)')
    plt.ylabel('Value')
    if title:
        plt.title(title)
    plt.legend(facecolor='white', edgecolor='gray', framealpha=0.8)
    plt.grid(True)
    
    # Format x-axis to show clean numbers with 2M increments
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))  # Set ticks every 2 million steps
    
    if file_name is not None:
        plt.savefig(file_name)
    plt.show()


def compare_algorithms_plot(log_paths, labels, tag, title=None, smoothing=95, save_figures=False, algorithm="ERROR"):
    MAIN_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c") 
    OTHER_COLORS = ("#6baed6", "#fdae6b", "#98df8a")
    if(algorithm == 'SingleChannel'):
        COLORS = OTHER_COLORS
    else: 
        COLORS = MAIN_COLORS 

    if(labels[0] == "Single-Channel"):
        if algorithm == "DQN":
            COLORS = (OTHER_COLORS[0], MAIN_COLORS[0])
        if algorithm == "PPO":
            COLORS = (OTHER_COLORS[1], MAIN_COLORS[1])
        if algorithm == "A2C":
            COLORS = (OTHER_COLORS[2], MAIN_COLORS[2])
        
    # plt.clf()
    plt.figure(figsize=(12, 6))
    for idx, log_path in enumerate(log_paths):
        steps, values = load_tensorboard_data(log_path, tag)
        # Convert steps to millions
        steps_millions = steps / 1e6
        
        if smoothing > 0:
            # Apply moving average smoothing
            kernel = np.ones(smoothing) / smoothing
            values_smooth = np.convolve(values, kernel, mode='valid')
            steps_smooth = steps_millions[smoothing-1:]
            plt.plot(steps_smooth, values_smooth, label=labels[idx], color=COLORS[idx], linewidth=1.7)
        else:
            plt.plot(steps_millions, values, label=labels[idx], color=COLORS[idx], linewidth=1.7)

    plt.xlim(left=0, right=20.5)  # Set x-axis range
    plt.xlabel('Timesteps (Million)')
    plt.ylabel('Value')
    if title:
        plt.title(title)
    else:
        plt.title(tag)
    plt.legend(facecolor='white', edgecolor='gray', framealpha=0.8)
    plt.grid(True)
    
    # Format x-axis to show clean numbers with 2M increments
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(2))
    
    if save_figures:  
        plt.savefig('gym_env/plots/results/' + algorithm + '/' + title + '.png')
    # plt.show()


def compare_algorithms_channels(PPO=True, DQN=True, A2C=True, save_figures=False):

    labels = ("Single-Channel", "Multi-Channel")
    
    tags = ("rollout/mean_ep_coverage", "rollout/ep_rew_mean", 'rollout/mean_ep_coverage', 'rollout/success_rate', 'eval/mean_ep_length', 'eval/mean_reward')
    
    if PPO:
        algorithm = "PPO"
        log_paths = ("gym_env/logs/final/PPO_SingleChannel_Final/level_1_0","gym_env/logs/final/PPO_MultiChannel_Final/level_1_0")
        for tag in tags:
            compare_algorithms_plot(log_paths, labels, tag, title=tag, smoothing=95, save_figures=save_figures, algorithm=algorithm)
        print("PPO comparison done.")

    if DQN:
        algorithm = "DQN"
        log_paths = ("gym_env/logs/final/DQN_SingleChannel_Final/level_1_0","gym_env/logs/final/DQN_MultiChannel_Final/level_1_0")
        for tag in tags:
            compare_algorithms_plot(log_paths, labels, tag, title=tag, smoothing=95, save_figures=save_figures, algorithm=algorithm)
        print("DQN comparison done.")
    if A2C:    
        algorithm = "A2C"
        log_paths = ("gym_env/logs/final/A2C_SingleChannel_Final/level_1_0","gym_env/logs/final/A2C_MultiChannel_Final/level_1_0")
        for tag in tags:
            compare_algorithms_plot(log_paths, labels, tag, title=tag, smoothing=95, save_figures=save_figures, algorithm=algorithm)
        print("A2C comparison done.")
    

def compare_algorithms(SingleChannel=True, MultiChannel=True, save_figures=False):

    labels = ("DQN", "PPO", "A2C")
    tags = ("rollout/mean_ep_coverage", "rollout/ep_rew_mean", 'rollout/mean_ep_coverage', 'rollout/success_rate', 'eval/mean_ep_length', 'eval/mean_reward')
    
    if SingleChannel:
        algorithm = "SingleChannel"
        log_paths = ("gym_env/logs/final/DQN_SingleChannel_Final/level_1_0", "gym_env/logs/final/PPO_SingleChannel_Final/level_1_0", "gym_env/logs/final/A2C_SingleChannel_Final/level_1_0")
        for tag in tags:
            compare_algorithms_plot(log_paths, labels, tag, title=tag, smoothing=95, save_figures=save_figures, algorithm=algorithm)
        print("Single channel comparison done.")

    if MultiChannel:
        algorithm = "MultiChannel"
        log_paths = ("gym_env/logs/final/DQN_MultiChannel_Final/level_1_0", "gym_env/logs/final/PPO_MultiChannel_Final/level_1_0", "gym_env/logs/final/A2C_MultiChannel_Final/level_1_0")
        for tag in tags:
            compare_algorithms_plot(log_paths, labels, tag, title=tag, smoothing=95, save_figures=save_figures, algorithm=algorithm)
        print("Multi channel comparison done.")

def main():

    compare_algorithms_channels(PPO=True, DQN=True, A2C=True, save_figures=True)
    compare_algorithms(MultiChannel=True, SingleChannel=True, save_figures=True)
    
if __name__ == "__main__":
    main()