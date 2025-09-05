import json
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_metrics(json_paths, labels, save_folder=None, approach='MultiChannel'):
    """
    Compare metrics from multiple JSON files
    
    Args:
        json_paths (list): List of paths to JSON files
        labels (list): List of labels for each algorithm
    """
    # Load JSON data
    data = []
    for path in json_paths:
        with open(path, 'r') as f:
            data.append(json.load(f))
    
    # Colors
    MAIN_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c") 
    OTHER_COLORS = ("#6baed6", "#fdae6b", "#98df8a")
    if(approach == 'MultiChannel'):
        COLORS = MAIN_COLORS
    else: 
        COLORS = OTHER_COLORS
    width = 0.8 / len(json_paths)  # Adjust width based on number of bars
    
    # Plot 1: Coverage
    plt.figure(figsize=(8, 6))
    x = np.arange(1)
    for i in range(len(data)):
        offset = width * (i - (len(data)-1)/2)  # Center the bars
        plt.bar(x + offset, data[i]['mean_coverage'], width, 
                yerr=data[i]['std_coverage'], capsize=5,
                color=COLORS[i], label=labels[i])
    plt.ylabel('Coverage Ratio')
    plt.ylim(top=1.0)
    plt.title('Mean Coverage Percentage')
    plt.xticks([])
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_folder is not None:
        plt.savefig(f'{save_folder}/{approach}/evaluation/mean_coverage.png')
    plt.show()
    
    # Plot 2: Success Rate
    plt.figure(figsize=(8, 6))
    for i in range(len(data)):
        offset = width * (i - (len(data)-1)/2)
        plt.bar(x + offset, data[i]['success_rate']/100.0, width,
                color=COLORS[i], label=labels[i])
    plt.ylabel('Success Rate')
    plt.ylim(top=1.0)
    plt.title('Mean Success Rate')
    plt.xticks([])
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_folder is not None:
        plt.savefig(f'{save_folder}/{approach}/evaluation/success_rate.png')
    plt.show()
    
    
    # Plot 3: Episode Length
    plt.figure(figsize=(8, 6))
    for i in range(len(data)):
        offset = width * (i - (len(data)-1)/2)
        plt.bar(x + offset, data[i]['mean_success_episode_length'], width,
                yerr=data[i]['std_success_episode_length'], capsize=5,
                color=COLORS[i], label=labels[i])
    plt.ylabel('Steps')
    plt.ylim(top=250)
    plt.title('Mean Episode Length on Successes')
    plt.xticks([])
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_folder is not None:
        plt.savefig(f'{save_folder}/{approach}/evaluation/mean_success_episode_length.png')
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
            COLORS = (MAIN_COLORS[0] + OTHER_COLORS[0])
        if algorithm == "PPO":
            COLORS = (MAIN_COLORS[1] + OTHER_COLORS[1])
        if algorithm == "A2C":
            COLORS = (MAIN_COLORS[2] + OTHER_COLORS[2])
    
    print(COLORS)
    print(algorithm + labels[0])

def main():
    json_paths = [
        "gym_env/evaluation/final_prediction_logs/SingleChannel/DQN.json",
        "gym_env/evaluation/final_prediction_logs/SingleChannel/PPO.json",
        "gym_env/evaluation/final_prediction_logs/SingleChannel/A2C.json"
    ]
    labels = ["DQN", "PPO", "A2C"]
    plot_comparison_metrics(json_paths, labels, save_folder="gym_env/plots/results", approach='SingleChannel')

    json_paths = [
        "gym_env/evaluation/final_prediction_logs/MultiChannel/DQN.json",
        "gym_env/evaluation/final_prediction_logs/MultiChannel/PPO.json",
        "gym_env/evaluation/final_prediction_logs/MultiChannel/A2C.json"
    ]
    labels = ["DQN", "PPO", "A2C"]
    plot_comparison_metrics(json_paths, labels, save_folder="gym_env/plots/results", approach='MultiChannel')

if __name__ == "__main__":
    main()