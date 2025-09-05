# Maze Exploration with Reinforcement Learning

This repository contains the implementation and evaluation of reinforcement learning algorithms for autonomous maze exploration in a custom Gymnasium environment. The project compares Proximal Policy Optimization (PPO), Deep Q-Network (DQN), and Advantage Actor-Critic (A2C) algorithms using curriculum learning, with both single-channel and multi-channel convolutional neural network (CNN) architectures for feature extraction.

## Project Overview

The goal of this thesis project is to develop and evaluate RL agents capable of efficiently exploring unknown mazes. Agents learn to navigate, avoid obstacles, and maximize coverage of explorable areas while adhering to a curriculum of increasing difficulty levels. The environment supports both single-channel (agent view) and multi-channel (explored areas, obstacles, agent position, target) observations.

Key components:
- Custom maze exploration environment (maze_exploration_env.py)
- Curriculum-based training scripts for PPO (train_ppo_curriculum.py), DQN (train_dqn_curriculum.py), and A2C (train_a2c_curriculum.py)
- Advanced CNN feature extractor (advanced_cnn.py)
- Evaluation and plotting utilities (evaluate_models.py, plot_maker.py, evaluation_plot_maker.py)

## Features

- **Curriculum Learning**: Progressive difficulty increase across maze sizes and densities
- **Multi-Algorithm Comparison**: PPO, DQN, A2C with hyperparameter tuning for single/multi-channel inputs
- **Custom Environment**: Procedural maze generation with exploration rewards and target-based incentives
- **Visualization**: TensorBoard logging and matplotlib-based plotting for training metrics and results
- **Evaluation Metrics**: Coverage percentage, success rate, episode length, and standard deviations

## Installation

1. Clone the repository and navigate to the project directory.
2. Install dependencies using pip:

   ```bash
   pip install -r gym_env/requirements.txt
   ```

   Key dependencies include:
   - gymnasium==1.2.0
   - stable-baselines3==2.7.0
   - torch==2.8.0
   - numpy, matplotlib, seaborn, pandas, opencv-python, tensorboard

3. Ensure Python 3.11+ is installed, as the code uses features from that version.

## Usage

### Training

Run the training scripts for each algorithm. For example, to train PPO with multi-channel input:

```bash
cd gym_env
python training/train_ppo_curriculum.py
```

Adjust [`CHANNELS`](gym_env/training/train_dqn_curriculum.py ) in the script (1 for single-channel, 4 for multi-channel). Training logs are saved to `logs/` and models to `models/`.

### Evaluation

Evaluate trained models using:

```bash
python evaluation/evaluate_models.py
```

This script loads final models and runs 1000 episodes, saving results to JSON files in `evaluation/final_prediction_logs/`.

### Plotting

Generate comparison plots for training metrics and evaluation results:

```bash
python common/plot_maker.py
python common/evaluation_plot_maker.py
```

Plots are saved to `plots/results/`.

## Results

Evaluation results (averaged over 1000 episodes) for the final models:

### Single-Channel Input
- **DQN**: Mean Coverage: 0.677, Success Rate: 0.0%, Mean Success Episode Length: 0
- **PPO**: Mean Coverage: 0.136, Success Rate: 0.0%, Mean Success Episode Length: 0
- **A2C**: Mean Coverage: 0.468, Success Rate: 0.0%, Mean Success Episode Length: 0

### Multi-Channel Input
- **DQN**: Mean Coverage: 0.957, Success Rate: 74.2%, Mean Success Episode Length: 30.66
- **PPO**: Mean Coverage: 0.490, Success Rate: 0.0%, Mean Success Episode Length: 0
- **A2C**: Mean Coverage: 0.430, Success Rate: 0.0%, Mean Success Episode Length: 0

Multi-channel DQN achieves the highest performance, with significant improvements in coverage and success rate compared to single-channel variants.

## Project Structure

```
gym_env/
├── common/
│   ├── evaluation_plot_maker.py  # Evaluation result plotting
│   ├── maze_generator.py         # Procedural maze generation
│   └── plot_maker.py             # Training metric plotting
├── environment/
│   └── maze_exploration_env.py   # Custom Gym environment
├── evaluation/
│   ├── evaluate_models.py        # Model evaluation script
│   └── final_prediction_logs/    # JSON results
├── logs/                         # TensorBoard logs
├── models/                       # Saved models
├── neural_networks/
│   └── advanced_cnn.py           # CNN feature extractor
├── plots/                        # Generated plots
├── training/
│   ├── train_a2c_curriculum.py   # A2C training
│   ├── train_dqn_curriculum.py   # DQN training
│   └── train_ppo_curriculum.py   # PPO training
├── requirements.txt               # Dependencies
└── .gitignore                     # Git ignore rules
```

## Configuration

- Maze size: 10x10, density 0.85, max steps 250
- Curriculum: Single level (configurable in scripts)
- Hyperparameters: Tuned per algorithm and channel type (e.g., learning rates, batch sizes)

## License

This project is for academic purposes but is free of use for everyone.

For questions or issues, refer to the code comments or contact the author.