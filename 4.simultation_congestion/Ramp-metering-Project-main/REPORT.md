# Highway Ramp Metering Reinforcement Learning Project Report

## 1. Required Packages
```
# Core Dependencies
torch>=1.9.0      # Deep Learning framework for DQN
numpy>=1.19.5     # Numerical computations
matplotlib>=3.3.4 # Plotting and visualization
sumo>=1.8.0       # Traffic simulation environment
pickle            # Model saving/loading (built-in)

# Optional Dependencies
tqdm              # Progress bars for training
traci>=1.8.0      # SUMO traffic simulation API
```

## 2. Project Structure

### Main Files
```
RL-Project/
├── train_and_evaluate.py    # Main training script
├── run_trained_model.py     # Script to run trained models
├── requirements.txt         # Package dependencies
└── README.md               # Project documentation
```

### Code Organization
```
code/
├── training/
│   ├── qlearning.py        # Q-Learning implementation
│   └── dqn.py             # Deep Q-Network implementation
├── utils/
│   └── env.py             # SUMO environment wrapper
└── __init__.py
```

### Configuration
```
config/
├── highway.net.xml         # Highway network definition
└── highway.rou.xml         # Traffic route definition
```

### Saved Models and Results
```
./
├── dqn_model.pth          # Trained DQN model weights
├── qlearning_table.pkl    # Q-Learning table
└── training_results.png   # Training performance plots
```

## 3. Execution Instructions

### Training New Models
```bash
# Train both Q-Learning and DQN models
python train_and_evaluate.py

# Training parameters can be modified in the script:
# - EPISODES: Number of training episodes
# - MAX_STEPS: Maximum steps per episode
```

### Running Trained Models
```bash
# Run DQN model
python run_trained_model.py --model dqn

# Run Q-Learning model
python run_trained_model.py --model qlearning

# Optional parameters:
# --simulation_time: Duration of simulation (default: 3600 steps)
# --delay: Visualization delay (default: 0.1s)
```

## 4. Evaluation and Results

### Training Metrics
The `training_results.png` plot shows:
- X-axis: Training episodes
- Y-axis: Total reward per episode
- Blue line: Q-Learning performance
- Orange line: DQN performance

### Performance Interpretation
1. **Q-Learning Performance**
   - Advantages:
     - Simpler implementation
     - More stable learning
     - Better for discrete state spaces
   - Limitations:
     - Limited scalability
     - Less precise in continuous spaces

2. **DQN Performance**
   - Advantages:
     - Better generalization
     - Handles continuous state spaces
     - More sophisticated feature learning
   - Limitations:
     - Longer training time
     - More hyperparameter tuning needed

### Key Metrics During Execution
The simulation provides real-time metrics:
- Average vehicle speed (km/h)
- Waiting time at ramps (seconds)
- Number of vehicles served
- Instantaneous rewards

### Visualization
During model execution (`run_trained_model.py`):
- SUMO-GUI shows traffic simulation
- Terminal displays real-time metrics
- Traffic light states (Red/Green) indicate ramp metering decisions

## 5. Model Parameters

### Q-Learning
- Learning rate (α): 0.1
- Discount factor (γ): 0.95
- Exploration rate (ε): 0.1
- State discretization: 6 dimensions

### DQN
- Neural Network: 3 layers (6→64→64→2)
- Learning rate: 0.001
- Batch size: 64
- Memory size: 10000
- Target network update: Every 100 steps
- Exploration decay: 0.995

## 6. Future Improvements
1. Implement Prioritized Experience Replay
2. Add A3C (Asynchronous Advantage Actor-Critic)
3. Include more traffic scenarios
4. Optimize hyperparameters
5. Add multi-ramp coordination
