import os
import torch
import pickle
import numpy as np
from code.utils.env import RampMeterEnv
from code.training.qlearning import QLearningAgent
from code.training.dqn import DQNAgent, DQN
import time

def load_model(model_type='dqn'):
    """Load the trained model"""
    if model_type.lower() == 'dqn':
        # Initialize DQN with same parameters as training
        state_size = 6  # From RampMeterEnv
        action_size = 2
        agent = DQNAgent(state_size, action_size)
        
        # Load trained weights
        if os.path.exists('dqn_model.pth'):
            agent.policy_net.load_state_dict(torch.load('dqn_model.pth'))
            print("Loaded DQN model successfully")
        else:
            raise FileNotFoundError("DQN model file not found. Please train the model first.")
            
    else:  # Q-Learning
        # Initialize Q-Learning agent
        state_size = 6
        action_size = 2
        agent = QLearningAgent(state_size, action_size)
        
        # Load Q-table
        if os.path.exists('qlearning_table.pkl'):
            with open('qlearning_table.pkl', 'rb') as f:
                agent.q_table = pickle.load(f)
            print("Loaded Q-Learning table successfully")
        else:
            raise FileNotFoundError("Q-Learning table file not found. Please train the model first.")
    
    return agent

def run_simulation(model_type='dqn', simulation_time=3600, delay=0.1):
    """Run SUMO simulation with the trained model"""
    # Environment configuration
    config = {
        'sumo_binary': 'sumo-gui',  # Use GUI for visualization
        'net_file': 'config/highway.net.xml',
        'route_file': 'config/highway.rou.xml',
        'gui': True
    }
    
    # Initialize environment
    env = RampMeterEnv(config)
    
    # Load trained agent
    agent = load_model(model_type)
    
    # Initialize metrics
    total_reward = 0
    total_waiting_time = 0
    total_vehicles = 0
    average_speeds = []
    
    # Run simulation
    state = env.reset()
    print("\nStarting simulation with trained model...")
    print("Green light = Allow ramp vehicles")
    print("Red light = Stop ramp vehicles")
    
    for step in range(simulation_time):
        # Get action from trained model
        if model_type.lower() == 'dqn':
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.policy_net(state_tensor).argmax().item()
        else:  # Q-Learning
            agent.epsilon = 0  # No exploration during running
            action = agent.get_action(state)
        
        # Take action in environment
        next_state, reward, done, info = env.step(action)
        
        # Update metrics
        total_reward += reward
        total_waiting_time += info['waiting_time']
        total_vehicles += info['vehicles_passed']
        average_speeds.append(info['avg_speed'])
        
        # Print status
        print(f"\rStep: {step + 1}/{simulation_time} | "
              f"Action: {'Green' if action == 1 else 'Red'} | "
              f"Reward: {reward:.2f} | "
              f"Avg Speed: {info['avg_speed']:.2f} km/h | "
              f"Waiting Time: {info['waiting_time']:.2f} s | "
              f"Vehicles Passed: {info['vehicles_passed']}", end="")
        
        state = next_state
        time.sleep(delay)  # Add delay for better visualization
        
        if done:
            break
    
    # Print final statistics
    print("\n\nSimulation Complete!")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Average Speed: {np.mean(average_speeds):.2f} km/h")
    print(f"Total Vehicles Served: {total_vehicles}")
    print(f"Average Waiting Time: {total_waiting_time/max(1, total_vehicles):.2f} seconds")
    
    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run trained ramp metering model')
    parser.add_argument('--model', type=str, choices=['dqn', 'qlearning'], 
                      default='dqn', help='Model type to use (dqn or qlearning)')
    parser.add_argument('--time', type=int, default=3600,
                      help='Simulation time in steps')
    parser.add_argument('--delay', type=float, default=0.1,
                      help='Delay between steps for visualization')
    
    args = parser.parse_args()
    
    try:
        run_simulation(args.model, args.time, args.delay)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run train_and_evaluate.py first to train the models.")
