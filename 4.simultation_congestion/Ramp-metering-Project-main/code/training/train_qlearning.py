import os
import sys
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.env import RampMeterEnv
from qlearning import QLearningAgent

def train_qlearning(env: RampMeterEnv, agent: QLearningAgent, 
                   num_episodes: int = 1000, max_steps: int = 3600) -> List[float]:
    """Train Q-learning agent and return episode rewards"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Train agent
            agent.train(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        episode_rewards.append(total_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")
            
    return episode_rewards

def plot_training_results(rewards: List[float], save_path: str):
    """Plot and save training results"""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Q-learning Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Configuration
    config = {
        'sumo_binary': 'sumo',
        'net_file': '../../config/highway.net.xml',
        'route_file': '../../config/highway.rou.xml',
        'gui': False
    }
    
    # Create environment and agent
    env = RampMeterEnv(config)
    agent = QLearningAgent(
        state_size=env.observation_space_size,
        action_size=env.action_space_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0
    )
    
    # Training
    print("Starting Q-learning training...")
    rewards = train_qlearning(env, agent)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "../../data/models/qlearning"
    os.makedirs(save_dir, exist_ok=True)
    
    agent.save(os.path.join(save_dir, f"qlearning_model_{timestamp}.pkl"))
    plot_training_results(rewards, os.path.join(save_dir, f"qlearning_training_{timestamp}.png"))
    
    print("Training completed. Model and results saved.")
