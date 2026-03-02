import os
import numpy as np
import torch
import pickle
from code.utils.env import RampMeterEnv
from code.training.qlearning import QLearningAgent
from code.training.dqn import DQNAgent
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

def train_agent(env: RampMeterEnv, agent, episodes: int, max_steps: int = 3600) -> List[float]:
    """Train the agent and return episode rewards"""
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.train(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")
        
    return episode_rewards

def evaluate_agent(env: RampMeterEnv, agent, episodes: int = 5, max_steps: int = 3600) -> List[float]:
    """Evaluate the trained agent"""
    evaluation_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Get action (no exploration during evaluation)
            if isinstance(agent, QLearningAgent):
                agent.epsilon = 0
                action = agent.get_action(state)
            else:  # DQNAgent
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    action = agent.policy_net(state_tensor).argmax().item()
            
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            print(f"\rStep {step + 1}, Action: {'Green' if action == 1 else 'Red'}, "
                  f"Reward: {reward:.2f}, Avg Speed: {info['avg_speed']:.2f}", end="")
            
            state = next_state
            time.sleep(0.1)  # Slow down visualization
            
            if done:
                break
                
        print(f"\nEpisode {episode + 1} finished with total reward: {total_reward:.2f}")
        evaluation_rewards.append(total_reward)
        
    return evaluation_rewards

def plot_training_results(qlearning_rewards: List[float], dqn_rewards: List[float]):
    """Plot training results for both algorithms"""
    plt.figure(figsize=(10, 6))
    plt.plot(qlearning_rewards, label='Q-Learning')
    plt.plot(dqn_rewards, label='DQN')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig('training_results.png')
    plt.close()

def main():
    # Environment configuration
    config = {
        'sumo_binary': 'sumo',  # Use sumo-gui for visualization
        'net_file': 'config/highway.net.xml',
        'route_file': 'config/highway.rou.xml',
        'gui': False
    }
    
    # Training parameters
    EPISODES = 100
    MAX_STEPS = 3600
    
    # Initialize environment
    env = RampMeterEnv(config)
    state_size = env.observation_space_size
    action_size = env.action_space_size
    
    # Initialize agents
    qlearning_agent = QLearningAgent(state_size, action_size)
    dqn_agent = DQNAgent(state_size, action_size)
    
    # Train Q-Learning agent
    print("\nTraining Q-Learning Agent...")
    config['sumo_binary'] = 'sumo'  # Use headless mode for training
    env = RampMeterEnv(config)
    qlearning_rewards = train_agent(env, qlearning_agent, EPISODES, MAX_STEPS)
    
    # Train DQN agent
    print("\nTraining DQN Agent...")
    dqn_rewards = train_agent(env, dqn_agent, EPISODES, MAX_STEPS)
    
    # Plot results
    plot_training_results(qlearning_rewards, dqn_rewards)
    
    # Save trained agents
    torch.save(dqn_agent.policy_net.state_dict(), 'dqn_model.pth')
    with open('qlearning_table.pkl', 'wb') as f:
        pickle.dump(qlearning_agent.q_table, f)
    
    # Evaluate agents
    print("\nEvaluating Q-Learning Agent...")
    config['sumo_binary'] = 'sumo-gui'  # Use GUI for evaluation
    env = RampMeterEnv(config)
    qlearning_eval = evaluate_agent(env, qlearning_agent)
    
    print("\nEvaluating DQN Agent...")
    dqn_eval = evaluate_agent(env, dqn_agent)
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Q-Learning Average Reward: {np.mean(qlearning_eval):.2f}")
    print(f"DQN Average Reward: {np.mean(dqn_eval):.2f}")

if __name__ == "__main__":
    main()
