import os
import sys
import numpy as np
import torch
from typing import Dict, List
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.env import RampMeterEnv
from dqn import DQNAgent

def train_dqn(env: RampMeterEnv, agent: DQNAgent,
              num_episodes: int = 1000, max_steps: int = 3600,
              target_update_frequency: int = 10) -> List[float]:
    """Train DQN agent and return episode rewards"""
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
                
        # Update target network periodically
        if (episode + 1) % target_update_frequency == 0:
            agent.update_target_network()
            
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
    plt.title('DQN Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Configuration
    # config = {
    #     'sumo_binary': 'sumo',
    #     'net_file': '../../config/highway.net.xml',
    #     'route_file': '../../config/highway.rou.xml',
    #     'gui': False
    # }
    config = {
        "sumo_binary": "sumo",
        "gui": False,
        "sumocfg": "../../config/highway.sumocfg",
        "episode_seconds": 3600,
        "step_length": 0.1,

        # 你场景的真实ID
        "tl_id": "meter",

        # 主线：合流前(main1) 和 合流后(main3) 分开采样是最合理的
        "main_in_edges": ["main_1"],  # main1
        "main_out_edges": ["main_3"],  # main3

        # 合流区段（main2）建议单独作为“瓶颈段”监测
        # 你可以先放在 out 里一起算，也可以单独加一个字段（更推荐后者）
        # 先给你一个最省改动的写法：把 main2 也算到主线上
        "main_mid_edges": ["main_2"],  # main2（如果你用我给的 env.py，可扩展支持）

        # 匝道：上游和计量段都算匝道队列更真实
        "ramp_edges": ["ramp_0", "ramp_1"],  # ramp0+ramp1

        # 动作映射（你先确认 meter 有几个 phase）
        "action_to_phase": {0: 0, 1: 1},

        "desired_speed_kmh": 100.0,
        "queue_penalty_w": 0.1,
    }

    # Create environment and agent
    env = RampMeterEnv(config)
    agent = DQNAgent(
        state_size=env.observation_space_size,
        action_size=env.action_space_size,
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon=1.0,
        memory_size=10000,
        batch_size=64
    )
    
    # Training
    # print("Starting DQN training...")
    # rewards = train_dqn(env, agent)
    print("Starting DQN training...")
    try:
        rewards = train_dqn(env, agent, num_episodes=10, max_steps=50)  # 你现在的参数
    finally:
        env.close()  # <- 这一句非常关键，保证不管怎样都关干净

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "../../data/models/dqn"
    os.makedirs(save_dir, exist_ok=True)
    
    agent.save(os.path.join(save_dir, f"dqn_model_{timestamp}.pth"))
    plot_training_results(rewards, os.path.join(save_dir, f"dqn_training_{timestamp}.png"))
    
    print("Training completed. Model and results saved.")
