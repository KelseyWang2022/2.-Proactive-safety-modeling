import numpy as np
from typing import Dict, List
import pickle

class QLearningAgent:
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Initialize Q-table
        self.q_table = {}
        
    def get_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        state_key = self._get_state_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
            
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state_key])
            
    def train(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool):
        """Update Q-values using Q-learning update rule"""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
            
        # Q-learning update
        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q * (1 - done) - current_q)
        self.q_table[state_key][action] = new_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def _get_state_key(self, state: np.ndarray) -> str:
        """Convert continuous state to discrete key for Q-table"""
        # Discretize state space for Q-table lookup
        discretized_state = np.round(state, decimals=1)
        return str(discretized_state.tolist())
        
    def save(self, filepath: str):
        """Save Q-table to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
            
    def load(self, filepath: str):
        """Load Q-table from file"""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
