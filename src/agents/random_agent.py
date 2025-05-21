"""
random_agent.py: 纯随机策略实现
"""
import random
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def act(self, state):
        """随机选择一个动作"""
        return random.choice([0, 1, 2, 3])
