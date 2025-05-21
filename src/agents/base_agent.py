"""
base_agent.py: 定义 Agent 抽象基类。
"""
import abc

class BaseAgent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def act(self, state):
        """给定游戏状态，输出动作 0:left,1:right,2:rotate,3:drop"""
        pass

    def learn(self, state, action, reward, next_state):
        """可选学习接口。"""
        return
