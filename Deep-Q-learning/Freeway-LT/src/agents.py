from abc import ABC
from abc import abstractmethod
from collections import defaultdict

import numpy as np

class Agent(ABC):

    @abstractmethod
    def __init__(self, **params):
        pass
    
    @abstractmethod
    def act(self, state):
        pass


class Baseline(Agent):

    def __init__(self):
        pass
    
    def act(self, state):
        return 1  # Always move up!

class QLearning(Agent):
    def __init__(self, gamma: float, available_actions: int, N0: float):
        self.gamma = gamma
        self.available_actions = available_actions
        self.N0 = N0

        self.Q = defaultdict(lambda: np.zeros(self.available_actions))
        self.state_visits = defaultdict(lambda: 0)
        self.Nsa = defaultdict(lambda: defaultdict(lambda: 0))

        self.epsilon = 0
        self.alpha = 0

    def act(self, state):
        self.epsilon = self.N0 / (self.N0 + self.state_visits[state])
        if np.random.choice(np.arange(2), p=[1 - self.epsilon, self.epsilon]):
            action = np.random.choice(self.available_actions)  # Explore!
        elif self.Q[state].max() == 0.0 and self.Q[state].min() == 0.0:
            action = 1  # Bias toward going forward
        else:
            action = self.Q[state].argmax()  # Greedy action

        self.state_visits[state] += 1
        self.Nsa[state][action] += 1

        return action

    def getEpsilon(self):
        return self.epsilon
    
    def getAlpha(self):
        return self.alpha
    
    def update_Q(self, old_state, new_state, action, reward):
        self.alpha = (1 / self.Nsa[old_state][action])
        self.Q[old_state][action] += self.alpha * (reward + (self.gamma * self.Q[new_state].max()) - self.Q[old_state][action])

    def print_parameters(self):
        print(f"gamma = {self.gamma}")
        print(f"available_actions = {self.available_actions}")
        print(f"N0 = {self.N0}")

