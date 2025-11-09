# DRL_TRAINING/model.py

import torch
import torch.nn as nn

# Define the Q-Network architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """
        Initialize the Deep Q-Network.
        
        Args:
            state_size (int): The size of the input state vector (12 for us)
            action_size (int): The number of possible actions (8 for us)
        """
        super(QNetwork, self).__init__()
        
        # We will use a simple MLP (Multi-Layer Perceptron).
        # This network learns to map our 12 heuristic features
        # to 8 "Q-values" (the expected future reward for each action).
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),  # ReLU is a standard, fast activation function
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size) # The output layer has 8 neurons
        )

    def forward(self, x):
        """
        Defines the forward pass.
        
        Args:
            x (torch.Tensor): The input state vector(s).
        
        Returns:
            torch.Tensor: The Q-values for each of the 8 possible actions.
        """
        return self.net(x)