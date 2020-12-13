import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """DQN Network Model"""

    def __init__(self, state_size, action_size, seed, hidden_layer_input_size=64, 
                 hidden_layer_output_size=64, softmax_output=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layer_input_size (int): Dimension of hidden layer input
            hidden_layer_output_size (int): Dimension of hidden layer output
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.softmax_output = softmax_output

        self.fc1 = nn.Linear(state_size, hidden_layer_input_size)
        self.fc2 = nn.Linear(hidden_layer_input_size, hidden_layer_output_size)
        self.fc3 = nn.Linear(hidden_layer_output_size, action_size)

    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if self.softmax_output:
            x = F.softmax(self.fc3(x), dim=0)
        else:
            x = self.fc3(x)

        return x
