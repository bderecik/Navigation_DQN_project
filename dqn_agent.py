import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork


# Hyperparameters
REPLAY_BUFFER_SIZE = int(2e5)
BATCH_SIZE = 64
GAMMA = 0.95
TAU = 1e-3  # Soft update of target parameters
LEARNING_RATE = 5e-4
UPDATE_EVERY = 4

# Uses GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, hidden_layer_input_size=64, 
                 hidden_layer_output_size=64, softmax_output=False, use_ddqn=False):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layer_input_size (int): Dimension of hidden layer input
            hidden_layer_output_size (int): Dimension of hidden layer output
            softmax_output (bool): Use softmax activation in the output layer
            use_ddqn (bool): Use Double DQN instead of vanilla DQN
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.use_ddqn = use_ddqn

        # Q-Network
        self.qnetwork_local = QNetwork(
            state_size, action_size, seed, hidden_layer_input_size, 
            hidden_layer_output_size, softmax_output).to(device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, seed, hidden_layer_input_size, 
            hidden_layer_output_size, softmax_output).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LEARNING_RATE)

        # Replay memory
        self.memory = ReplayBuffer(action_size, REPLAY_BUFFER_SIZE, BATCH_SIZE, seed)
        self.time_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience to replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.time_step = (self.time_step + 1) % UPDATE_EVERY
        if self.time_step == 0:
            # Need enough samples in memory
            if len(self.memory) > BATCH_SIZE:
                # Get random subset and learn
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        if self.use_ddqn:
            # Double DQN
            non_final_next_states = next_states * (1 - dones)
            _, next_state_actions = self.qnetwork_local(non_final_next_states).max(1, keepdim=True)  # gets the actions themselves, not their output value
            next_Q_targets = self.qnetwork_target(non_final_next_states).gather(1, next_state_actions)
            target_Q = rewards + (gamma * next_Q_targets * (1 - dones))
            expected_Q = self.qnetwork_local(states).gather(1, actions)
        else:
            # Vanilla DQN
            next_max_a = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            target_Q = rewards + (gamma * next_max_a * (1 - dones))       # (1 - dones) ignores the actions that ended the game
            expected_Q = self.qnetwork_local(states).gather(1, actions)
        
        # Compute and minimize the loss
        loss = F.mse_loss(expected_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
