import gymnasium as gym
import gymnasium.wrappers as gym_wrap
import math
import random
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class ReplayBuffer:
    
    def __init__(self, capacity, state_dim):
        """
        Initializes the replay buffer.
        Args:
            capacity (int): Maximum number of transitions to store in the buffer.
            state_dim (int or tuple): Shape of a single state.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim

    def push(self, state, action, reward, next_state, done):
        """
        Stores a transition in the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a random batch of experiences from the replay buffer and processes them into tensors.
    
        Args:
            batch_size (int): Number of transitions to sample from the buffer.
            device (torch.device): Device on which to place the tensors (e.g., 'cpu' or 'cuda').
    
        Returns:
            Tuple[torch.Tensor]: A tuple of tensors: (states, actions, rewards, next_states, dones),
                                 all moved to the specified device.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([torch.tensor(state) for state in states]).float().to(device)
        next_states = torch.stack([torch.tensor(next_state) for next_state in next_states]).float().to(device)
        
        actions = torch.stack([torch.tensor(a) for a in actions]).float().to(device)
                                                                                     
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        
        return len(self.buffer)

class CriticNetwork(nn.Module):
    def __init__(self, n_states, fc1_dims, fc2_dims, n_actions):
        super(CriticNetwork, self).__init__()
        self.n_states = n_states
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.n_states, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims + self.n_actions, self.fc2_dims)

        self.ln1 = nn.LayerNorm(self.fc1_dims) 
        self.ln2 = nn.LayerNorm(self.fc2_dims)
        
        self.q = nn.Linear(self.fc2_dims, 1)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0]) 
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003  
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

    def forward(self, state, action):
        state_rep = self.fc1(state)
        state_rep = self.ln1(state_rep)
        state_rep = F.relu(state_rep)
        
        
        state_action_value = torch.cat([state_rep, action], dim=1) 
        state_action_value = self.fc2(state_action_value)          
        state_action_value = self.ln2(state_action_value)
        state_action_value = F.relu(state_action_value)
        
        q = self.q(state_action_value)
        
        return q

class ActorNetwork(nn.Module):
    def __init__(self, n_states, fc1_dims, fc2_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.n_states = n_states
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.n_states, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.ln1 = nn.LayerNorm(self.fc1_dims) 
        self.ln2 = nn.LayerNorm(self.fc2_dims)

        self.mu = nn.Linear(self.fc2_dims, self.n_actions) 

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003     
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

    def forward(self, state):
        x = self.fc1(state)
        x = self.ln1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)

        x = self.mu(x)
        x = torch.tanh(x) 

        return x

class OUActionNoise: 
    """Ornstein-Uhlenbeck process generating temporally correlated noise for continuous actions."""

    def __init__(self, n_actions=2, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        """
        Args:
            n_actions (int): Number of actions in the environment
            sigma (float): Noise volatility
            theta (float): Mean-reversion speed
            dt (float): Time step for the noise process
            x0 (np.array): Optional initial noise state (shape must match n_actions)
        """
        self.n_actions = n_actions
        self.mu = np.zeros(self.n_actions)              
        self.sigma = sigma
        self.theta = theta
        self.dt = dt                                   
        self.x0 = x0                                            
        self.reset()                                    

    def __call__(self):
        """
        Generate the next noise sample using:
        x_{t+1} = x_t + θ*(μ - x_t)*dt + σ*sqrt(dt)*N(0,1)
        """
        noise = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.n_actions)
        self.x_prev = noise
        return noise

    def reset(self):
        """
        Reset the noise process to initial state (usually at the start of each episode)
        """
        if self.x0 is not None:
            self.x_prev = self.x0.copy()
        else:
            self.x_prev = np.zeros(self.n_actions)

class DDPGAgent:
    
    def __init__(self, state_dim, action_dim, buffer, beta=0.001, alpha=0.0002,  gamma=0.99, batch_size=128):
        """
        Initializes the DDPG agent with necessary parameters and networks.
    
        Args:
            state_dim (int or tuple): Dimension or shape of the input state.
            action_dim (int): Number of possible actions in the environment.
            buffer (ReplayBuffer): Experience replay buffer for sampling training data.
            gamma (float): Discount factor for future rewards.
            batch_size (int): Number of transitions sampled per training step.
            alpha (float): Learning rate for the actor.
            beta (float): Learning rate for the critic.
        """
        self.actor_net = ActorNetwork(state_dim, 400, 300, action_dim).to(device)
        self.critic_net = CriticNetwork(state_dim, 400, 300, action_dim).to(device)
        
        self.target_actor = copy.deepcopy(self.actor_net)
        self.target_critic = copy.deepcopy(self.critic_net)
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.ou_noise = OUActionNoise()
        
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.TAU = 0.005
        
        self.batch_size = batch_size
        self.buffer = buffer

        self.optimizer_critic = torch.optim.Adam(self.critic_net.parameters(), lr=beta) 

        self.optimizer_actor = torch.optim.Adam(self.actor_net.parameters(), lr=alpha)
        

    def choose_actions(self, state):
        """Select an action using the actor network + OU exploration noise."""
       
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            actions = self.actor_net(state)
            
        noise = torch.tensor(self.ou_noise(), dtype=torch.float32).to(device)

        actions = actions + noise
        actions = torch.clamp(actions, -1, 1)
        
        return actions

    def train(self):
        """Train the DDPG agent."""
        if len(self.buffer) < self.batch_size:
            return  

        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        with torch.no_grad(): 
            target_actions = self.target_actor.forward(next_states)
            next_q_value = self.target_critic.forward(next_states, target_actions)
            target_q_value = rewards + self.gamma * (1 - dones) * next_q_value

        q_value = self.critic_net.forward(states, actions)

        self.optimizer_critic.zero_grad()
        critic_loss = F.mse_loss(q_value, target_q_value)
        critic_loss.backward()
        self.optimizer_critic.step()

        self.optimizer_actor.zero_grad()
        actor_loss = -self.critic_net.forward(states, self.actor_net.forward(states))
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.optimizer_actor.step()

        target_net_actor_dict = self.target_actor.state_dict()
        target_net_critic_dict = self.target_critic.state_dict()

        actor_net_dict = self.actor_net.state_dict()
        critic_net_dict = self.critic_net.state_dict()

        
        for key_critic in critic_net_dict:
            target_net_critic_dict[key_critic] = critic_net_dict[key_critic]*self.TAU + target_net_critic_dict[key_critic]*(1-self.TAU)

        for key_actor in actor_net_dict:
            target_net_actor_dict[key_actor] = actor_net_dict[key_actor]*self.TAU + target_net_actor_dict[key_actor]*(1-self.TAU)
            
        self.target_critic.load_state_dict(target_net_critic_dict)
        self.target_actor.load_state_dict(target_net_actor_dict)

    def policy(self, state):
        """
        Selects deterministic actions from the actor network (no exploration noise).
        Used for evaluation / inference.
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        
        with torch.no_grad():
            actions = self.actor_net(state)
    
        return actions

    def save_full_model(self, filename_actor, filename_critic):
        """
        Save the entire model (architecture and weights) to a file.
        """
        torch.save(self.actor_net, filename_actor)
        torch.save(self.critic_net, filename_critic)
        print(f"Full actor model saved to {filename_actor}")
        print(f"Full critic model saved to {filename_critic}")

    def load_full_model(self, filename1, filename2):
        """
        Load the entire model (architecture and weights) from a file.
        """
        model1 = torch.load(filename1, weights_only=False)
        model2 = torch.load(filename2, weights_only=False)
        
        self.actor_net = model1.to(device)
        self.target_actor = copy.deepcopy(model1).to(device)

        self.critic_net = model2.to(device)
        self.target_critic = copy.deepcopy(model2).to(device)
        
        print(f"Full critic model loaded from {filename2}")
        print(f"Full actor model loaded from {filename1}")

