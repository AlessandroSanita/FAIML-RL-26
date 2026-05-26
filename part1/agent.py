import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal



def discount_rewards(r, gamma):
    """
    Create vector 'discounted_r', with as many entries as reward vector r. 
    Starting from the t=T, calcualte the discounted reward value until time t_0.
    """
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, mode = 'REINFORCE'):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.mode_is_reinforce = mode == 'REINFORCE'
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm

        if not self.mode_is_reinforce:
            self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
            self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
            self.fc3_critic = torch.nn.Linear(self.hidden, 1) 

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network
        if not self.mode_is_reinforce:
            x_critic = self.tanh(self.fc1_critic(x))
            x_critic = self.tanh(self.fc2_critic(x_critic))
            state_value = self.fc3_critic(x_critic) 
    
            return normal_dist, state_value 
        
        return normal_dist


class Agent(object):
    """ policy = ANN """
    def __init__(self, policy, mode = 'REINFORCE', device='cpu', baseline = 20):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        # self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        # self.baseline = baseline
        self.mode_is_reinforce = mode == 'REINFORCE'
        self.baseline = baseline if self.mode_is_reinforce else 0 # I think this way it looks cooler lol + accounts for actor-critic
        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []


        if self.mode_is_reinforce:
            # ====================================
            #               TASK 2:
            # ====================================
            #   - compute discounted returns
            G_t = discount_rewards(rewards, self.gamma)

            #   - compute policy gradient loss function given actions and returns
            loss = - ((G_t - self.baseline) * action_log_probs).mean() 

        else:
            # ====================================
            #               TASK 3:
            # ====================================

            # As we already have the next states, we need to find the state values for the formula 
            _, state_value = self.policy(states)
            _, next_state_value = self.policy(next_states)
            
            # print("state_value", state_value)
            # print("next_state_value", next_state_value)

            state_value, next_state_value = state_value.squeeze(-1), next_state_value.squeeze(-1)

            #   - compute boostrapped discounted return estimates
            # \delta <-- R + \gamma * v_hat (S', w)
            bootstrapped_return = rewards + self.gamma * next_state_value * (1 - done)

            #   - compute advantage terms
            # \delta <-- \delta - v_hat(S, w) 
            advantage = bootstrapped_return - state_value

            #   - compute actor loss and critic loss
            # detach, according to what I found, is used so the gradient from critic doesn't "flow" to actor
            actor_loss = - (advantage.detach() * action_log_probs).mean() 
            critic_loss = F.mse_loss(state_value, bootstrapped_return.detach())

            loss = actor_loss + critic_loss

        #   - compute gradients and step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def get_action(self, state, evaluation=False):
        """ state -- ANN --> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        if self.mode_is_reinforce:
            normal_dist = self.policy(x)
        else:
            normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

