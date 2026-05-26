"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
# from random import random, seed
import random
import gymnasium as gym
import torch
from agent import Policy, Agent

def main():
    
    
    seed = 7
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = gym.make('Hopper-v4')
    
    # set seed on environment for reproducibility
    env.reset(seed=seed)

    # print('State space:', env.observation_space)  # state-space
    # print('Action space:', env.action_space)  # action-space
    # print('-------------------')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = 'actor-critic' # options are 'REINFORCE' or 'actor-critic'

    policy = Policy(state_space=env.observation_space.shape[0], action_space=env.action_space.shape[0], mode=mode)
    agent = Agent(policy=policy, mode=mode, device=device, baseline=20)

    
    for episode in range(20000): 

        state, _ = env.reset()
        done = False 

        while not done: 

            action, action_log_prob = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
            done = terminated or truncated

            agent.store_outcome(state, next_state, action_log_prob, reward, done)

            state = next_state


        agent.update_policy()


        if (episode+1) % 500 == 0:
            print(f"\nEpisode {episode+1} has been completed")
            print(f"Reward is: {reward}")


    env.close()

    env_ = gym.make('Hopper-v4', render_mode='human')


    # this to see the performance of the model
    for episode in range(1000):
            
        state, _ = env_.reset()
        done = False

        while not done: 

            action, action_log_prob = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env_.step(action.detach().cpu().numpy())
            
            if truncated: print("trauncated")

            done = terminated or truncated

            agent.store_outcome(state, next_state, action_log_prob, reward, done)

            state = next_state
            env_.render()

if __name__ == '__main__':
    main()