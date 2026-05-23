"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import gymnasium as gym
import torch
from agent import Policy, Agent

def main():
    env = gym.make('Hopper-v4')

    # print('State space:', env.observation_space)  # state-space
    # print('Action space:', env.action_space)  # action-space
    # print('-------------------\n\n')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = Policy(state_space=env.observation_space.shape[0], action_space=env.action_space.shape[0])
    agent = Agent(policy=policy, device=device)

    
    for episode in range(10000): 

        state, _ = env.reset()
        done = False

        while not done: 

            action, action_log_prob = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
            done = terminated or truncated

            agent.store_outcome(state, next_state, action_log_prob, reward, done)

            state = next_state


        agent.update_policy(baseline=20)

        
        if (episode+1) % 500 == 0:
            print(f"\nEpisode {episode+1} has been completed")
            print(f"Reward is: {reward}")


    env.close()
    
    env_ = gym.make('Hopper-v4', render_mode='human')


    # this to see the performance of the model

    for episode in range(100):
            
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