"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import gymnasium as gym
import torch
from agent import Policy, Agent

def main():
    env = gym.make('Hopper-v4')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('-------------------\n\n')

    #TODO: implement training loop for REINFORCE and Actor-Critic using the agent defined in agent.py

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = Policy(state_space=env.observation_space.shape[0], action_space=env.action_space.shape[0])
    agent = Agent(policy=policy, device=device)

    
    for episode in range(150): 

        state, _ = env.reset()
        done = False

        while not done: 

            action, action_log_prob = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_outcome(state, next_state, action_log_prob, reward, done)

            state = next_state


        agent.update_policy()

        
        if episode % 20 == 0:
            print(f"Episode {episode+1} has been completed\n")

if __name__ == '__main__':
    main()