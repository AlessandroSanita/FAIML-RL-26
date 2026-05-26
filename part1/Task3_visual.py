"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
# from random import random, seed
import matplotlib.pyplot as plt
import pandas as pd
import time 
import random
import gymnasium as gym
import torch
from agent import Policy, Agent

def train_model(num_episodes=5000):
    seed = 7
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = gym.make('Hopper-v4')
    env.reset(seed=seed)
    # print('State space:', env.observation_space)  # state-space
    # print('Action space:', env.action_space)  # action-space
    # print('-------------------\n\n')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = 'actor-critic'

    policy = Policy(state_space=env.observation_space.shape[0], action_space=env.action_space.shape[0], mode=mode)
    agent = Agent(policy=policy, mode=mode, device=device, baseline=20)

    final_rewards = []
    time_taken = []
    time_update = []

    for episode in range(num_episodes): 

        start_time = time.time()
        state, _ = env.reset()
        done = False 

        while not done: 

            action, action_log_prob = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
            done = terminated or truncated

            agent.store_outcome(state, next_state, action_log_prob, reward, done)

            state = next_state
        end_time = time.time()

        final_rewards.append(reward)
        time_taken.append(end_time - start_time)


        start_time = time.time()
        agent.update_policy()
        end_time = time.time()
        time_update.append(end_time - start_time)

        # if (episode+1) % 500 == 0:
        #     print(f"\nEpisode {episode+1} has been completed")
        #     print(f"Reward is: {reward}")


    env.close()

    return final_rewards, time_taken, time_update


def main():
    
    rewards = []
    time_taken = []
    time_update = []

    time_start = time.time()

    (rewards, time_taken, time_update) = train_model(num_episodes=50)

    time_end = time.time()

    print(f"\nTotal time taken for training: {time_end - time_start} seconds")

    pd.DataFrame(rewards).to_csv('part1\\Results\\Task3_rewards.csv', index=False)
    pd.DataFrame(time_taken).to_csv('part1\\Results\\Task3_time_taken.csv', index=False)
    pd.DataFrame(time_update).to_csv('part1\\Results\\Task3_time_update.csv', index=False)


    # visualizing rewards
    plt.figure(figsize=(12, 8))
    plt.plot(rewards, label=f'Reward', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.title('Reward vs Episode')
    plt.show() 

    # visualizing time taken
    plt.figure(figsize=(12, 8))
    plt.plot(time_taken, label=f'Time Taken', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Time Taken')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.title('Time Taken vs Episode')
    plt.show()

    #Visualizing time taken for policy update
    plt.figure(figsize=(12, 8))
    plt.plot(time_update, label=f'Time Update', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Time Update')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.title('Time Update vs Episode')
    plt.show()

if __name__ == '__main__':
    
    main()

