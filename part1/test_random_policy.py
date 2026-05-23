"""Test a random policy on the Gym Hopper environment

    Play around with this code to get familiar with the
    Hopper environment.

    For example, what happens if you don't reset the environment
    even after the episode is over?
    When exactly is the episode over?
    What is an action here?
"""
import gymnasium as gym

def main():
    render = False

    if render:
        env = gym.make('Hopper-v4', render_mode='human')
    else:
        env = gym.make('Hopper-v4', render_mode='rgb_array')
    
    # print('State space:', env.observation_space)  # state-space
    # print('Action space:', env.action_space)  # action-space


    print("State of the robot")
    print(f"body_names: {env.unwrapped.get_wrapper_attr("mass")}")
    print(f"body_names: {env.unwrapped.sim.model.body_names}")
    print(f"body_mass: {env.sim.model.body_mass}")
    print(f"nv: {env.sim.model.nv}")
    print(f"body_dofnum: {env.sim.model.body_dofnum}")
    print(f"nu: {env.sim.model.nu}")



    n_episodes = 5

    for ep in range(n_episodes):  
        done = False
        state, info = env.reset()  # Reset environment to initial state
        
        step = 0

        print("================================================================")

        while not done:  # Until the episode is over
            action = env.action_space.sample()  # Sample random action

            state, reward, terminated, truncated, _ = env.step(action)  # Step the simulator to the next timestep
            
            
            if step%5 == 0:
                print(f"\n\nSummary at step {step} from episode: {ep}")
                print("Current state:")
                print(state)
                print("\nAction selected")
                print(action)
                print("\nReward")
                print(reward)

            done = terminated or truncated

            step+=1

            if render:
                env.render()


if __name__ == '__main__':
    main()