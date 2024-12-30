#import necessary modules (libraries)
import random
import numpy as np
import matplotlib.pyplot as plt
import gym

# Create FrozenLake environment
# Your code here:
#env=gym.make("FrozenLake-v1",is_slippery=True)
env = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4", render_mode="human")


# Q-table initialization
# Your code here:

action_space_size=env.action_space.n
state_space_size=env.observation_space.n
q_table=np.zeros((state_space_size,action_space_size))

# Hyperparameters
# Your code here:

learning_rate=0.7 # 0.05, 0.1,0.2,0.3
discount_rate=0.99  #0.9 ,0.99, 0.98
epsilon=1.0
max_epsilon=1.0
min_epsilon=0.01   # 0.01,0.05
epsilon_decay=0.00028 #0.001, 0.0005 0.00025
#minimum number of episodes=2000
num_episodes=30000
max_step_per_episode=100  #300




# Track rewards
rewards_all_episodes = []

#Track decay
decay_rates = []

# Q-learning algorithm
# Your code here:
for episode in range(num_episodes):
    state=env.reset()[0]
    done=False
    rewards_current_episode=0
    current_learning_rate=max(0.1,learning_rate*(0.99**episode))
    for step in range(max_step_per_episode):
        #exploration and explotation "rate"

        warm_up=random.uniform(0,1)
        if warm_up>epsilon:
            action=np.argmax(q_table[state,:]) #explotation
        else:
            action=env.action_space.sample() #exploration

        #Take action and observe the result
        new_state,reward, done,_,_=env.step(action)
        # Bellman equation update
        q_table[state, action] = (1 - current_learning_rate) * q_table[state, action] + \
            current_learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        state = new_state
        rewards_current_episode += reward  # Accumulate reward

        if done:
            break


    
    # Decay exploration rate
    decay_rates.append(epsilon)
    epsilon = min_epsilon + \
                       (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)
    
    
    # Track total reward for this episode
    rewards_all_episodes.append(rewards_current_episode)

# Calculate the average reward per 1000 episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
#rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1)
rewards_avg_per_thousand_episodes = [np.mean(rewards) for rewards in rewards_per_thousand_episodes]

# Plot the graph
fig, axs = plt.subplots(2, figsize=(8, 10))

# Plot decay
axs[0].plot(range(num_episodes), decay_rates, color='blue')
axs[0].set_title('Epsilon Decay Over Time')
axs[0].set_xlabel('Episodes')
axs[0].set_ylabel('Decay(Rate)')
axs[0].grid(True)

# Plot average reward per thousand episodes
axs[1].plot(range(1, len(rewards_avg_per_thousand_episodes) + 1), rewards_avg_per_thousand_episodes, color='green')
axs[1].set_title('Average Reward per Thousand Episodes')
axs[1].set_xlabel('Episodes (in thousands)')
axs[1].set_ylabel('Average Reward')
axs[1].grid(True)

plt.tight_layout()
plt.show()