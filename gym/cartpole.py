import gym
import numpy as np
import random
env = gym.make("CartPole-v1")
observation = env.reset()


print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))


alpha = 0.01 # learning rate
gamma = 0.9 # discount factor
epsilon  = 0.1 # exploration factor
q_table = np.zeros([env.observation_space.shape[0], env.action_space.n]) # initialize qtable to zero 

for i in range(1, 1000):
    state = env.reset()
    print(state)
    epochs, penalties, reward, = 0, 0, 0
    done = False
    while not done:
        explore  = random.random()
        if explore < epsilon:
            action = env.action_space.sample() # int(random.uniform(0, 5))
        else:
            action = np.argmax(q_table[state])
        next_state, reward, done, info = env.step(action)

        next_action_q_value = np.max(q_table[next_state])
        q_table[state, action] = (1-alpha)*q_table[state, action] + alpha * (reward + gamma * next_action_q_value)
        state = next_state
        epochs += 1
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")
print('done!')

# for _ in range(10):
#   env.render()
#   action = env.action_space.sample() # your agent here (this takes random actions)
#   observation, reward, done, info = env.step(action)
#   print(reward)

#   if done:
#     observation = env.reset()
# env.close()