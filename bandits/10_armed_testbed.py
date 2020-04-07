import numpy as np
import random
import matplotlib.pyplot as plt


def testbed(n_arms, mean, variance):
    action_values = np.random.normal(mean, variance, (1, n_arms))
    return action_values[0]

def do_action(a, testbed):
    return np.random.normal(testbed[a], 1)

def e_greedy(epsilon, test_bed, n_timesteps):
    rewards = []
    action_values = np.zeros((len(test_bed)))
    for step in range(1, n_timesteps + 1):
        if random.random() > epsilon:
            action = np.argmax(action_values)
        else:
            action = random.choice(list(range(len(action_values))))
        reward = do_action(action, test_bed)
        rewards.append(reward)
        if step < n_timesteps - 1:
            action_values[action] = action_values[action] * (1/float(step)) * (reward - action_values[action])
    return rewards

def greedy(test_bed, n_timesteps):
    rewards = []
    action_values = np.zeros((len(test_bed)))
    for step in range(1, n_timesteps + 1):
        action = np.argmax(action_values)
        reward = do_action(action, test_bed)
        rewards.append(reward)
        if step < n_timesteps - 1:
            action_values[action] = action_values[action] * (1/float(step)) * (reward - action_values[action])
    return rewards

test = testbed(10, 0, 1)
r1 = greedy(test, 1000)
r2 = e_greedy(0.1, test, 1000)

fig, ax = plt.subplots()

ax.plot(np.arange(1000), r1, label=f"greedy")
ax.plot(np.arange(1000), r2, label=f"e = 0.1")
        
ax.legend()

ax.set(xlabel='steps', ylabel='reward')
ax.grid()

#fig.savefig(f"figs/all_task_execution_time__{data_id}.png")
plt.show()
