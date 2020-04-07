import numpy as np
import random
import matplotlib.pyplot as plt


def testbed(n_arms, mean, variance):
    action_values = np.random.normal(mean, variance, (1, n_arms))
    return action_values[0]

def do_action(a, testbed):
    return np.random.normal(testbed[a], 1)

def e_greedy(epsilon, test_bed, n_timesteps):
    # with sample averages
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
            action_values[action] = action_values[action] + (1/float(step)) * (reward - action_values[action])
    return rewards

def greedy(test_bed, n_timesteps):
    # with sample averages
    rewards = []
    action_values = np.zeros((len(test_bed)))
    for step in range(1, n_timesteps + 1):
        action = np.argmax(action_values)
        reward = do_action(action, test_bed)
        rewards.append(reward)
        if step < n_timesteps - 1:
            action_values[action] = action_values[action] + (1/float(step)) * (reward - action_values[action])
    return rewards

N_RUNS = 2000
N_TIMESTEPS = 1000
one = np.ndarray((N_RUNS, N_TIMESTEPS))
two = np.ndarray((N_RUNS, N_TIMESTEPS))
three = np.ndarray((N_RUNS, N_TIMESTEPS))
for i in range(N_RUNS):
    test = testbed(10, 0, 1)
    r1 = greedy(test, N_TIMESTEPS)
    r2 = e_greedy(0.1, test, N_TIMESTEPS)
    r3 = e_greedy(0.01, test, N_TIMESTEPS)
    one[i] = r1
    two[i] = r2
    three[i] = r3


fig, ax = plt.subplots()

ax.plot(np.arange(N_TIMESTEPS), one.mean(axis=0), label=f"greedy")
ax.plot(np.arange(N_TIMESTEPS), two.mean(axis=0), label=f"e = 0.1")
ax.plot(np.arange(N_TIMESTEPS), three.mean(axis=0), label=f"e = 0.01")
        
ax.legend()

ax.set(xlabel='steps', ylabel='average reward')
ax.grid()

#fig.savefig(f"figs/all_task_execution_time__{data_id}.png")
plt.show()
