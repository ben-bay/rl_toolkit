import numpy as np
import random
import matplotlib.pyplot as plt

"""
> Add UCB
> Add gradient algorithm
> Add TS?
"""

def testbed(n_arms, mean, variance):
    action_values = np.random.normal(mean, variance, (1, n_arms))
    return action_values[0]

def do_action(a, testbed):
    return np.random.normal(testbed[a], 1)

def e_greedy(epsilon, test_bed, n_timesteps, optimism=0):
    # with sample averages
    rewards = []
    action_values = np.zeros((len(test_bed))) + optimism
    action_counts = np.zeros((len(test_bed)))
    for step in range(1, n_timesteps + 1):
        if random.random() > epsilon:
            action = np.argmax(action_values)
        else:
            action = random.choice(list(range(len(action_values))))
        reward = do_action(action, test_bed)
        rewards.append(reward)
        action_counts[action] += 1.0
        if step < n_timesteps - 1:
            action_values[action] = action_values[action] + (1 / action_counts[action]) * (reward - action_values[action])
    return rewards

def ucb(c, test_bed, n_timesteps):
    rewards = []
    action_values = np.zeros((len(test_bed)))
    action_counts = np.zeros((len(test_bed)))
    for step in range(1, n_timesteps + 1):
        action = np.argmax(action_values + (c * np.sqrt(np.log(step) / action_counts)))
        reward = do_action(action, test_bed)
        rewards.append(reward)
        action_counts[action] += 1.0
        if step < n_timesteps - 1:
            action_values[action] = action_values[action] + (1 / action_counts[action]) * (reward - action_values[action])
    return rewards

def gradient(step_size, test_bed, n_timesteps, baseline=True):
    rewards = []
    action_preferences = np.zeros((len(test_bed)))
    action_counts = np.zeros((len(test_bed)))
    action_probabilities = np.zeros((len(test_bed))) + (1.0 / len(test_bed))
    mean_reward = 0
    for step in range(1, n_timesteps + 1):
        rand = random.random()
        prob_sum = 0
        for a, prob in enumerate(action_probabilities):
            prob_sum += prob
            if prob_sum > rand:
                action = a
        reward = do_action(action, test_bed)
        rewards.append(reward)
        if baseline:
            #mean_reward = mean_reward + (1.0 / step) * (reward - mean_reward)
            mean_reward = np.mean(rewards)
        action_counts[action] += 1.0
        if step < n_timesteps - 1:
            action_preferences[action] = action_preferences[action] + step_size * (reward - mean_reward) * (1 - action_probabilities[action])
            action_probabilities[action] = (np.e ** action_preferences[action]) / (np.e ** np.sum(action_preferences))
            for alt_action in range(len(test_bed)):
                if alt_action != action:
                    action_preferences[alt_action] = action_preferences[alt_action] - step_size * (reward - mean_reward) * action_probabilities[alt_action]
    return rewards

N_ARMS = 100
N_RUNS = 1000
N_TIMESTEPS = 1000
#one = np.ndarray((N_RUNS, N_TIMESTEPS))
#two = np.ndarray((N_RUNS, N_TIMESTEPS))
#three = np.ndarray((N_RUNS, N_TIMESTEPS))
#four = np.ndarray((N_RUNS, N_TIMESTEPS))
#five = np.ndarray((N_RUNS, N_TIMESTEPS))
#six = np.ndarray((N_RUNS, N_TIMESTEPS))
seven = np.ndarray((N_RUNS, N_TIMESTEPS))
for i in range(N_RUNS):
    test = testbed(N_ARMS, 0, 1)
    #r1 = e_greedy(0, test, N_TIMESTEPS)
    #r2 = e_greedy(0.1, test, N_TIMESTEPS)
    #r3 = e_greedy(0.01, test, N_TIMESTEPS)
    #r4 = e_greedy(0, test, N_TIMESTEPS, optimism=5)
    #r5 = ucb(2, test, N_TIMESTEPS)
    #r6 = ucb(1, test, N_TIMESTEPS)
    r7 = gradient(100, test, N_TIMESTEPS)
    #one[i] = r1
    #two[i] = r2
    #three[i] = r3
    #four[i] = r4
    #five[i] = r5
    #six[i] = r6
    seven[i] = r7


fig, ax = plt.subplots()

#ax.plot(np.arange(N_TIMESTEPS), one.mean(axis=0), label=f"greedy")
#ax.plot(np.arange(N_TIMESTEPS), two.mean(axis=0), label=f"e=0.1")
#ax.plot(np.arange(N_TIMESTEPS), three.mean(axis=0), label=f"e=0.01")
#ax.plot(np.arange(N_TIMESTEPS), four.mean(axis=0), label=f"optimistic, e=0, o=5")
#ax.plot(np.arange(N_TIMESTEPS), five.mean(axis=0), label=f"ucb, c=2")
#ax.plot(np.arange(N_TIMESTEPS), six.mean(axis=0), label=f"ucb, c=1")
ax.plot(np.arange(N_TIMESTEPS), seven.mean(axis=0), label=f"gradient, alpha=100")
        
ax.legend()

ax.set(xlabel='steps', ylabel='average reward')
ax.grid()

fig.savefig(f"bandits_k{N_ARMS}_{N_RUNS}runs.png")
plt.show()
