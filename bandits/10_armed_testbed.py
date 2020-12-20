import argparse
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

"""
> Add gradient algorithm
> Add Thompson sampling
"""


def testbed(n_arms, mean, variance):
    action_values = np.random.normal(mean, variance, (1, n_arms))
    return action_values[0]


def do_action(a, testbed):
    return np.random.normal(testbed[a], 1)


def e_greedy(test_bed, n_timesteps, epsilon=0, optimism=0):
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
            action_values[action] = action_values[action] + (
                1 / action_counts[action]
            ) * (reward - action_values[action])
    return rewards


def ucb(test_bed, n_timesteps, c):
    rewards = []
    action_values = np.zeros((len(test_bed)))
    action_counts = np.zeros((len(test_bed)))
    for step in range(1, n_timesteps + 1):
        action = np.argmax(action_values + (c * np.sqrt(np.log(step) / action_counts)))
        reward = do_action(action, test_bed)
        rewards.append(reward)
        action_counts[action] += 1.0
        if step < n_timesteps - 1:
            action_values[action] = action_values[action] + (
                1 / action_counts[action]
            ) * (reward - action_values[action])
    return rewards


def gradient(test_bed, n_timesteps, step_size, baseline=True):
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
            # mean_reward = mean_reward + (1.0 / step) * (reward - mean_reward)
            mean_reward = np.mean(rewards)
        action_counts[action] += 1.0
        if step < n_timesteps - 1:
            action_preferences[action] = action_preferences[action] + step_size * (
                reward - mean_reward
            ) * (1 - action_probabilities[action])
            action_probabilities[action] = (np.e ** action_preferences[action]) / (
                np.e ** np.sum(action_preferences)
            )
            for alt_action in range(len(test_bed)):
                if alt_action != action:
                    action_preferences[alt_action] = (
                        action_preferences[alt_action]
                        - step_size
                        * (reward - mean_reward)
                        * action_probabilities[alt_action]
                    )
    return rewards


def process_args(args):
    n_arms = args.narms
    n_runs = args.nruns
    n_timesteps = args.ntimesteps
    alg_specs = []
    # algortihm tuple: function, plot label, kwargs
    alg_specs.append((e_greedy, "greedy", {}))
    alg_specs.append((e_greedy, "greedy e=0.1", {"epsilon": 0.1}))
    alg_specs.append((e_greedy, "greedy e=0.01", {"epsilon": 0.01}))
    alg_specs.append((e_greedy, "optimistic, e=0, o=5", {"optimism": 5}))
    alg_specs.append((ucb, "ucb, c=2", {"c": 2}))
    alg_specs.append((ucb, "ucb, c=1", {"c": 1}))
    # alg_specs.append((gradient, "gradient, alpha=100", {"step_size": 100}))
    n_algs = len(alg_specs)
    algorithm_runs = []
    for i in range(n_algs):
        algorithm_runs.append(np.ndarray((n_runs, n_timesteps)))
    for run_i in range(n_runs):
        test = testbed(n_arms, args.mean, args.variance)
        rewards = []
        for alg_i in range(n_algs):
            alg = alg_specs[alg_i]
            rewards.append(alg[0](test, n_timesteps, **alg[2]))
        for alg_i in range(n_algs):
            algorithm_runs[alg_i][run_i] = rewards[alg_i]

    fig, ax = plt.subplots()

    for alg_i in range(n_algs):
        ax.plot(
            np.arange(n_timesteps),
            algorithm_runs[alg_i].mean(axis=0),
            label=alg_specs[alg_i][1],
        )

    ax.legend()

    ax.set(xlabel="steps", ylabel="average reward")
    ax.grid()

    fig.savefig(f"bandits_k{n_arms}_{n_runs}runs.png")
    plt.show()


def setup_argparse():
    parser = argparse.ArgumentParser(
        description='Run simple exploration algorithms on the toy multi-armed bandit problem "10-armed testbed"'
    )
    parser.add_argument(
        "--narms", action="store", type=int, help="Number of arms", default=10
    )
    parser.add_argument(
        "--mean", action="store", type=float, help="Arm reward mean", default=0.0
    )
    parser.add_argument(
        "--variance",
        action="store",
        type=float,
        help="Arm reward variance",
        default=1.0,
    )
    parser.add_argument(
        "--ntimesteps",
        action="store",
        type=int,
        help="Number of timesteps per run",
        default=100,
    )
    parser.add_argument(
        "--nruns",
        action="store",
        type=int,
        help="Number of runs per algorithm",
        default=1000,
    )
    parser.set_defaults(func=process_args)
    return parser


def main():
    parser = setup_argparse()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
