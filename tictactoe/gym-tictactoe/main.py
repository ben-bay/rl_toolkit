import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')

import random

import gym
import gym_tictactoe
import gym_tictactoe.envs
from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv
import gym_tictactoe.envs.agent as agent

interactive = True
control_player = 1
experiment_player = 2

def myprint(string):
    if interactive:
        print(string)

def game(control_player, experiment_player, control_agent, experiment_agent, policy):
    if random.random() > 0.5:
        first_turn = control_player
        second_turn = experiment_player
        first_agent = control_agent
        second_agent = experiment_agent
    else:
        first_turn = experiment_player
        second_turn = control_player
        first_agent = experiment_agent
        second_agent = control_agent
    
    time = 0
    env = gym.make('tictactoe-v0')
    env.init2(first_turn, second_turn, first_agent, second_agent)

    state = env.state_history[0]
    game_over = False
    while not game_over:
        time += 1
        action = agent.get_agent_action(state, first_agent, first_turn, policy, experiment_player, control_player)
        state = env.step(action)
        env.render()
        if TicTacToeEnv.is_terminal_state(state):
            break
    return state, TicTacToeEnv.is_win(state, experiment_player)

def main():
    control_player = 1
    experiment_player = 2
    policy = {}
    experiment_wins = 0
    experiment_losses = 0
    final_state, exp_won = game(control_player, experiment_player, "random", "e-greedy", policy)
    if exp_won:
        experiment_wins += 1
    else:
        experiment_losses += 1
    myprint("GAME OVER!")

if __name__ == "__main__":
    sys.exit(main())
