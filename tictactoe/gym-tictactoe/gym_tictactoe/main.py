import sys
#sys.path.append('/usr/local/lib/python3.7/site-packages')

import random

import gym
import envs
from envs.custom_env_dir.tictactoe_env import TicTacToeEnv
import envs.custom_env_dir.agent as agent

interactive = False
symbol_map = {1:"X",2:"O"}

def myprint(string):
    if interactive:
        print(string)

class Player:
    def __init__(self, experimental, agent_type, is_first, the_id):
        self.experimental = experimental
        self.agent_type = agent_type
        self.is_first = is_first
        self.id = the_id

def game(player_a, player_b, policy):
    env = gym.make('tictactoe-v0')

    player_a.is_first = True
    player_a.id = 1
    env.init2(player_a, player_b)
    player1 = player_a
    player2 = player_b
    experiment_player = player_b
    """
    if random.random() > 0.5:
        player_a.is_first = True
        player_a.id = 1
        env.init2(player_a, player_b)
        player1 = player_a
        player2 = player_b
        experiment_player = player_b
    else:
        player_b.is_first = True
        player_b.id = 1
        env.init2(player_b, player_a)
        player1 = player_b
        player2 = player_a
        experiment_player = player_a
    """

    print(player1.agent_type + " " + symbol_map[player1.id] + " vs " + player2.agent_type + " " + symbol_map[player2.id])

    state = env.state_history[0]
    game_over = False
    env.render()
    while not game_over:
        action = agent.get_agent_action(state, player1, policy, past_state=env.state_history[env.time])
        state = env.step(action, policy)
        agent.add_new_state_to_policy(state, policy)
        env.render()
        if TicTacToeEnv.is_terminal_state(state):
            break
    return state, TicTacToeEnv.is_win(state, experiment_player.id)

def main():
    policy = {}
    experiment_wins = 0
    experiment_losses = 0
    GAMES = 1000
    for g in range(GAMES):
        final_state, exp_won = game(Player(False, "random", False, 2), Player(True, "e-greedy", False, 2), policy)
        myprint("GAME OVER!")
        if exp_won:
            experiment_wins += 1
        else:
            experiment_losses += 1
        #print(len(policy))
        #print(policy)
    winrate = round(experiment_wins/(experiment_wins+experiment_losses)*100, 3)
    print(f"Winrate: {winrate}%")
    print(len(policy))
    print(policy.values())

if __name__ == "__main__":
    sys.exit(main())
