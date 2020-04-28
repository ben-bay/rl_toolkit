import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import envs.custom_env_dir.agent as agent

class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human','fast']}

    def __init__(self):
        self.time = 0
        self.state_history = {}
        self.state_history[self.time] = np.zeros((3,3))
        self.game_over = False

        self.player1 = None
        self.player2 = None

    def init2(self, player1, player2):
        self.player1 = player1
        self.player2 = player2

        # environment player goes first
        self.time += 1
        self.state_history[1] = np.zeros((3,3))
        self.env_action(self.state_history[1])

    def step(self, action):
        new_state = TicTacToeEnv.do_action(action, self.player2, self.state_history[self.time])
        self.time += 1
        self.state_history[self.time] = new_state 
        if TicTacToeEnv.is_terminal_state(new_state):
            return
        self.render()
        new_state = self.env_action(new_state) 
        return new_state

    def env_action(self, state):
        new_state = TicTacToeEnv.do_action(agent.get_agent_action(self.state_history[self.time], self.player1, {}, past_state=self.state_history[self.time - 1]), self.player1, state)
        self.time += 1
        self.state_history[self.time] = new_state
        return new_state

    def reset(self):
        self.time = 0
        self.state_history[time] = np.zeros((3,3))
        self.game_over = False

    def render(self, mode='human', close=False):
        if mode == "human":
            i = 0
            for cell in np.nditer(self.state_history[self.time]):
                if cell == 0:
                    print(". ", end="")
                if cell == 1:
                    print("X ", end="")
                if cell == 2:
                    print("O ", end="")
                i += 1
                if i % 3 == 0:
                    print("")
            print("\n")

    @staticmethod
    def do_action(square, player, pre_state):
        result_state = np.copy(pre_state)
        result_state[square] = player.id 
        return result_state

    @staticmethod
    def is_terminal_state(state):
        return TicTacToeEnv.is_win(state, 1) or TicTacToeEnv.is_win(state, 2) or (np.sum(np.where(state == 0)) == 0)

    @staticmethod
    def is_win(state, _id):
        if state[0,0] == _id and state[1,1] == _id and state[2,2] == _id:
            return True

        if state[0,2] == _id and state[1,1] == _id and state[2,0] == _id:
            return True

        return any((state[:]==[_id,_id,_id]).all(1)) or any((state.T[:]==[_id,_id,_id]).all(1))

