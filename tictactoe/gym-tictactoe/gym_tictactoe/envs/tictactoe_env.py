import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import gym_tictactoe.envs.agent as agent

class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human','fast']}

    def __init__(self):
        self.time = 0
        self.state_history = {}
        self.state_history[self.time] = np.zeros((3,3))
        self.game_over = False

        self.first_turn = None
        self.second_turn = None
        self.first_agent = None
        self.second_agent = None

    def init2(self, first_turn, second_turn, first_agent, second_agent):
        self.first_turn = first_turn
        self.second_turn = second_turn
        self.first_agent = first_agent 
        self.second_agent = second_agent

    def step(self, action):
        self.time += 1
        new_state = TicTacToeEnv.do_action(action, 2, self.state_history[self.time - 1])
        self.state_history[self.time] = new_state 
        if TicTacToeEnv.is_terminal_state(new_state):
            return
        self.render()
        self.time += 1
        #state = action(get_agent_action(self.state_history[self.time], "random", second_turn), second_turn, state)
        new_state = TicTacToeEnv.do_action(agent.get_agent_action(self.state_history[self.time - 1], self.second_agent, self.second_turn, {}, self.second_turn, self.first_turn, past_state=self.state_history[self.time - 2]), self.second_turn, new_state)
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
    def do_action(square, player_id, pre_state):
        result_state = np.copy(pre_state)
        result_state[square] = player_id 
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

