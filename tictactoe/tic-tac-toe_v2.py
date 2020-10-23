import numpy as np
import random
import sys
from copy import deepcopy


def is_full(board):
    return not board.__contains__(0)

def is_win(board, player):
    mask = board == player
    out = mask.all(0).any() | mask.all(1).any()
    out |= np.diag(mask).all() | np.diag(mask[:,::-1]).all()
    return out

def is_terminal(board):
    return is_win(board, 1) or is_win(board, 2) or is_full(board)

def x_move(board, state_values):
    for i in range(3):
        for j in range(3):
            if board[i,j] != 0:
                continue
            move = (i,j)
            board[move] = 1
            if board.tobytes() in state_values:
                board[move] = 0
                continue
            elif is_win(board, 1):
                state_values[board.tobytes()] = 1.0
            elif is_full(board):
                state_values[board.tobytes()] = 0.0
            else:
                state_values[board.tobytes()] = 0.5
                state_values = o_move(board, state_values)
            board[move] = 0
    return state_values

def o_move(board, state_values):
    for i in range(3):
        for j in range(3):
            if board[i,j] != 0:
                continue
            move = (i,j)
            board[move] = 2
            if board.tobytes() in state_values:
                board[move] = 0
                continue
            elif is_win(board, 2) or is_full(board):
                state_values[board.tobytes()] = 0.0
            else:
                state_values[board.tobytes()] = 0.5
                state_values = x_move(board, state_values)
            board[move] = 0
    return state_values

def init_value_table():
    board = np.zeros((3,3))
    state_values = x_move(board, {board.tobytes(): 0.5})
    return state_values

def human_input(state):
    row = input("\nrow: ")
    col = input("\ncol: ")
    while int(row) > 2 or int(row) < 0 or int(col) > 2 or int(col) < 0 or state[int(row), int(col)] != 0:
        print("Invalid!")
        row = input("\nrow: ")
        col = input("\ncol: ")
    return int(row), int(col)

def get_available_actions(state):
    return list(zip(list(np.where(state==0)[0]), list(np.where(state==0)[1])))

def get_greedy_action(state, player_id, value_function, maximize):
        candidate_actions = get_available_actions(state)
        if len(candidate_actions) == 0:
            return None, None
        if maximize:
            max_value = -np.inf
        else:
            max_value = np.inf
        max_value_actions = [random.choice(candidate_actions)]
        max_value_states = [np.copy(state)]
        for candidate_action in candidate_actions:
            candidate_state = np.copy(state)
            candidate_state[candidate_action] = player_id
            #print(candidate_state)
            value = value_function[candidate_state.tobytes()]
            if maximize:
                if value > max_value:
                    max_value = value
                    max_value_actions = [candidate_action]
                    max_value_states = [candidate_state]
            else:
                if value < max_value:
                    max_value = value
                    max_value_actions = [candidate_action]
                    max_value_states = [candidate_state]

            # stochastic action decision
            #elif value == max_value:
            #    max_value_actions.append(candidate_action)
            #    max_value_states.append(candidate_state)
        index = random.choice(range(len(max_value_actions)))
        return max_value_actions[index], max_value_states[index]

class Agent:
    def __init__(self, _id):
        self.id = _id

    def update_value_function(self, state, past_state, value_function):
        return value_function

class AutonomousAgent(Agent):
    def __init__(self, _id, is_learning=False):
        super().__init__(_id)
        self.is_learning = is_learning

class Human(Agent):
    def __init__(self, _id):
        super().__init__(_id)

    def get_action(self, state, value_function):
        return human_input(state)#, value_function

class Random(AutonomousAgent):
    def __init__(self, _id):
        super().__init__(_id, False)

    def get_action(self, state, value_function):
        return random.choice(get_available_actions(state))#, value_function

class Greedy(AutonomousAgent):
    def __init__(self, _id, step_size=0.001, is_learning=False):
        super().__init__(_id, is_learning)
        self.step_size = step_size

    def get_action(self, state, value_function):
        greedy_action, _ = get_greedy_action(state, self.id, value_function, self.id==1)
        return greedy_action

    def update_value_function(self, state, past_state, value_function):
        if value_function is None:
            raise ValueError("value_function is None")
        result_value_function = deepcopy(value_function)
        if past_state is not None and self.is_learning:
            # temporal difference: update value of current state
            result_value_function[past_state.tobytes()] = result_value_function[past_state.tobytes()] + (self.step_size * (result_value_function[state.tobytes()] - result_value_function[past_state.tobytes()]))
        if result_value_function is None:
            raise ValueError("value_function is None")
        return result_value_function

class E_greedy(Greedy):
    def __init__(self, _id, step_size=0.1, e=0.1, is_learning=True):
        super().__init__(_id, step_size, is_learning)
        self.epsilon = e

    def get_action(self, state, value_function):
        if random.random() <= self.epsilon:
            return random.choice(get_available_actions(state))#, value_function
        else:
            return super().get_action(state, value_function)

def train_value_function(): 
    value_function = init_value_table()
    values = list(value_function.values())
    print(set(values))
    print(f"mean state value = {np.mean(values)}")
    print(f"std state value = {np.std(values)}")
    o_value_function = deepcopy(value_function)
    x_value_function = deepcopy(value_function)
    for i in range(1000):
        x_value_function, o_value_function = game(Random(1), E_greedy(2), x_value_function, o_value_function)
    values = list(o_value_function.values())
    print(set(values))
    print(f"mean state value = {np.mean(values)}")
    print(f"std state value = {np.std(values)}")
    return o_value_function


def game(x_player, o_player, x_value_function, o_value_function, render=False):
    state = np.zeros((3,3))
    x_past_state = None
    o_past_state = np.copy(state)
    iteration = 0
    while not is_terminal(state): 
        x_action = x_player.get_action(state, x_value_function)
        state[x_action] = x_player.id
        if render:
            print(state)
        if iteration > 0:
            x_value_function = x_player.update_value_function(state, x_past_state, x_value_function)
        x_past_state = np.copy(state)
        #print(state)
        #if is_terminal(state) and x_player.is_learning:
        #    x_value_function = x_player.update_value_function(state, x_past_state, x_value_function)
        if is_terminal(state):
            break
        o_action = o_player.get_action(state, o_value_function)
        state[o_action] = o_player.id
        if render:
            print(state)
        #if is_terminal(state):
        #    print("O wins")
        #    print(o_past_state)
        #    print(o_value_function[o_past_state.tobytes()])
        #    print(state)
        #    print(o_value_function[state.tobytes()])
        #    o_value_function = o_player.update_value_function(state, o_past_state, o_value_function)
        #    print(o_value_function[o_past_state.tobytes()])
        #    print(o_player.is_learning)
        #    sys.exit()
        o_value_function = o_player.update_value_function(state, o_past_state, o_value_function)
        o_past_state = np.copy(state)
        #print(state)
        #if is_terminal(state) and o_player.is_learning:
        #    o_value_function = o_player.update_value_function(state, o_past_state, o_value_function)
        #    break
        iteration += 1
    return x_value_function, o_value_function
    
def main():
    value_function = train_value_function()
    game(E_greedy(1), Human(2), deepcopy(value_function), deepcopy(value_function), render=True)
    #value_table = init_value_table()
    #print(f"number of states (should be 5478): {len(value_table)}")
    #print(f"value_table state values: {set(value_table.values())}")


if __name__ == "__main__":
    sys.exit(main())
