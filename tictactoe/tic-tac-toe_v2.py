import numpy as np
import random


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

def get_greedy_action(state, player_id, value_function):
        candidate_actions = get_available_actions(state)
        max_value = -np.inf
        max_value_actions = [random.choice(candidate_actions)]
        max_value_states = [np.copy(state)]
        for candidate_action in candidate_actions:
            candidate_state = np.copy(state)
            candidate_state[candidate_action] = player_id
            value = value_function[candidate_state.tobytes()]
            if value > max_value:
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
    pass

class Human(Agent):
    def get_action(state, past_state, value_function):
        return human_input(state), value_function

class Random(Agent):
    def get_action(state, past_state, value_function):
        return random.choice(get_available_actions(state)), value_function

class Greedy(Agent):
    def __init__(self, step_size=0.001):
        self.step_size = step_size

    def get_action(state, past_state, value_function):
        if past_state is not None:
            # temporal difference: update value of current state
            value_function[past_state.tobytes()] = value_function[past_state.tobytes()] + (self.step_size * (value_function[state.tobytes()] - value_function[past_state.tobytes()])) # TODO ensure this is correct
        return greedy_action, value_function

class E_greedy(Greedy):
    def __init__(self, e=0.01):
        self.epsilon = e

    def get_action(state, past_state, value_function):
        if random.random() <= self.epsilon:
            return random.choice(get_available_actions(state)), value_function
        else:
            return super().get_action(state, past_state, value_function)

def train_value_function(): 
    value_function = init_value_table()
    for i in range(2):
        value_function = game(Random(), E_greedy(), value_function)
    values = value_function.values()
    print(values)
    print("mean state value = {np.mean(values)}")
    print("std state value = {np.std(values)}")
    import sys
    sys.exit()
    return value_function


def game(x_player, o_player, value_function):
    state = np.zeros((3,3))
    past_state = None
    while not is_terminal(state): 
        move, _ = x_player.get_action(state, past_state, value_function)
        state = get_greedy_action(state, 1, value_function)[1]
        print(state)
        if is_terminal(state):
            break
        move = human_input(state)
        state[move] = 2
        print(state)
    return value_function
    
value_function = train_value_function()
game(E_greedy(), Human(), value_function)
#value_table = init_value_table()
#print(f"number of states (should be 5478): {len(value_table)}")
#print(f"value_table state values: {set(value_table.values())}")
