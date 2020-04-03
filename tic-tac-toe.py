import random
import numpy as np

# X = 1
# O = 2
# . = 0

step_size = 0.1
states = {}
control_player = 1
experiment_player = 2

"""
state = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        for action in range(3):
            state[i,j] = action
            states.add(np.copy(state))
states.add()
"""


def human_input(state):
    row = input("\nrow: ")
    col = input("\ncol: ")
    while state[int(row), int(col)] != 0:
        print("Invalid!")
        row = input("\nrow: ")
        col = input("\ncol: ")

    return int(row), int(col)

def agent_action(state, agent, symbol):
    if agent == "random":
        action = random.randint(0,2), random.randint(0,2)
        while state[action] != 0:
            action = random.randint(0,2), random.randint(0,2)
        return action
    elif agent == "greedy":
        greedy_action, next_state = get_greedy_action(state, symbol)
        add_new_state(state) # just in case
        # temporal difference: update value of current state
        states[state.tobytes()] = states[state.tobytes()] + (step_size * (states[next_state.tobytes()] - states[state.tobytes()]))
        return greedy_action
    elif agent == "e-greedy":
        greedy_action = get_greedy_action(state, symbol)
        if random.random() > 0.1: # TODO add e parameter
            return greedy_action
        else:
            return random.choice(get_available_actions(state)) #TODO can this be the highest value action?

def get_greedy_action(state, symbol):
        candidate_actions = get_available_actions(state)
        #print(f"candidate_actions:\n{candidate_actions}\n")
        max_value = -np.inf
        max_value_action = random.choice(candidate_actions)
        max_value_state = np.copy(state)
        for candidate_action in candidate_actions:
            candidate_state = np.copy(state)
            candidate_state[candidate_action] = symbol
            add_new_state(candidate_state)
            value = states[candidate_state.tobytes()]
            if value > max_value:
                max_value = value
                max_value_action = candidate_action
                max_value_state = candidate_state
        #print(f"max_value_action: {max_value_action}")
        #print(f"max_value_state:\n{max_value_state}\n")
        return max_value_action, max_value_state

def add_new_state(state):
    if is_win(state, experiment_player):
        states[state.tobytes()] = 1.0
    elif is_win(state, control_player):
        states[state.tobytes()] = 0.0
    elif state.tobytes() not in states:
        states[state.tobytes()] = 0.5

def action(square, action, pre_state):
    result_state = np.copy(pre_state)
    result_state[square] = action
    #states[result_state.tobytes()] = 0.5
    return result_state 

def get_available_actions(state):
    return list(zip(list(np.where(state==0)[0]) ,list(np.where(state==0)[1])))

def render(state):
    i = 0
    for cell in np.nditer(state.T):
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
    
def is_terminal_state(state):
    #TODO diagonals
    #print(state)
    return is_win(state, 1) or is_win(state, 2) or (np.sum(np.where(state == 0)) == 0)

def is_win(state, _id):
    diags = list(zip(list(np.where(np.identity(3)==1)[0]), list(np.where(np.identity(3)==1)[1])))
    diag_win = True
    for diag in diags:
        if state[diag] != _id:
            diag_win = False
            break
    if diag_win:
        return True
    diags = list(zip(list(np.where(np.rot90(np.identity(3))==1)[0]), list(np.where(np.rot90(np.identity(3))==1)[1])))
    diag_win = True
    for diag in diags:
        if state[diag] != _id:
            diag_win = False
            break
    if diag_win:
        return True
    
    return any((state[:]==[_id,_id,_id]).all(1)) or any((state.T[:]==[_id,_id,_id]).all(1))

state = np.zeros((3,3))
states[state.tobytes()] = 0.5
N_GAMES = 100
for game in range(N_GAMES):
    print(f"GAME {game+1}")
    state = np.zeros((3,3))
    game_over = False
    while not game_over:
        state = action(agent_action(state, "random", control_player), control_player, state)
        #state = action(human_input(state), control_player, state)
        render(state)
        if is_terminal_state(state):
            break
        state = action(agent_action(state, "greedy", experiment_player), experiment_player, state)
        render(state)
        game_over = is_terminal_state(state)
    print("GAME OVER!")
print(states.values())
