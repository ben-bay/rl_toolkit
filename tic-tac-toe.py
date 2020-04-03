import random
import numpy as np

# X = 1
# O = 2
# . = 0

step_size = 0.0001
policy = {}
control_player = 1
experiment_player = 2
experiment_wins = 0
experiment_losses = 0
interactive = True
epsilon = 0.1

def myprint(string):
    if interactive:
        print(string)

def human_input(state):
    row = input("\nrow: ")
    col = input("\ncol: ")
    while int(row) > 2 or int(row) < 0 or int(col) > 2 or int(col) < 0 or state[int(row), int(col)] != 0:
        print("Invalid!")
        row = input("\nrow: ")
        col = input("\ncol: ")

    return int(row), int(col)

def agent_action(state, agent, player_id, past_state=None):
    add_new_state(state) # just in case
    if agent == "random":
        action = random.randint(0,2), random.randint(0,2)
        while state[action] != 0:
            action = random.randint(0,2), random.randint(0,2)
        return action
    elif agent == "human":
        return human_input(state)
    else:
        greedy_action, next_state = get_greedy_action(state, player_id)
        add_new_state(past_state) # just in case
        if agent == "e-greedy":
            if random.random() <= epsilon:
                return random.choice(get_available_actions(state)) #TODO can this be the highest value action?
        if past_state is not None:
            # temporal difference: update value of current state
            policy[past_state.tobytes()] = policy[past_state.tobytes()] + (step_size * (policy[next_state.tobytes()] - policy[past_state.tobytes()]))
        return greedy_action

def get_greedy_action(state, player_id):
        candidate_actions = get_available_actions(state)
        #print(f"candidate_actions:\n{candidate_actions}\n")
        max_value = -np.inf
        max_value_action = random.choice(candidate_actions)
        max_value_state = np.copy(state)
        for candidate_action in candidate_actions:
            candidate_state = np.copy(state)
            candidate_state[candidate_action] = player_id
            add_new_state(candidate_state)
            value = policy[candidate_state.tobytes()]
            if value > max_value:
                max_value = value
                max_value_action = candidate_action
                max_value_state = candidate_state
        #print(f"max_value_action: {max_value_action}")
        #print(f"max_value_state:\n{max_value_state}\n")
        return max_value_action, max_value_state

def add_new_state(state):
    if state is None:
        return
    if is_win(state, experiment_player):
        policy[state.tobytes()] = 1.0
    elif is_win(state, control_player):
        print("win for control player")
        policy[state.tobytes()] = 0.0
    elif state.tobytes() not in policy.keys():
        policy[state.tobytes()] = 0.5

def action(square, action, pre_state):
    result_state = np.copy(pre_state)
    result_state[square] = action
    return result_state 

def get_available_actions(state):
    return list(zip(list(np.where(state==0)[0]) ,list(np.where(state==0)[1])))

def render(state):
    if interactive:
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
    return is_win(state, 1) or is_win(state, 2) or (np.sum(np.where(state == 0)) == 0)

def is_win(state, _id):
    if state[0,0] == _id and state[1,1] == _id and state[2,2] == _id:
        return True

    if state[0,2] == _id and state[1,1] == _id and state[2,0] == _id:
        return True
    
    return any((state[:]==[_id,_id,_id]).all(1)) or any((state.T[:]==[_id,_id,_id]).all(1))


def game(control_agent, experiment_agent):
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

    state = np.zeros((3,3))
    history = {}
    game_over = False
    time = 0
    history[time] = state
    while not game_over:
        time += 1
        state = action(agent_action(state, first_agent, first_turn), first_turn, state)
        #state = action(human_input(state), control_player, state)
        history[time] = state
        render(state)
        if is_terminal_state(state):
            break
        time += 1
        #state = action(agent_action(state, "random", experiment_player), experiment_player, state)
        state = action(agent_action(state, second_agent, second_turn, past_state=history[time-2]), second_turn, state)
        history[time] = state
        render(state)
        game_over = is_terminal_state(state)
    return state, is_win(state, experiment_player)

state = np.zeros((3,3))
policy[state.tobytes()] = 0.5
N_GAMES = 10000
for game_id in range(N_GAMES):
    myprint(f"\nGAME {game_id+1}")
    final_state, exp_won = game("random", "e-greedy")
    if exp_won:
        experiment_wins += 1
    else:
        experiment_losses += 1
    myprint("GAME OVER!")
    myprint(f"Winrate: {experiment_wins/(experiment_wins+experiment_losses)}")
vals = list(policy.values())
print(vals)
print(f"max val: {np.max(vals)}")
print(f"min val: {np.min(vals)}")
print(f"mean val: {np.mean(vals)}")
print(f"median val: {np.median(vals)}")
print(len(policy))
print(f"Winrate: {experiment_wins/(experiment_wins+experiment_losses)}")

game_id = 0
experiment_wins = 0
experiment_losses = 0
while True:
    myprint(f"\nGAME {game_id+1}")
    final_state, exp_won = game("human", "e-greedy")
    if exp_won:
        experiment_wins += 1
    else:
        experiment_losses += 1
    myprint("GAME OVER!")
    myprint(f"Winrate: {experiment_wins/(experiment_wins+experiment_losses)}")
