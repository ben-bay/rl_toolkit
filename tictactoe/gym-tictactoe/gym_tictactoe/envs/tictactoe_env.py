import gym
from gym import error, spaces, utils
from gym.utils import seeding

class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human','fast']}

    def __init__(self):
        self.time = 0
        self.state_history[time] = np.zeros((3,3))
        self.game_over = False

    def step(self, action):
	self.time += 1
	new_state = do_action(action, 1, self.state_history[self.time])
        self.state_history[self.time] = new_state 
	if is_terminal_state(new_state):
            return
        self.time += 1
        #state = action(get_agent_action(self.state_history[self.time], "random", experiment_player), experiment_player, state)
        new_state = do_action(get_agent_action(self.state_history[self.time], second_agent, second_turn, past_state=self.state_history[self.time-2]), second_turn, state)
        self.state_history[self.time] = new_state
	return new_state

    def reset(self):
        self.time = 0
        self.state_history[time] = np.zeros((3,3))
        self.game_over = False

    def render(self, mode='human', close=False):
        if interactive:
            i = 0
            for cell in np.nditer(self.history[self.time]):
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

    while not game_over:
        time += 1
        state = action(agent_action(state, first_agent, first_turn), first_turn, state)
        #state = action(human_input(state), control_player, state)
        history[time] = state
        render(state)
        if is_terminal_state(state):
            break
        time += 1
        #state = action(get_agent_action(state, "random", experiment_player), experiment_player, state)
        state = action(get_agent_action(state, second_agent, second_turn, past_state=self.state_history[self.time-1]), second_turn, state)
        history[time] = state
        render(state)
        game_over = is_terminal_state(state)
    return state, is_win(state, experiment_player)


def do_action(square, player_id, pre_state):
    result_state = np.copy(pre_state)
    result_state[square] = player_id 
    return result_state


def get_agent_action(state, agent, player_id, past_state=None):
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

def is_terminal_state(state):
    return is_win(state, 1) or is_win(state, 2) or (np.sum(np.where(state == 0)) == 0)

def is_win(state, _id):
    if state[0,0] == _id and state[1,1] == _id and state[2,2] == _id:
        return True

    if state[0,2] == _id and state[1,1] == _id and state[2,0] == _id:
        return True

    return any((state[:]==[_id,_id,_id]).all(1)) or any((state.T[:]==[_id,_id,_id]).all(1))
