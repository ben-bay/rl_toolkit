import gym
import gym_tictactoe

interactive = True

def myprint(string):
    if interactive:
        print(string)

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
    
    time = 0
    env = gym.make('tictactoe-v0')
    
    while not game_over:
        time += 1
        action = get_agent_action(state, first_agent, first_turn)
        state = env.step(action)
        render(state)
        if is_terminal_state(state):
            break
    return state, is_win(state, experiment_player)

def main():
    myprint(f"\nGAME {game_id+1}")
    final_state, exp_won = game("random", "e-greedy")
    if exp_won:
        experiment_wins += 1
    else:
        experiment_losses += 1
    myprint("GAME OVER!")
    myprint(f"Winrate: {experiment_wins/(experiment_wins+experiment_losses)}")

if __name__ == "__main__":
    sys.exit(main())
