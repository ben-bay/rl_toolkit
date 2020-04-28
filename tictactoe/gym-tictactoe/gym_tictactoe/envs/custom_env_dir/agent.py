import numpy as np
import random


def get_agent_action(state, current_player, policy, epsilon=0.1, past_state=None, step_size=0.0001):
    add_new_state_to_policy(state, policy) # just in case
    if current_player.agent_type == "random":
        action = random.randint(0,2), random.randint(0,2)
        while state[action] != 0:
            action = random.randint(0,2), random.randint(0,2)
        return action
    elif current_player.agent_type == "human":
        return human_input(state)
    else:
        greedy_action, next_state = get_greedy_action(state, current_player, policy)
        add_new_state_to_policy(past_state, policy) # just in case
        if current_player.agent_type == "e-greedy":
            #explore; no learning
            if random.random() <= epsilon:
                return random.choice(get_available_actions(state)) #TODO can this be the highest value action?
        #greedy
        if past_state is not None:
            # learn. temporal difference: update value of past state
            policy[past_state.tobytes()] = policy[past_state.tobytes()] + (step_size * (policy[next_state.tobytes()] - policy[past_state.tobytes()]))
        return greedy_action

def add_new_state_to_policy(state, policy):
    if state is None:
        return
    if is_win(state, 2):
        policy[state.tobytes()] = 1.0
    elif is_win(state, 1):
        print("win for control player")
        policy[state.tobytes()] = 0.0
    elif state.tobytes() not in policy.keys():
        policy[state.tobytes()] = 0.5
    return policy

def is_win(state, _id):
    if state[0,0] == _id and state[1,1] == _id and state[2,2] == _id:
        return True

    if state[0,2] == _id and state[1,1] == _id and state[2,0] == _id:
        return True

    return any((state[:]==[_id,_id,_id]).all(1)) or any((state.T[:]==[_id,_id,_id]).all(1))

def get_greedy_action(state, current_player, policy):
        candidate_actions = get_available_actions(state)
        #print(f"candidate_actions:\n{candidate_actions}\n")
        max_value = -np.inf
        max_value_action = random.choice(candidate_actions)
        max_value_state = np.copy(state)
        for candidate_action in candidate_actions:
            candidate_state = np.copy(state)
            candidate_state[candidate_action] = current_player.id
            add_new_state_to_policy(candidate_state, policy)
            value = policy[candidate_state.tobytes()]
            if value > max_value:
                max_value = value
                max_value_action = candidate_action
                max_value_state = candidate_state
        #print(f"max_value_action: {max_value_action}")
        #print(f"max_value_state:\n{max_value_state}\n")
        return max_value_action, max_value_state

def action(square, action, pre_state):
    result_state = np.copy(pre_state)
    result_state[square] = action
    return result_state

def get_available_actions(state):
    return list(zip(list(np.where(state==0)[0]) ,list(np.where(state==0)[1])))
