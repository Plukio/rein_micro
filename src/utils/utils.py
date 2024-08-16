# src/utils/utils.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def dict2array(state):
    new_state = []
    for key in state.keys():
        if key != 'sw':
            new_state.append(state[key])
        else:
            new_state += list(state['sw'])        
    return np.asarray(new_state)

def array2str(state):
    state_str = ""
    for i, num in enumerate(state):
        state_str += str(round(num)) + " "
    return state_str

def get_reward(state, n_action, w_action, next_state, done, k1, k2, k3, k4):
    if done:
        reward = k1 * state[4] - k2 * n_action - k3 * w_action
    else:
        reward = -k2 * n_action - k3 * w_action
    return reward

def plot_results(scores, list_eps, n_amount_list, w_amount_list, yield_list, output_file):
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(scores)
    axs[0, 0].set_title('Scores')
    axs[0, 1].plot(list_eps)
    axs[0, 1].set_title('List Eps')
    axs[1, 0].plot(n_amount_list)
    axs[1, 0].set_title('N Amount List')
    axs[1, 1].plot(w_amount_list)
    axs[1, 1].set_title('W Amount List')
    axs[2, 0].plot(yield_list)
    axs[2, 0].set_title('Yield Amount List')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def save_to_excel(scores, list_eps, n_amount_list, w_amount_list, yield_list, output_file):
    df = pd.DataFrame({
        'Scores': scores,
        'List_eps': list_eps,
        'N_amount_list': n_amount_list,
        'W_amount_list': w_amount_list,
        'Yield_list': yield_list
    })
    df.to_excel(output_file, index=False)
