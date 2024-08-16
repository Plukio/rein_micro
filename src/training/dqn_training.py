# src/training/dqn_training.py

import time
import numpy as np
from collections import deque
from src.utils.utils import dict2array, get_reward

def dqn(env, agent, n_episodes=2000, max_t=500, eps_start=1.0, eps_end=0, eps_decay=0.994, solved=8000, exp=0):
    start = time.time()
    scores = []
    scores_window = deque(maxlen=100)
    list_eps = []
    eps = eps_start
    n_amount_list = []
    w_amount_list = []
    yield_list = []

    k1 = 0.158
    k2 = 0.79
    k3 = 1.1

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        state = dict2array(state)
        score = 0
        n_amount = 0
        w_amount = 0
        
        for t in range(max_t):
            action1 = agent.act(state, eps)
            action = {
                'anfer': (action1 % 5) * 40,
                'amir': int(action1 / 5) * 6,
            }
            if state[0] >= 10000:
                action['anfer'] = 0
            if state[21] >= 1600:
                action['amir'] = 0
            
            next_state, reward, done, _ = env.step(action)
            next_state = dict2array(next_state)
            agent.step(state, action1, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                n_amount_list.append(n_amount)
                w_amount_list.append(w_amount)
                yield_list.append(state[4])
                break
            
            n_amount += action['anfer']
            w_amount += action['amir']

        scores_window.append(score)
        scores.append(score)

        if score > 1400 and state[4] > 11000:
            agent.save(str(i_episode))
        
        eps = max(eps_end, eps_decay * eps)
        list_eps.append(eps)
        
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}/{n_episodes} \t Score: {score:.2f}')
        
        if np.mean(scores_window) > solved:
            print(f'Game Solved after {i_episode} episodes')
            break

    time_elapsed = time.time() - start
    print(f"Time Elapse: {time_elapsed:.2f} seconds")
    
    return scores, list_eps, n_amount_list, w_amount_list, yield_list
