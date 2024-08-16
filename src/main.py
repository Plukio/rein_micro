# src/main.py

import torch
from src.environment.environment_setup import setup_environment
from src.agents.agent import Agent
from src.training.dqn_training import dqn
from src.utils.utils import plot_results, save_to_excel

def main():

    env = setup_environment()
    agent = Agent(state_size=25, action_size=25)
    scores, list_eps, n_amount_list, w_amount_list, yield_list = dqn(env, agent, n_episodes=3000, exp=1)
    plot_results(scores, list_eps, n_amount_list, w_amount_list, yield_list, 'outputs/combined_plots_distilbert.pdf')
    save_to_excel(scores, list_eps, n_amount_list, w_amount_list, yield_list, 'outputs/data_distilbert.xlsx')

if __name__ == "__main__":
    main()
