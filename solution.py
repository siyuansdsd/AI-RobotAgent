import sys
import time
from constants import *
from environment import *
from state import State
import matplotlib.pyplot as plt
import numpy as np

"""
solution.py

This file is a template you should use to implement your solution.

You should implement code for each of the TODO sections below.

COMP3702 2022 Assignment 3 Support Code

Last updated by njc 12/10/22
"""

ALPHA = 0.1

class RLAgent:

    #
    # TODO: (optional) Define any constants you require here.
    #

    def __init__(self, environment: Environment, alpha):
        self.environment = environment
        #
        # TODO: (optional) Define any class instance variables you require (e.g. Q-value tables) here.
        #
        self.discount = self.environment.gamma
        self.learning_rate = alpha
        # self.learning_rate = self.environment.alpha
        self.Q_table = dict()
        self.initial_s = self.environment.get_init_state()
        self.Q_es = []
        self.Q_re = []
        self.S_es = []
        self.S_re = []

    # === Q-learning ===================================================================================================
    def q_learn_train(self):
        """
        Train this RL agent via Q-Learning.
        """
        #
        # TODO: Implement your Q-learning training loop here.
        #
        max_loop = 10000
        # for i in range(max_loop):
        old_table = {"mam": 231313131}
        new_table = {"sdsdsd": 0}
        rewards_list = []
        x_list = []
        while not self.if_converge(new_table, old_table):
            current_state = self.initial_s
            old_table = self.Q_table.copy()
            total_reward = 0
            while current_state.robot_posit not in self.environment.target_list:
                action = self.e_choose_action(current_state)
                reward, new_state = self.environment.perform_action(current_state, action)
                # if self.Q_table.get((current_state, action)) is None:
                #     self.Q_table[(current_state, action)] = reward
                best_next = self._get_best_action(new_state)
                best_next_reward = self.Q_table.get((new_state, best_next), 0)
                if new_state.robot_posit in self.environment.target_list:
                    best_next_reward = 120
                target = reward + self.discount * best_next_reward
                new_reward = self.Q_table.get((current_state, action), 0) + self.learning_rate * (target - self.Q_table.get((
                    current_state,
                    action),
                    0))
                self.Q_table[(current_state, action)] = new_reward
                current_state = new_state
                total_reward += reward
            new_table = self.Q_table.copy()
            rewards_list.append(total_reward)
        print(rewards_list)
        for i in range(len(rewards_list)):
            x_list.append(i)
        rewards_list2 = []
        for i in range(len(rewards_list)):
            sum = 0
            if i + 1 < 50:
                for x in range(i + 1):
                    sum += rewards_list[x]
                num = sum / (i + 1)
                rewards_list2.append(num)
            else:
                for x in range(50):
                    sum += rewards_list[i - x]
                num = sum / 50
                rewards_list2.append(num)

        self.Q_es = x_list
        self.Q_re = rewards_list2

        # x = x_list
        # y = rewards_list2
        # plt.title('Q_learning')
        # plt.xlabel('es')
        # plt.ylabel('v')
        # plt.plot(x, y)
        # plt.show()

    def q_learn_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via Q-learning.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        #
        # TODO: Implement code to return an approximately optimal action for the given state (based on your learned
        #  Q-learning Q-values) here.
        #
        return self._get_best_action(state)

    # === SARSA ========================================================================================================

    def sarsa_train(self):
        """
        Train this RL agent via SARSA.
        """
        #
        # TODO: Implement your SARSA training loop here.
        #
        old_table = {"mam": 231313131}
        new_table = {"sdsdsd": 0}
        rewards_list = []
        x_list = []
        while not self.if_converge_sar(new_table, old_table):
            current_state = self.initial_s
            old_table = self.Q_table.copy()
            action = None

            total_reward = 0
            while current_state.robot_posit not in self.environment.target_list:
                if action is None:
                    action = self.e_choose_action(current_state)
                reward, new_state = self.environment.perform_action(current_state, action)
                # if self.Q_table.get((current_state, action)) is None:
                #     self.Q_table[(current_state, action)] = reward
                best_next = self.e_choose_action(new_state)
                best_next_reward = self.Q_table.get((new_state, best_next), 0)
                if new_state.robot_posit in self.environment.target_list:
                    best_next_reward = 120
                target = reward + self.discount * best_next_reward
                new_reward = self.Q_table.get((current_state, action), 0) + self.learning_rate * (target - self.Q_table.get((
                    current_state,
                    action),
                    0))
                self.Q_table[(current_state, action)] = new_reward
                current_state = new_state
                action = best_next
                total_reward += reward
            new_table = self.Q_table.copy()
            rewards_list.append(total_reward)
        print(rewards_list)
        for i in range(len(rewards_list)):
            x_list.append(i)
        rewards_list2 = []
        for i in range(len(rewards_list)):
            sum = 0
            if i + 1 < 50:
                for x in range(i + 1):
                    sum += rewards_list[x]
                num = sum / (i + 1)
                rewards_list2.append(num)
            else:
                for x in range(50):
                    sum += rewards_list[i - x]
                num = sum / 50
                rewards_list2.append(num)
        self.S_es = x_list
        self.S_re = rewards_list2
        # x = x_list
        # y = rewards_list2
        # plt.title('sarse')
        # plt.xlabel('es')
        # plt.ylabel('v')
        # plt.plot(x, y)
        # plt.show()

    def sarsa_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via SARSA.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        #
        # TODO: Implement code to return an approximately optimal action for the given state (based on your learned
        #  SARSA Q-values) here.
        #
        return self._get_best_action(state)

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: (optional) Add any additional methods here.
    #
    #
    def e_choose_action(self, state):
        current_state = state
        best_action = self._get_best_action(state)
        if best_action is None or random.random() < ALPHA:
            return random.choice(ROBOT_ACTIONS)
        return best_action

    def _get_best_action(self, state):
        best_q = float('-inf')
        best_a = None
        for action in ROBOT_ACTIONS:
            this_q = self.Q_table.get((state, action))
            if this_q is not None and this_q > best_q:
                best_q = this_q
                best_a = action
        return best_a

    def if_converge(self, dict_a: dict, dict_b: dict):
        max = 0.05
        for ak, av in dict_a.items():
            if dict_b.get(ak) is not None:
                if abs(dict_b.get(ak) - av) > max:
                    return False
            else:
                return False
        return True

    def if_converge_sar(self, dict_a: dict, dict_b: dict):
        max = 0.05
        for ak, av in dict_a.items():
            if dict_b.get(ak) is not None:
                if abs(dict_b.get(ak) - av) > max:
                    return False
            else:
                return False
        return True

    def comp(self, Dict: dict):
        t_v = 0
        i = 1
        for k, v in dict:
            i += 1
            t_v += v
            if i == 100:
                break
        return t_v / i


def main():
    Env1 = Environment("testcases/ex3.txt")
    Env2 = Environment("testcases/ex4.txt")
    # Env3 = Environment("testcases/ex4.txt")
    ag1 = RLAgent(Env1, 0.02)
    ag2 = RLAgent(Env2, 0.02)
    # ag3 = RLAgent(Env3, 0.02)
    ag1.q_learn_train()
    ag2.sarsa_train()
    # ag3.q_learn_train()
    # ag3.q_learn_train()
    # x = x_list
    # y = rewards_list2
    plt.title('compare different algorithm')
    plt.xlabel('episodes')
    plt.ylabel('AVG_rewards')
    line1, = plt.plot(ag1.Q_es, ag1.Q_re, color='tab:red', label='Q_learning')
    line2, = plt.plot(ag2.S_es, ag2.S_re, color='tab:blue', label='Sarsa')
    # line3, = plt.plot(ag3.Q_es, ag3.Q_re, color='tab:orange', label='graph when Alpha = 0.02')
    plt.xlim(0, 5000)
    plt.ylim(-1000, 0)
    plt.legend(handles=[line1, line2])
    plt.show()


main()
