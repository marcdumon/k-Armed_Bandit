# --------------------------------------------------------------------------------------------------------
# 2020/12/03
# src - k_bandit.py
# md
# --------------------------------------------------------------------------------------------------------
import time

import numpy as np
from numba import int64, float64, jit, prange
from numba.experimental import jitclass

# Implementation of the n-Armed bandit Problem
# To roughly assess the effectiveness of the greedy and epsilon-greedy methods.
# From "Reinforcement Learning - An Introduction" by Richard Stutton and Andrew Barton

spec1 = [('k', int64),
         ('mean', float64),
         ('std', float64),
         ('noise_std', float64),
         ('action_values', float64[:]),
         ('optimal_action', int64),
         ('arm', int64)]


@jitclass(spec1)
class Testbed_Environment:
    def __init__(self, k=10, mean=0., std=1., noise_std=1.):
        """
            Testbed_Environment with k-armed bandit, where each arm returns a random action value from a normal distribution with mean=mean and std=std.
            Which arm should be chosen to obtain the maximum reward?

            :param k: Number of arms in the Testbed_Environment
            :param mean: Mean of the normal-distributed rewards
            :param std: Standard deviation of the normal-distributed rewards
            :param reward_std: Standard deviation of the noise term of the actual reward
        """
        self.k = k
        self.mean = mean
        self.std = std
        self.noise_std = noise_std
        self.action_values = np.zeros(k)  # the action values q∗(a), a = 1, . . . , 10
        self.optimal_action = 0

    def pull_arm(self, arm):
        reward = np.random.normal(self.action_values[arm], self.noise_std)  # normal dist with mean=Q*(at) and std=1
        return reward

    def reset(self):
        self.action_values = np.random.normal(self.mean, self.std, self.k)
        self.optimal_action = np.argmax(self.action_values)


spec2 = [('n_plays', int64),
         ('epsilon', float64),
         ('action_count', float64[:]),  # np.zeros has type float
         ('environment', Testbed_Environment.class_type.instance_type),
         ('action_value_estimate', float64[:]),
         ('action_total', float64[:]),
         ('reward_history', float64[:]),
         ('optimal_action_history', float64[:])]


@jitclass(spec2)
class Agent:
    def __init__(self, environment, n_plays=2000, epsilon=0.):
        """

        :param environment: A Testbed_Environment instance
        :param n_plays: Number of times to play the same Testbed_Environment
        :param epsilon: The probability to select a random action in stead of the action with maximum action value. Set to 0 for greedy actions.
        """
        self.environment = environment
        self.n_plays = n_plays
        self.epsilon = epsilon
        self.action_count = np.zeros(environment.k)
        self.action_value_estimate = np.zeros(environment.k)  # the estimate of the value of each action
        self.action_total = np.zeros(environment.k)  # total rewards for each action

        self.reward_history = np.zeros(n_plays)  # rewards for n_plays timesteps, for  chart 1
        self.optimal_action_history = np.zeros(n_plays)  # timesteps when optimal action was selected, for chart 2

    def play(self):
        # Play the same bandit (environment) for n_plays time steps
        self.environment.reset()
        for p in range(self.n_plays):
            rand = np.random.random()  # random float in [0-1]
            if rand < self.epsilon:  # epsilon method
                action = np.random.choice(len(self.action_value_estimate))  # random action in [0-k]
            else:  # greedy method
                max_action_value = np.max(self.action_value_estimate)
                # if more than one max_action_value in action_values, select a random one.
                actions = np.where(self.action_value_estimate == max_action_value)[0]
                if len(actions) == 1:
                    action = np.argmax(self.action_value_estimate)
                else:
                    action = np.random.choice(actions)  # randomly choose one action

            reward = self.environment.pull_arm(arm=action)
            self.action_count[action] += 1
            self.action_total[action] += reward
            self.action_value_estimate[action] = self.action_total[action] / self.action_count[action]

            # Update histories
            self.reward_history[p] = reward
            self.optimal_action_history[p] = int(action == self.environment.optimal_action)


@jit('UniTuple(float64[:], 2)(int64, float64, float64, float64, int64, int64, float64)',
     locals={'reward_history': float64[:], 'n_time_steps': int64,
             'testbed_env': Testbed_Environment.class_type.instance_type,
             'agent': Agent.class_type.instance_type},
     nopython=True, cache=True, parallel=False, nogil=True, fastmath=True)  # Todo: parallel is slower !!!
def run_experiment(k=10, mean=0., std=1., noise_std=1., n_runs=2000, n_time_steps=1000, epsilon=0.):
    reward_history = np.zeros(n_time_steps)
    optimal_action_history = np.zeros(n_time_steps)
    for run in range(n_runs):  # change range to prange for parallel
        testbed_env = Testbed_Environment(k, mean, std, noise_std)  # Don't use keyword arguments, Bug in Numba jitclass. See: https://github.com/numba/numba/issues/4495
        agent = Agent(testbed_env, n_time_steps, epsilon)
        agent.play()
        reward_history += agent.reward_history
        optimal_action_history += agent.optimal_action_history
    return reward_history, optimal_action_history


if __name__ == '__main__':
    t0 = time.time()
    # k = 10
    # mean = 0.
    # std = 1.
    # noise_std = 1.
    # n_runs = 2000
    # n_time_steps = 1000
    # epsilon = 0
    # reward_hist, optimal_action_hist = run_experiment(k, mean, std, noise_std, n_runs, n_time_steps, epsilon)
    print(time.time() - t0)
