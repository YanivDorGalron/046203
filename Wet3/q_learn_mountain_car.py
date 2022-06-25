import numpy as np
import time

from data_transformer import DataTransformer
from mountain_car_with_data_collection import MountainCarWithResetEnv
from radial_basis_function_extractor import RadialBasisFunctionExtractor


class Solver:
    def __init__(self, number_of_kernels_per_dim, number_of_actions, gamma, learning_rate):
        # Set max value for normalization of inputs
        self._max_normal = 1
        # get state \action information
        self.data_transformer = DataTransformer()
        state_mean = [-3.00283763e-01,  5.61618575e-05]
        state_std = [0.51981243, 0.04024895]
        self.data_transformer.set(state_mean, state_std)
        self._actions = number_of_actions
        # create RBF features:
        self.feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
        self.number_of_features = self.feature_extractor.get_number_of_features()
        # the weights of the q learner
        self.theta = np.random.uniform(-0.001, 0, size=number_of_actions * self.number_of_features)
        # discount factor for the solver
        self.gamma = gamma
        self.learning_rate = learning_rate

    def _normalize_state(self, s):
        return self.data_transformer.transform_states(np.array([s]))[0]

    def get_features(self, state):
        normalized_state = self._normalize_state(state)
        features = self.feature_extractor.encode_states_with_radial_basis_functions([normalized_state])[0]
        return features

    def get_q_val(self, features, action):
        theta_ = self.theta[action*self.number_of_features: (1 + action)*self.number_of_features]
        return np.dot(features, theta_)

    def get_all_q_vals(self, features):
        all_vals = np.zeros(self._actions)
        for a in range(self._actions):
            all_vals[a] = solver.get_q_val(features, a)
        return all_vals

    def get_max_action(self, state):
        sparse_features = solver.get_features(state)
        q_vals = solver.get_all_q_vals(sparse_features)
        return np.argmax(q_vals)

    def get_state_action_features(self, state, action):
        state_features = self.get_features(state)
        all_features = np.zeros(len(state_features) * self._actions)
        all_features[action * len(state_features): (1 + action) * len(state_features)] = state_features
        return all_features

    def update_theta(self, state, action, reward, next_state, done):
        # compute the new weights and set in self.theta. also return the bellman error (for tracking).
        assert False, "implement update_theta"
        return 0.0


def modify_reward(reward):
    reward -= 1
    if reward == 0.:
        reward = 100.
    return reward


def run_episode(env, solver, is_train=True, epsilon=None, max_steps=200, render=False):
    episode_gain = 0
    deltas = []
    if is_train:
        start_position = np.random.uniform(env.min_position, env.goal_position - 0.01)
        start_velocity = np.random.uniform(-env.max_speed, env.max_speed)
    else:
        start_position = -0.5
        start_velocity = np.random.uniform(-env.max_speed / 100., env.max_speed / 100.)
    state = env.reset_specific(start_position, start_velocity)
    step = 0
    if render:
        env.render()
        time.sleep(0.1)
    while True:
        if epsilon is not None and np.random.uniform() < epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            action = solver.get_max_action(state)
        if render:
            env.render()
            time.sleep(0.1)
        next_state, reward, done, _ = env.step(action)
        reward = modify_reward(reward)
        step += 1
        episode_gain += reward
        if is_train:
            deltas.append(solver.update_theta(state, action, reward, next_state, done))
        if done or step == max_steps:
            return episode_gain, np.mean(deltas)
        state = next_state


if __name__ == "__main__":
    env = MountainCarWithResetEnv()
    seed = 123
    # seed = 234
    # seed = 345
    np.random.seed(seed)
    env.seed(seed)

    gamma = 0.99
    learning_rate = 0.01
    epsilon_current = 0.1
    epsilon_decrease = 1.
    epsilon_min = 0.05

    max_episodes = 100000

    solver = Solver(
        # learning parameters
        gamma=gamma, learning_rate=learning_rate,
        # feature extraction parameters
        number_of_kernels_per_dim=[7, 5],
        # env dependencies (DO NOT CHANGE):
        number_of_actions=env.action_space.n,
    )

    for episode_index in range(1, max_episodes + 1):
        episode_gain, mean_delta = run_episode(env, solver, is_train=True, epsilon=epsilon_current)

        # reduce epsilon if required
        epsilon_current *= epsilon_decrease
        epsilon_current = max(epsilon_current, epsilon_min)

        print(f'after {episode_index}, reward = {episode_gain}, epsilon {epsilon_current}, average error {mean_delta}')

        # termination condition:
        if episode_index % 10 == 9:
            test_gains = [run_episode(env, solver, is_train=False, epsilon=0.)[0] for _ in range(10)]
            mean_test_gain = np.mean(test_gains)
            print(f'tested 10 episodes: mean gain is {mean_test_gain}')
            if mean_test_gain >= -75.:
                print(f'solved in {episode_index} episodes')
                break

    run_episode(env, solver, is_train=False, render=True)
