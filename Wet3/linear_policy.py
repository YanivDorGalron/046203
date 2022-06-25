import numpy as np


class LinearPolicy:
    def __init__(self, number_of_state_features, number_of_actions, include_bias=True):
        self.features_per_action = number_of_state_features + int(include_bias)
        self.number_of_actions = number_of_actions
        self.include_bias = include_bias
        all_features = number_of_actions * self.features_per_action
        self.w = np.zeros((all_features, 1))

    def set_w(self, w):
        assert self.w.shape == w.shape
        change = np.linalg.norm(w - self.w)
        print(f'changed w, norm diff is {change}')
        self.w = w
        return change

    def get_max_action(self, encoded_states):
        all_q_values = []
        for a in range(self.number_of_actions):
            a_vec = a * np.ones((len(encoded_states)), np.int32)
            q_value = self.get_q_values(encoded_states, a_vec)
            all_q_values.append(q_value)
        all_q_values = np.concatenate(all_q_values, axis=1)
        max_action = np.argmax(all_q_values, axis=1)
        return max_action

    def get_q_values(self, encoded_states, action_vector):
        q_features = self.get_q_features(encoded_states, action_vector)
        return np.dot(q_features, self.w)

    def get_q_features(self, encoded_states, actions):
        number_of_states = len(encoded_states)
        # init an empty array
        all_features = np.zeros((number_of_states, self.number_of_actions, self.features_per_action), np.float64)
        # set bias if needed
        if self.include_bias:
            encoded_states = np.concatenate((encoded_states, np.ones((number_of_states, 1), np.float64)), axis=1)
        for i in range(number_of_states):
            current_action = actions[i]
            all_features[i][current_action] = encoded_states[i]
        return np.reshape(all_features, (number_of_states, -1))

