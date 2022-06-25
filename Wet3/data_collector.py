import numpy as np


class DataCollector:
    def __init__(self, env_with_reset):
        self._env_with_reset = env_with_reset

    def state_selection(self):
        position = np.random.uniform(self._env_with_reset.min_position, self._env_with_reset.max_position)
        speed = np.random.uniform(-self._env_with_reset.max_speed, self._env_with_reset.max_speed)
        return position, speed

    def action_selection(self):
        return np.random.choice(3)

    def collect_data(self, number_of_samples):
        # result should be (s_t, a_t, r_t, s_{t+1})
        result = []
        for _ in range(number_of_samples):
            state = self.state_selection()
            self._env_with_reset.reset_specific(state[0], state[1])
            action = self.action_selection()
            next_state, reward, done, _ = self._env_with_reset.step(action)
            result_tuple = (state, action, reward, next_state, done)
            result.append(result_tuple)
        return self.process_data(result)

    @staticmethod
    def process_data(data):
        states, actions, rewards, next_states, done_flags = zip(*data)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        done_flags = np.array(done_flags)
        return states, actions, rewards, next_states, done_flags
