import numpy as np
import time


class GamePlayer:
    def __init__(self, env, data_transformer, feature_extractor, policy):
        self.env = env
        self.data_transformer = data_transformer
        self.feature_extractor = feature_extractor
        self.policy = policy

    def _process_single_state(self, state):
        state = np.expand_dims(state, axis=0)
        state = self.data_transformer.transform_states(state)
        state = self.feature_extractor.encode_states_with_radial_basis_functions(state)
        return state

    def play_game(self, max_steps_per_game, exploration_probability=0.0, render=False, start_state=None):
        if start_state is None:
            current_state = self.env.reset()
        else:
            starting_position, starting_velocity = start_state
            current_state = self.env.reset_specific(starting_position, starting_velocity)
        if render:
            self.env.render()
        done = False
        for _ in range(max_steps_per_game):
            encoded_state = self._process_single_state(current_state)
            use_random = np.random.uniform() < exploration_probability
            if use_random:
                # get a random action
                action = np.random.choice(self.policy.number_of_actions)
            else:
                # get maximal action
                action = self.policy.get_max_action(encoded_state)
                action = int(action[0])
            current_state, _, done, _ = self.env.step(action)
            if render:
                self.env.render()
                time.sleep(0.01)
            if done:
                break
        return done

    def play_games(self, number_of_games, max_steps_per_game):
        all_results = [self.play_game(max_steps_per_game) for _ in range(number_of_games)]
        success_rate = np.mean(all_results)
        print(f'success rate is {success_rate}')
        return success_rate
