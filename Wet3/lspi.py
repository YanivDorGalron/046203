import numpy as np
from tqdm import tqdm

from mountain_car_with_data_collection import MountainCarWithResetEnv
from data_collector import DataCollector
from data_transformer import DataTransformer
from radial_basis_function_extractor import RadialBasisFunctionExtractor
from linear_policy import LinearPolicy
from game_player import GamePlayer
import pickle
import matplotlib.pyplot as plt


def compute_lspi_iteration(encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma):
    q_features = linear_policy.get_q_features(encoded_states, actions)
    policy_actions = linear_policy.get_max_action(encoded_next_states)
    greedy_q_features = linear_policy.get_q_features(encoded_next_states, policy_actions)
    A = q_features.T @ (q_features - gamma * (greedy_q_features.T * ~done_flags).T) / len(rewards)
    b = q_features.T @ np.expand_dims(rewards, axis=1) / len(rewards)
    next_w = np.linalg.inv(A) @ b
    return next_w


def plot_mean_success_rate(seed_list):
    save_path = './Results/'
    min_len = 1000
    summed_p = np.zeros(min_len)
    for seed in seed_list:
        with open(save_path + 'perf' + str(seed) + '.pickle', 'rb') as handle:
            p = pickle.load(handle)
            if len(p) < min_len:
                min_len = len(p)
            summed_p[:min_len] += p[:min_len]
    avg = summed_p[:min_len] / 5
    plt.figure()
    plt.plot(avg)
    plt.xlabel("iteration")
    plt.ylabel("average success rate")
    plt.savefig(save_path + "Q2_iter.png")


def main_lspi(seed, w_updates=20, samples_to_collect=100000, evaluation_number_of_games=1,
              evaluation_max_steps_per_game=200, thresh=0.00001, only_final=False):
    save_path = './Results/'
    np.random.seed(seed)
    number_of_kernels_per_dim = [12, 10]
    gamma = 0.999
    env = MountainCarWithResetEnv()
    # collect data
    states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
    # get data success rate
    data_success_rate = np.sum(rewards) / len(rewards)
    print('success rate: {}'.format(data_success_rate))
    # standardize data
    data_transformer = DataTransformer()
    data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
    states = data_transformer.transform_states(states)
    next_states = data_transformer.transform_states(next_states)
    # process with radial basis functions
    feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
    # encode all states:
    encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)
    encoded_next_states = feature_extractor.encode_states_with_radial_basis_functions(next_states)
    # set a new linear policy
    linear_policy = LinearPolicy(feature_extractor.get_number_of_features(), 3, True)
    # but set the weights as random
    linear_policy.set_w(np.random.uniform(size=linear_policy.w.shape))
    # start an object that evaluates the success rate over time
    evaluator = GamePlayer(env, data_transformer, feature_extractor, linear_policy)

    # success_rate = evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)
    # print("Initial success rate: {}".format(success_rate))
    performances = []
    if not only_final:
        performances.append(evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game))
    read = False
    if read:
        with open(save_path + 'weight.pickle', 'rb') as handle:
            new_w = pickle.load(handle)
            linear_policy.set_w(np.expand_dims(new_w, 1))
    for lspi_iteration in range(w_updates):
        print('starting lspi iteration {}'.format(lspi_iteration))
        new_w = compute_lspi_iteration(
            encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
        )
        with open(save_path + 'weight.pickle', 'wb') as handle:
            pickle.dump(new_w, handle, protocol=pickle.HIGHEST_PROTOCOL)

        norm_diff = linear_policy.set_w(new_w)
        if not only_final:
            performances.append(evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game))
        if norm_diff < thresh:
            break
    print('done lspi')
    if not only_final:
        with open(save_path + 'perf' + str(seed) + '.pickle', 'wb') as handle:
            pickle.dump(performances, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if only_final:
        score = evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)
        with open(save_path + 'final_perf' + str(samples_to_collect) + '.pickle', 'wb') as handle:
            pickle.dump(score, handle, protocol=pickle.HIGHEST_PROTOCOL)
    evaluator.play_game(evaluation_max_steps_per_game, render=True)


def plot_final_scores(num_samples):
    plt.figure()
    finals = []
    save_path = './Results/'
    for sample in num_samples:
        with open(save_path + 'final_perf' + str(int(sample)) + '.pickle', 'rb') as handle:
            score = pickle.load(handle)
            finals.append(score)
    plt.plot(num_samples, finals)
    plt.xlabel("Number of samples")
    plt.ylabel("Final success rate")
    plt.savefig(save_path + "Q2_samples.png")


def run_and_plot_mean_for_seeds(seed_list):
    for seed in seed_list:
        main_lspi(seed, samples_to_collect=100000, w_updates=10, thresh=0, evaluation_number_of_games=10)

    plot_mean_success_rate(seed_list)


def run_samples_graph(num_samples):
    for sample in tqdm(num_samples):
        main_lspi(0, samples_to_collect=int(sample), w_updates=20, evaluation_number_of_games=20, only_final=True)
    plot_final_scores(num_samples)


if __name__ == '__main__':
    seed_list = np.arange(0, 5)
    run_and_plot_mean_for_seeds(seed_list)

    num_samples = np.linspace(1000, 500000, 30)
    run_samples_graph(num_samples)
