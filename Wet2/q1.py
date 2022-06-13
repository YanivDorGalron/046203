import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import pickle


class BlackJack:
    def __init__(self):
        self.value_probs = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 4, 1]) / 13
        self.card_values = np.asarray([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        self.n_xs = 17
        self.n_ys = 10
        self.n_states = self.n_xs * self.n_ys + 1

    def plot_value_func(self, value_func, fig):
        ax = fig.gca(projection='3d')

        X = np.arange(4, 21, 1)
        Y = np.arange(2, 12, 1)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros(list(np.shape(X)))
        for i in range(np.shape(X)[0]):
            for j in range(np.shape(X)[1]):
                Z[i, j] = value_func[self.x_y_to_ind(X[i, j], Y[i, j])]

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_xlabel('Player sum of card')
        ax.set_ylabel('Dealers showing')
        ax.set_zlabel('value function')

    def plot_policy(self, policy):
        max_hit_policy = np.full(self.n_ys, self.n_xs)
        for y in range(self.n_ys):
            for x in range(self.n_xs):
                if policy[self.x_y_to_ind(x + 4, y + 2)] == 0:
                    max_hit_policy[y] = x
                    break
        plt.plot(list(range(2, 2 + self.n_ys)), max_hit_policy + 4)
        plt.xticks(list(range(2, 2 + self.n_ys)), list(range(2, 2 + self.n_ys)))
        plt.ylim(top=20, bottom=4)
        plt.xlim(right=11, left=2)
        plt.fill_between(list(range(2, 2 + self.n_ys)), max_hit_policy + 4, np.zeros(self.n_ys), color='g')
        plt.fill_between(list(range(2, 2 + self.n_ys)), max_hit_policy + 4, np.full(self.n_ys, self.n_xs + 5),
                         color='r')
        plt.text(6, 8, "Hit", size=20)
        plt.text(6, 18, "Stick", size=20)

    def ind_to_x_y(self, state):
        y = int(state / self.n_xs + 2)
        x = state - (y - 2) * self.n_xs + 4
        return x, y

    def x_y_to_ind(self, x, y):
        if x >= 21:
            return 171
        return (y - 2) * self.n_xs + x - 4

    def get_dealer_prob(self, y, stopy):
        if y >= 17:  # The dealer stops
            if stopy > 21:  # the dealer is losing in this scenario
                return y > 21
            return y == stopy
        tot_prob = 0
        for i, card_val in enumerate(self.card_values):
            tot_prob += self.value_probs[i] * self.get_dealer_prob(y + card_val, stopy)
        return tot_prob

    def get_r_exp_on_stick(self, state):
        x, y = self.ind_to_x_y(state)
        chance_win_stick = 0
        for y_stop in range(y + 2, x):
            chance_win_stick += self.get_dealer_prob(y, y_stop)
        chance_win_stick += self.get_dealer_prob(y, 22)

        chance_draw_stick = self.get_dealer_prob(y, x)
        r_exp_stick = chance_win_stick - (1 - chance_win_stick - chance_draw_stick)
        return r_exp_stick

    def get_r_exp_on_hit(self, state, curr_value):
        exp = 0
        x, y = self.ind_to_x_y(state)
        for i, card in enumerate(self.card_values):
            next_x = x + card
            next_state = self.x_y_to_ind(next_x, y)
            if next_x > 21:
                exp += -1 * self.value_probs[i]
            elif next_x == 21:
                exp += 1 * self.value_probs[i]
            else:
                exp += curr_value[next_state] * self.value_probs[i]
        return exp

    def value_iteration(self, max_num_iter):
        curr_value = np.zeros(self.n_states)
        curr_policy = np.zeros(self.n_states)
        next_value = np.zeros(self.n_states)

        iteration = 1
        while iteration <= max_num_iter:
            iteration += 1
            for s in range(self.n_states - 1):
                r_exp_stick = self.get_r_exp_on_stick(s)
                r_exp_hit = self.get_r_exp_on_hit(s, curr_value)
                curr_policy[s] = int(r_exp_hit >= r_exp_stick)
                next_value[s] = np.max([r_exp_stick, r_exp_hit])
            if np.array_equal(curr_value, next_value):
                return curr_value, curr_policy
            curr_value = next_value.copy()
        return curr_value, curr_policy


def create_pixel_files(directory, optimal_value, optim_policy):
    with open(directory + 'value_function.pickle', 'wb') as handle:
        pickle.dump(optimal_value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(directory + 'policy.pickle', 'wb') as handle:
        pickle.dump(optim_policy, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    results_dir = "Results/"
    game = BlackJack()

    # creating value_function and policy pickle files
    optimal_val, optimal_policy = game.value_iteration(50)
    # create_pixel_files(results_dir, optimal_val, optimal_policy)

    # section b
    fig = plt.figure()
    with open(results_dir + 'value_function.pickle', 'rb') as handle:
        value = pickle.load(handle)
        game.plot_value_func(value, fig)
        # plt.show()
        plt.savefig(results_dir + 'Q1_value_function.png')

    # section c
    fig = plt.figure()
    with open(results_dir + 'policy.pickle', 'rb') as handle:
        policy = pickle.load(handle)
        game.plot_policy(policy)
        plt.savefig(results_dir + 'Q1_policy.png')
