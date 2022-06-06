import numpy as np
import matplotlib.pyplot as plt


class Cmu():
    def __init__(self, probs, costs):
        self.probs = probs
        self.costs = costs
        self.N = len(costs)
        self.n_states = 2 ** self.N

    def _get_binary_state(self, state):
        """
        Converts a state to a binary representation
        :param state: int represented state
        :return: binary representation of state
        """
        b_str = np.binary_repr(state, self.N)
        b_list = [int(c) for c in b_str]
        return np.asarray(b_list)

    def _calc_state_cost(self, state_b):
        """
        This function will implement the total cost function
        :param state_b: binary representation of the state
        :return: the cost of the state
        """
        total_cost = 0
        for i in range(self.N):
            if state_b[i]:
                # this means that the i'th job is still waiting in the queue
                total_cost += self.costs[i]
        return total_cost

    def _bellman_equation_one_step(self, action, curr_val, state):
        """
        Calculates one step of the bellman equation given a current value function, action and state. This can be used
        in the policy evaluation and policy iteration for example.
        :param action: The action for the bellman equation
        :param curr_val: The value function (represented as a vector)
        :param state: The state (represented as int)
        :return: The value function calculated from the bellman equation for this state and policy
        """
        next_state_val_exp = 0
        b_start_state = self._get_binary_state(state)
        job = action
        assert b_start_state[job] != 0, "Invalid policy, you can only choose an unfinished job"
        next_state_1 = b_start_state.copy()
        next_state_2 = b_start_state.copy()
        next_state_2[job] = 0
        next_state_1_int = next_state_1.dot(1 << np.arange(next_state_1.shape[-1] - 1, -1, -1))
        next_state_2_int = next_state_2.dot(1 << np.arange(next_state_2.shape[-1] - 1, -1, -1))

        next_state_val_exp += curr_val[next_state_1_int] * (1 - self.probs[job])
        next_state_val_exp += curr_val[next_state_2_int] * self.probs[job]
        return -self._calc_state_cost(self._get_binary_state(state)) + next_state_val_exp

    def policy_eval(self, policy):
        """
        This function will evaluate the value function for a fixed (and given) policy
        :param policy: a vector of length self.n_states, corresponding to the given policy
        :return: a vector of length self.n_states, corresponding to the value function
        """
        curr_val = np.zeros(self.n_states)
        next_val = np.zeros(self.n_states)
        while True:
            for i in range(1, self.n_states):
                next_val[i] = self._bellman_equation_one_step(policy[i], curr_val, i)
            if np.array_equal(curr_val, next_val):
                return curr_val
            curr_val = next_val.copy()

    def get_cost_policy(self):
        """
        This will calculate the policy with the rule argmax(c) over the left jobs
        :return: The calculated policy
        """
        policy = np.zeros(self.n_states, dtype=np.uint8)
        for i in range(self.n_states):
            b_state = list(self._get_binary_state(i))
            max_c_job = 0
            max_c = 0
            for job in range(self.N):
                if b_state[job] == 1:
                    # we can only choose from the unfinished jobs
                    if self.costs[job] > max_c:
                        max_c = costs[job]
                        max_c_job = job
            policy[i] = max_c_job
        return policy

    def plot_value(self, value_function, symbol, color):
        """
        Simple plot of the value function
        :param value_function: the value function to plot
        :param symbol: symbol for the plot
        :param color: color for the plot
        :return: None
        """
        x = list(range(self.n_states))
        plt.plot(x, value_function, symbol, color=color)
        # plt.title("Value function of the 'cost only' policy")
        plt.xlabel("State [decimal representation]")
        plt.ylabel("Value funciton")

    def plot_policy(self, policy, symbol, color):
        """
        Simple plot of the policy
        :param policy: the policy to plot
        :param symbol: symbol for the plot
        :param color: color for the plot
        :return: None
        """
        x = list(range(self.n_states))
        plt.plot(x, policy + 1, symbol, color=color)
        plt.xlabel("State [decimal representation]")
        plt.ylabel("Policy")


if __name__ == '__main__':
    probs = [0.6, 0.5, 0.3, 0.7, 0.1]
    costs = [1, 4, 6, 2, 9]
    Cmu = Cmu(probs, costs)
    result_path = "./Results/"

    # Planning section:

    # Cost only:
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    cost_only_policy = Cmu.get_cost_policy()
    cost_only_value = Cmu.policy_eval(cost_only_policy)
    Cmu.plot_value(cost_only_value, 'o', 'b')
    plt.subplot(122)
    Cmu.plot_policy(cost_only_policy, 'o', 'b')
    plt.savefig(result_path + "Q2_c" + ".png")
