import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

result_path = "./Results/"


mu = [0.6, 0.5, 0.3, 0.7, 0.1]
C = [1, 4, 6, 2, 9]
VI_num_iter = 400
TD_0_iterations = 10000

N = 5
num_states = 2**N
St = range(num_states)
St = [int(bin(s)[2:]) for s in St]
St = [[int(d) for d in str(s)[::-1]] for s in St]
S = []
for i in range(num_states):
    s = np.zeros(N, dtype=np.uint8)
    j=0
    for job in St[i]:
        s[j] += job
        j += 1
    S.append(s)


def s_to_ix(s):
    s = s[::-1]
    binary = (''.join(map(str, s)))
    return int(binary, 2)


def finished(s):
    return not s.any()


def cost(s):
    return sum([C[i] for i in range(N) if s[i] == 1])


def next_s(s, a):
    assert not finished(s)
    s_finished_job = s.copy()
    s_finished_job[a] -= 1
    return s_finished_job, s_to_ix(s_finished_job)


####### Part 1 Planning #######

def get_V_by_VI(policy):
    V = np.zeros(num_states)
    for iter in range(VI_num_iter):
        old_V = V
        for s_ix in range(num_states):
            s = S[s_ix]
            if finished(s):
                continue
            job = policy[s_ix]
            #print(s, job)
            assert s[job] == 1
            s_finished_job, s_finished_job_ix = next_s(s, job)

            V[s_ix] = cost(s) + mu[job] * old_V[s_finished_job_ix] + (1 - mu[job]) * old_V[s_ix]

    return V


def max_cost_pi():
    pi = np.zeros(num_states, dtype=np.uint)
    for i in range(num_states):
        s = S[i]
        max_cost = 0
        max_ix = 0
        for job in range(len(s)):
            if s[job] == 1 and C[job] > max_cost:
                max_cost = C[job]
                max_ix = job
        pi[i] = max_ix
    return pi


def cmu_pi():
    pi = np.zeros(num_states, dtype=np.uint)
    for i in range(num_states):
        s = S[i]
        max_cost = 0
        max_ix = 0
        for job in range(len(s)):
            if s[job] == 1 and C[job] * mu[job] > max_cost:
                max_cost = C[job] * mu[job]
                max_ix = job
        pi[i] = max_ix
    return pi

def get_possible_action(s):
    return [job_ix for job_ix, e in enumerate(s) if e != 0]

def PI(initial_policy):

    current_policy = initial_policy
    new_policy = np.zeros(num_states, dtype=np.uint8)
    states_V = get_V_by_VI(current_policy)
    initial_state_value = [states_V[-1]] #last state is [1 1 1 1 1]

    while not (new_policy == current_policy).all():
        current_policy = new_policy.copy()
        for s_ix in range(num_states):
            s = S[s_ix]
            if finished(s):
                continue
            action_list = get_possible_action(s)
            best_a_value = float("inf")
            for a in action_list:
                next, next_ix = next_s(s, a)
                a_value = mu[a] * states_V[next_ix] + (1 - mu[a]) * states_V[s_ix]
                if a_value < best_a_value:
                    best_a_value = a_value
                    new_policy[s_ix] = a
        states_V = get_V_by_VI(new_policy)
        initial_state_value += [states_V[-1]]

    return new_policy, initial_state_value


def pt_c(show=False):
    fig, ax = plt.subplots()
    values = get_V_by_VI(max_cost_pi())
    x = np.arange(num_states)
    ax.plot(x, values, label='Maximal cost policy')
    ax.set_ylabel('State Value')
    ax.set_xlabel('States')
    ax.set_xticks(x)
    ax.set_xticklabels(S, rotation='vertical')
    ax.legend()
    fig.tight_layout()
    plt.savefig(result_path + "q2_c.png")
    if show:
        plt.show()


def pt_d(show=False):
    optimal_policy, initial_state_value = PI(max_cost_pi())
    fig, ax = plt.subplots()
    ax.plot(initial_state_value, label='Initial state Value per iteration')
    ax.set_ylabel('Initial State Value')
    ax.set_xlabel('Iteration')
    ax.set_xticks(range(len(initial_state_value)))

    ax.legend()
    fig.tight_layout()
    plt.savefig(result_path + "q2_d.png")
    if show:
        plt.show()


def pt_e(show=False):
    optimal_policy, _ = PI(max_cost_pi())
    optimal_policy_values = get_V_by_VI(optimal_policy)
    max_cost_values = get_V_by_VI(max_cost_pi())
    bug_ix = s_to_ix([1,0,0,0,1])
    print(bug_ix, optimal_policy[bug_ix], cmu_pi()[bug_ix])
    fig, ax = plt.subplots()
    x = np.arange(num_states)
    ax.plot(x, max_cost_values, label='Maximal cost policy values')
    ax.plot(x, optimal_policy_values, label='Optimal policy values')
    ax.set_ylabel('State Value')
    ax.set_xlabel('States')
    ax.set_xticks(x)
    ax.set_xticklabels(S, rotation='vertical')
    ax.set_title('Optimal Pi VS Max Cost Pi')
    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.savefig(result_path + "q2_e.png")
    if show:
        plt.show()

    fig, (ax, ax2) = plt.subplots(2)
    x = np.arange(num_states)
    ax.plot(x, max_cost_values, label='Maximal cost policy values')
    ax.plot(x, optimal_policy_values, label='Optimal policy values')
    ax.set_ylabel('State Value')
    ax.set_xlabel('States')
    ax.set_xticks(x)
    ax.set_xticklabels(S, rotation='vertical')
    ax.set_title('Optimal Pi VS Max Cost Pi')
    ax.legend()
    ax.grid()

    ax2.plot(x, cmu_pi(), 'o', label='CMU policy')
    ax2.plot(x, optimal_policy, 'x', label='Optimal policy')
    ax2.set_ylabel('Policy')
    ax2.set_xlabel('States')
    ax2.set_xticks(x)
    ax2.set_xticklabels(S, rotation='vertical')
    ax2.set_title('Optimal Pi VS cmu rule Pi')
    ax2.legend()

    fig.tight_layout()
    plt.savefig(result_path + "q2_e.png")
    if show:
        plt.show()


def simulator(s, a):
    finished = 1 if random.random() < mu[a] else 0
    next_state = next_s(s, a)[0] if finished else s
    return cost(s), next_state


####### Part 2 Learning #######
###############################


def TD_0(pi, V, real_V, step_size):
    s0_errs = []
    max_norms = []
    visits = np.zeros(num_states)

    for i in range(TD_0_iterations):
        state = S[random.randint(1, num_states-1)]
        while not finished(state):
            s_ix = s_to_ix(state)
            visits[s_ix] += 1
            action = pi[s_ix]
            cost, next_state = simulator(state, action)

            # Now we chose the step-size
            if step_size == 0:
                alpha = 1 / visits[s_ix]
            elif step_size == 1:
                alpha = 0.01
            else:
                alpha = 10 / (100 + visits[s_ix])

            # TD0 update
            V[s_ix] = V[s_ix] + alpha * (cost + V[s_to_ix(next_state)] - V[s_ix])
            state = next_state

        s0_errs.append(abs(real_V[-1] - V[-1]))
        max_norms.append(max([abs(real_V[s] - V[s]) for s in range(num_states)]))

    return V, max_norms, s0_errs


def TD_lambda(pi, V, real_V, step_size, lbda):
    s0_errs = []
    max_norms = []
    visits = np.zeros(num_states)

    for i in range(TD_0_iterations):
        eligibility = np.zeros(num_states)
        state = S[random.randint(1, num_states-1)]
        while not finished(state):
            s_ix = s_to_ix(state)
            visits[s_ix] += 1
            action = pi[s_ix]

            # Eligibility trace updates
            eligibility = eligibility*lbda
            eligibility[s_ix] += 1
            #print('e= ' , eligibility)
            cost, next_state = simulator(state, action)

            # Now we chose the step-size
            if step_size == 0:
                alpha = 1 / visits[s_ix]
            elif step_size == 1:
                alpha = 0.01
            else:
                alpha = 10 / (100 + visits[s_ix])

            td_error = alpha * (cost + V[s_to_ix(next_state)] - V[s_ix])
            #print(eligibility, td_error)
            V += eligibility * td_error
            state = next_state

        s0_errs.append(abs(real_V[-1] - V[-1]))
        max_norms.append(max([abs(real_V[s] - V[s]) for s in range(num_states)]))

    return V, max_norms, s0_errs


def pt_g(show=False):
    policy = max_cost_pi()
    real_vals = get_V_by_VI(policy)
    alphas = ['alpha = 1/n_visits', 'alpha = 0.01', 'alpha = 10/(100+n_visits)']
    for i in range(3):
        initial_V = np.zeros(num_states)
        V_approx, max_norms, s0_errs = TD_0(policy, initial_V, real_vals, i)

        fig, ax = plt.subplots()
        x = np.arange(TD_0_iterations)

        ax.plot(x, max_norms, label='Max Norm Error')
        ax.plot(x, s0_errs, label='Initial State V(s0) error')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Error')
        ax.set_title('TD(0) {}. Error= |V_pi_cost - V_TD|'.format(alphas[i]))
        ax.legend()
        fig.tight_layout()
        plt.savefig(result_path + "q2_g{}.png".format(i))
        if show:
            plt.show()


def pt_h(show=False):
    policy = max_cost_pi()
    real_vals = get_V_by_VI(policy)
    lmbdas = [0.2, 0.4, 0.7, 0.9]
    max_norms_list = []
    s0_errs_list = []
    for lbda in lmbdas:
        max_norms = np.zeros(TD_0_iterations)
        s0_errs = np.zeros(TD_0_iterations)
        for i in range(20):
            initial_V = np.zeros(num_states)
            V_approx, max_norms_i, s0_errs_i = TD_lambda(policy, initial_V, real_vals, step_size=0, lbda=lbda)
            max_norms += max_norms_i
            s0_errs += s0_errs_i
        max_norms_list.append(max_norms / 20)
        s0_errs_list.append(s0_errs / 20)

    fig, ax = plt.subplots()
    x = np.arange(TD_0_iterations)

    ax.plot(x, max_norms_list[0], label='lambda={}'.format(lmbdas[0]))
    ax.plot(x, max_norms_list[1], label='lambda={}'.format(lmbdas[1]))
    ax.plot(x, max_norms_list[2], label='lambda={}'.format(lmbdas[2]))
    ax.plot(x, max_norms_list[3], label='lambda={}'.format(lmbdas[3]))

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Max Norm Error')
    ax.set_title('TD(lambda). Max norm Error= |V_pi_cost - V_TD|')
    ax.legend()
    fig.tight_layout()
    plt.savefig(result_path + "q2_h1.png")

    fig, ax = plt.subplots()
    ax.plot(x, s0_errs_list[0], label='lambda={}'.format(lmbdas[0]))
    ax.plot(x, s0_errs_list[1], label='lambda={}'.format(lmbdas[1]))
    ax.plot(x, s0_errs_list[2], label='lambda={}'.format(lmbdas[2]))
    ax.plot(x, s0_errs_list[3], label='lambda={}'.format(lmbdas[3]))

    ax.set_xlabel('Iterations')
    ax.set_ylabel('V(s0) Error')
    ax.set_title('TD(lambda). V(s0) Error= |V_pi_cost - V_TD|')
    ax.legend()
    fig.tight_layout()
    plt.savefig(result_path + "q2_h2.png")


def epsilon_greedy_pick_action(epsilon, s, q_values):
    if s == 0:
        return 0
    action_ix_list = get_possible_action(S[s])
    l_action_ix_list = len(action_ix_list)
    if l_action_ix_list == 1:
        return action_ix_list[0]
    assert len(action_ix_list) > 0
    if epsilon and np.random.rand() < epsilon:
        return action_ix_list[np.random.randint(0, l_action_ix_list-1)]
    max_q = float("inf")
    max_a = action_ix_list[0]
    for action_ix in action_ix_list:
        if q_values[s][action_ix] < max_q:
            max_a = action_ix
            max_q = q_values[s][action_ix]
    return max_a


def greedy_pick_action(s, q_values):
    return epsilon_greedy_pick_action(0, s, q_values)


def get_greedy_policy_values(q_values):
    greedy_pi = [greedy_pick_action(s, q_values) for s in range(num_states)]
    return get_V_by_VI(greedy_pi)


def greedy_exploration(Q_vals, state, s_ix, eps=0.1):
    if not s_ix:
        return 0
    if random.random() <= eps:
        action = random.choice(get_possible_action(state))
    else:  # greedy
        actions = get_possible_action(state)
        action = actions[np.argmin([Q_vals[s_ix, actions]])]

    return action


def get_Q_greedy_policy_values(Q_vals):
    V = np.ones(num_states) * np.inf
    V[0] = 0
    for s_ix in range(num_states):
        for a in get_possible_action(S[s_ix]):
            if V[s_ix] > Q_vals[s_ix, a]:
                V[s_ix] = Q_vals[s_ix, a]
    return V

Q_learning_iterations = 100000
def Q_learning(epsilon, optimal_V, step_size):
    max_errs = []
    initial_state_errs = []

    Q = np.zeros((num_states, N))
    visits = np.zeros((num_states, N))
    for i in range(Q_learning_iterations):

        min_state_sampling = 1
        state = S[random.randint(min_state_sampling, len(S) - 1)]
        s_ix = s_to_ix(state)

        possible_actions = get_possible_action(state)
        action = random.choice(list(possible_actions))

        visits[s_ix, action] += 1
        cost, next_state = simulator(state, action)
        if 0 == step_size:
            alpha = 1 / visits[s_ix, action]
        elif 1 == step_size:
            alpha = 0.01
        else:
            alpha = 10 / (100 + visits[s_ix, action])

        next_state_ix = s_to_ix(next_state)
        next_action = greedy_exploration(Q, next_state, next_state_ix, epsilon)
        # Q update
        Q[s_ix, action] = Q[s_ix, action] + alpha * (cost + Q[next_state_ix, next_action] - Q[s_ix, action])

        if not i%100:
            print(i)
            # Error computation for every 100 iteration
            greedy_vals = get_Q_greedy_policy_values(Q)
            max_errs.append(max([abs(optimal_V[s_ix] - greedy_vals[s_ix]) for s_ix in range(num_states)]))
            initial_state_errs.append(abs(optimal_V[-1] - greedy_vals[-1]))

    return Q, max_errs, initial_state_errs

def pt_i():
    policy = cmu_pi()
    real_vals = get_V_by_VI(policy)

    alphas = ['alpha = 1/n_visits', 'alpha = 0.01', 'alpha = 10/(100+n_visits)']

    for i in range(3):
        V_approx, max_norms, s0_errs = Q_learning(0.1, real_vals, i)

        fig, ax = plt.subplots()
        x = np.arange(len(max_norms))

        ax.plot(x, max_norms, label='Max Norm Error')
        ax.plot(x, s0_errs, label='Initial State V(s0) error')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Error')
        ax.set_title('Q-learning {}. Error= |V_cmu_pi - V_approx|'.format(alphas[i]))
        ax.legend()
        fig.tight_layout()
        plt.savefig(result_path + "q2_i{}.png".format(i))


def pt_j():
    policy = cmu_pi()
    real_vals = get_V_by_VI(policy)

    alphas = ['alpha = 1/n_visits', 'alpha = 0.01', 'alpha = 10/(100+n_visits)']
    i = 2
    epsilon = 0.01
    V_approx, max_norms, s0_errs = Q_learning(epsilon, real_vals, step_size=i)

    fig, ax = plt.subplots()
    x = np.arange(len(max_norms))

    ax.plot(x, max_norms, label='Max Norm Error')
    ax.plot(x, s0_errs, label='Initial State V(s0) error')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error')
    ax.set_title('Q-learning {}. Error= |V_cmu_pi - V_approx|'.format(alphas[i]))
    ax.legend()
    fig.tight_layout()
    plt.savefig(result_path + "q2_j{}.png".format(i))


if __name__ == '__main__':
    print(S)
    #pt_c(show=False)
    #pt_d()
    #pt_e()
    #pt_g()
    #pt_h()

    # Q learning:
    #pt_i()

    pt_j()