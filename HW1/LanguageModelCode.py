import numpy as np


def mostProbableWord(P, letters, K):
    costs = -np.log(P)
    n_letters = len(letters)
    cumulative_cost = np.zeros((n_letters, K + 1))
    actions = np.zeros((n_letters, K + 1))
    for stage in reversed(range(0, K)):
        for state in range(n_letters - 1):  # avoid last letter
            if stage == K - 1:  # final stage - only last letter is legal
                cumulative_cost[state, stage] = cumulative_cost[n_letters - 1, stage + 1] + costs[state, n_letters - 1]
                actions[state, stage] = n_letters - 1
            else:
                possible = range(n_letters - 1)  # intermediate stage - last letter is illegal
                cumulative_cost[state, stage] = np.min(cumulative_cost[possible, stage + 1] + costs[state, possible])
                actions[state, stage] = np.argmin(cumulative_cost[possible, stage + 1] + costs[state, possible])
    # Now we have the chosen actions for each state in each stage, use this to find optimal path:
    actions = actions.astype(int)
    optimal_path = np.zeros(K, dtype=int)
    for i in range(1, K):
        optimal_path[i] = actions[optimal_path[i - 1], i - 1]
    word = letters[optimal_path]
    word[0] = word[0].upper()
    return ''.join(word)


# Q3:
P = np.array([[0.1, 0.325, 0.25, 0.325], [0.4, 0, 0.4, 0.2], [0.2, 0.2, 0.2, 0.4], [1, 0, 0, 0]])
# first letter in letters is the must be first letter  and last letter in letters is the end token
letters = np.array(['b', 'k', 'o', '-'])
probable_word = mostProbableWord(P, letters, 5)
print('Q3: length K = ', 5, probable_word)
